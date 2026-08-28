"""Token-bucket and historical-data pacing for IBKR's connection-wide limits.

Reconstructed from the test suite and every call site that uses it: the
original module was never actually committed to git (an unanchored `data/`
line in `.gitignore` silently swallowed this whole package alongside the
intended top-level runtime `./data` directory), so no version history exists
to restore it from.

Everything the broker adapter sends to IBKR passes through one of the two
limiters here first:

* `TokenBucket` paces the raw outbound message rate (IBKR closes the socket
  above roughly 50 messages/second).
* `HistoricalDataPacer` paces `reqHistoricalData` calls specifically, which
  IBKR additionally caps at roughly 60 requests per rolling 10 minutes.
* `MarketDataLineBudget` tracks concurrent market-data line usage, which is
  capped per connection independently of message rate.
"""

from __future__ import annotations

import asyncio
import time
from collections import deque

from ..logging_setup import get_logger

log = get_logger(__name__)


class TokenBucket:
    """Classic token bucket: `rate` tokens/sec refill, capped at `capacity`.

    `acquire()` holds a single `asyncio.Lock` for its entire wait, including
    across `asyncio.sleep` calls. That is deliberate rather than an
    oversight: it serializes every concurrent caller through one queue, so
    two coroutines can never both observe "enough tokens" and both consume
    them before either mutation is applied. Given what this bucket guards —
    IBKR closing the socket outright above ~50 msg/sec — correctness under
    concurrency matters more than maximizing throughput here.
    """

    def __init__(self, rate: float, capacity: float | None = None) -> None:
        #: Tokens minted per second.
        self.rate = max(float(rate), 0.001)
        #: Bucket ceiling; also the largest single `acquire(n)` that can ever
        #: succeed without waiting for every token to regenerate from empty.
        self.capacity = float(capacity) if capacity is not None else self.rate
        self._tokens = self.capacity
        self._last = time.monotonic()
        self._lock = asyncio.Lock()

    def _refill_locked(self) -> None:
        now = time.monotonic()
        elapsed = now - self._last
        if elapsed > 0:
            self._tokens = min(self.capacity, self._tokens + elapsed * self.rate)
            self._last = now

    async def acquire(self, n: int = 1) -> None:
        """Block until `n` tokens are available, then consume them atomically."""
        # A request larger than the whole bucket can never be satisfied as
        # asked; treat it as "drain the bucket" rather than waiting forever.
        need = min(float(n), self.capacity)
        async with self._lock:
            while True:
                self._refill_locked()
                if self._tokens >= need:
                    self._tokens -= need
                    return
                deficit = need - self._tokens
                wait = deficit / self.rate
                await asyncio.sleep(max(wait, 0.001))


class HistoricalDataPacer:
    """Paces historical-data requests under IBKR's ~60-per-10-minutes cap.

    Two constraints are enforced together:

    * No more than `capacity` requests in any rolling `window` seconds.
    * At least `min_spacing` seconds between any two individual requests —
      a burst of many small requests the instant the rolling window has room
      is still hard on IBKR's historical-data pipeline even if it stays
      under the raw count cap.
    """

    def __init__(self, capacity: int, window: float = 600.0, min_spacing: float = 1.0) -> None:
        self.capacity = int(capacity)
        self.window = float(window)
        #: Mutable on purpose — callers (and tests) adjust this directly.
        self.min_spacing = float(min_spacing)
        self._timestamps: deque[float] = deque()
        self._last_request: float | None = None
        self._lock = asyncio.Lock()
        #: Multiplies `min_spacing` after `register_violation`; reset never
        #: happens automatically, since a PACING error means our estimate of
        #: what IBKR will tolerate was wrong and should stay corrected.
        self._penalty = 1.0

    def _prune(self, now: float) -> None:
        cutoff = now - self.window
        while self._timestamps and self._timestamps[0] <= cutoff:
            self._timestamps.popleft()

    def _wait_locked(self, now: float) -> float:
        self._prune(now)
        wait = 0.0
        spacing = self.min_spacing * self._penalty
        if self._last_request is not None:
            wait = max(wait, spacing - (now - self._last_request))
        if len(self._timestamps) >= self.capacity:
            # Time until the oldest request in the window ages out of it.
            wait = max(wait, self._timestamps[0] + self.window - now)
        return max(wait, 0.0)

    async def acquire(self, symbol: str = "", bar_size: str = "", what_to_show: str = "") -> None:
        """Block until a historical-data request may be issued.

        `symbol` / `bar_size` / `what_to_show` are accepted for call-site
        symmetry with the request they precede (useful for future per-symbol
        logging) but do not affect pacing: the limit IBKR enforces is on the
        connection as a whole, not per symbol.
        """
        async with self._lock:
            while True:
                now = time.monotonic()
                wait = self._wait_locked(now)
                if wait <= 0:
                    self._timestamps.append(now)
                    self._last_request = now
                    return
                await asyncio.sleep(wait)

    def estimated_wait(self, n: int) -> float:
        """Rough estimate of how long `n` sequential requests will take."""
        now = time.monotonic()
        self._prune(now)
        spacing = self.min_spacing * self._penalty
        remaining_capacity = max(self.capacity - len(self._timestamps), 0)
        if n <= remaining_capacity:
            return max(n - 1, 0) * spacing
        # Requests beyond what the window currently has room for must also
        # wait for old entries to age out, at roughly one slot every
        # `window / capacity` seconds once the bucket is full.
        over = n - remaining_capacity
        per_slot = self.window / max(self.capacity, 1)
        return max(remaining_capacity - 1, 0) * spacing + over * max(per_slot, spacing)

    def utilization(self) -> float:
        """Fraction (0..1) of the rolling window's capacity currently in use."""
        now = time.monotonic()
        self._prune(now)
        if self.capacity <= 0:
            return 0.0
        return min(1.0, len(self._timestamps) / self.capacity)

    def register_violation(self) -> None:
        """Called when IBKR reports a PACING error: back off harder from now on."""
        self._penalty = min(self._penalty * 2.0, 8.0)
        log.warning("historical_pacer_violation_registered", penalty=self._penalty)


class MarketDataLineBudget:
    """Tracks concurrent market-data line usage against IBKR's per-connection cap."""

    def __init__(self, capacity: int) -> None:
        self.capacity = int(capacity)
        self._reserved: dict[str, int] = {}
        self._lock = asyncio.Lock()

    @property
    def used(self) -> int:
        return sum(self._reserved.values())

    @property
    def free(self) -> int:
        return max(self.capacity - self.used, 0)

    async def reserve(self, symbol: str, n: int = 1) -> bool:
        """Reserve `n` lines for `symbol`. Returns False without reserving
        anything if that would exceed `capacity`."""
        async with self._lock:
            if symbol in self._reserved:
                return True  # already holding lines for this symbol
            if self.used + n > self.capacity:
                return False
            self._reserved[symbol] = n
            return True

    async def release(self, symbol: str) -> None:
        """Free whatever was reserved for `symbol`, if anything."""
        async with self._lock:
            self._reserved.pop(symbol, None)
