// Live state stream for the topbar. Per-page detail refreshes on navigation
// (this is a server-rendered dashboard, not an SPA) — the WebSocket keeps the
// bits that matter every second: connection health and the kill switch,
// specifically the "last update" clock so a dead stream never looks alive.
(function () {
  const connDot = document.getElementById("conn-dot");
  const connText = document.getElementById("conn-text");
  const modeBadge = document.getElementById("mode-badge");
  const resetBtn = document.getElementById("reset-btn");
  const killBtn = document.getElementById("kill-btn");
  const flattenBtn = document.getElementById("flatten-btn");

  let lastMessageAt = 0;

  function setConn(state, label) {
    if (!connDot) return;
    connDot.className = "dot " + state;
    connText.textContent = label;
  }

  function applySnapshot(s) {
    lastMessageAt = Date.now();

    if (modeBadge) {
      modeBadge.textContent = s.mode;
      modeBadge.className = "badge " + s.mode;
    }

    const gw = s.connection || {};
    const tripped = (s.risk && s.risk.kill_switch && s.risk.kill_switch.tripped) || false;

    if (tripped) {
      setConn("bad", "KILL SWITCH ACTIVE");
    } else if (gw.needs_manual_2fa) {
      setConn("warn", "needs manual 2FA");
    } else if (gw.state === "CONNECTED") {
      setConn("ok", "connected");
    } else {
      setConn("bad", gw.state || "disconnected");
    }

    if (resetBtn) resetBtn.style.display = tripped ? "inline-block" : "none";
    if (killBtn) killBtn.disabled = tripped;
    if (flattenBtn) flattenBtn.disabled = false;

    // Best-effort element updates on the overview page; harmless no-ops elsewhere.
    const set = (id, text) => {
      const el = document.getElementById(id);
      if (el) el.textContent = text;
    };
    if (s.phase) set("phase", s.phase);
    if (s.session) set("session-desc", s.session);
    set("halted", s.halted_reason || "no");
    if (s.account) {
      set("acct-id", s.account.id);
      set("equity", "$" + Number(s.account.equity).toFixed(2));
      set("bp", "$" + Number(s.account.buying_power).toFixed(2));
      set("acct-age", (s.account.age_seconds ?? "-") + "s");
    }
    if (s.pnl) {
      const el = document.getElementById("daypnl");
      if (el) {
        el.textContent = s.pnl.day.toFixed(2) + " (" + s.pnl.day_pct.toFixed(2) + "%)";
        el.className = s.pnl.day >= 0 ? "pos" : "neg";
      }
    }
    if (s.llm) {
      set("llm-strategy", s.llm.strategy);
      set("llm-avail", s.llm.available);
      set("llm-det", s.llm.deterministic_only);
      set("llm-calls", s.llm.total_calls);
    }
    set("gw-state", gw.state || "-");
    set("gw-2fa", gw.needs_manual_2fa);
    set("gw-reconnects", gw.reconnect_attempts);
  }

  function connect() {
    const proto = location.protocol === "https:" ? "wss:" : "ws:";
    const ws = new WebSocket(proto + "//" + location.host + "/ws/stream");
    ws.onmessage = (ev) => {
      try {
        applySnapshot(JSON.parse(ev.data));
      } catch (e) {
        console.error("bad snapshot", e);
      }
    };
    ws.onclose = () => {
      setConn("bad", "disconnected — retrying");
      setTimeout(connect, 2000);
    };
    ws.onerror = () => ws.close();
  }

  // A frozen screen is worse than no dashboard: flag a stale stream loudly.
  setInterval(() => {
    if (lastMessageAt && Date.now() - lastMessageAt > 5000) {
      setConn("warn", "stream stale (" + Math.round((Date.now() - lastMessageAt) / 1000) + "s)");
    }
  }, 1000);

  connect();

  window.killSwitch = async function (mode) {
    if (!confirm("Trip the kill switch (" + mode + ")? This will halt new entries" +
      (mode === "flatten_all" ? " and close every open position." : "."))) {
      return;
    }
    await fetch("/control/kill?mode=" + encodeURIComponent(mode), { method: "POST" });
  };

  window.resetKill = async function () {
    if (!confirm("Reset the kill switch? Trading will resume on the next cycle.")) return;
    await fetch("/control/reset", { method: "POST" });
  };
})();
