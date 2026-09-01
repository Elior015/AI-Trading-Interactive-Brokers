// Live state stream for the topbar. Per-page detail refreshes on navigation
// (this is a server-rendered dashboard, not an SPA) — the WebSocket keeps the
// bits that matter every second: connection health and the kill switch,
// specifically the "last update" clock so a dead stream never looks alive.
(function () {
  const token = new URLSearchParams(location.search).get("token") || "";

  // Every internal link loses ?token= on click since it's a plain href;
  // carry it forward so navigating between pages doesn't re-prompt for it.
  if (token) {
    document.querySelectorAll('a[href^="/"]').forEach((a) => {
      const url = new URL(a.href, location.href);
      if (!url.searchParams.has("token")) {
        url.searchParams.set("token", token);
        a.href = url.toString();
      }
    });
  }

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

    const modeAutoBtn = document.getElementById("mode-auto-btn");
    const modeManualBtn = document.getElementById("mode-manual-btn");
    if (modeAutoBtn && modeManualBtn && s.execution_mode) {
      modeAutoBtn.classList.toggle("mode-active", s.execution_mode === "auto");
      modeManualBtn.classList.toggle("mode-active", s.execution_mode === "manual");
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

    // "What's happening now" card, positions, and recent cycles — server-rendered
    // once on page load, kept live here so the Overview page never needs a
    // manual reload to reflect the next decision cycle.
    if (s.session) set("session-line", s.session);
    if (s.focus) set("focus-list", s.focus.length ? s.focus.join(", ") : "none yet");

    const decisionEl = document.getElementById("last-decision");
    if (decisionEl) {
      if (s.last_decision) {
        const d = s.last_decision;
        const actionable = (d.proposals || []).filter((p) => p.action !== "HOLD");
        let html = "<p><b>Tape read:</b> " + escapeHtml(d.market_read || "—") + "</p>";
        html += actionable.length
          ? actionable
              .map(
                (p) =>
                  "<p>&bull; <b>" + escapeHtml(p.symbol) + "</b> — " + escapeHtml(p.action) +
                  " — " + escapeHtml(p.rationale || "") + "</p>"
              )
              .join("")
          : '<p class="muted">No trade proposals this cycle.</p>';
        decisionEl.innerHTML = html;
      } else {
        decisionEl.innerHTML = '<p class="muted">No decision cycle has run yet.</p>';
      }
    }

    const planEl = document.getElementById("plan-summary");
    if (planEl) {
      planEl.innerHTML = s.plan
        ? "<p><b>Today's plan (" + escapeHtml(s.plan.bias) + ", " + escapeHtml(s.plan.risk_posture) +
          "):</b> " + escapeHtml(s.plan.reasoning || "") + "</p>"
        : '<p class="muted">No pre-market plan yet.</p>';
    }

    const posBody = document.querySelector("#positions-table tbody");
    if (posBody && s.positions) {
      posBody.innerHTML = s.positions.length
        ? s.positions
            .map((p) => {
              const cls = p.unrealized_pnl >= 0 ? "pos" : "neg";
              return (
                "<tr><td>" + escapeHtml(p.symbol) + "</td><td>" + p.quantity + "</td><td>" +
                p.avg_cost + "</td><td>" + p.market_price + '</td><td class="' + cls + '">' +
                p.unrealized_pnl + "</td></tr>"
              );
            })
            .join("")
        : '<tr><td colspan="5" class="muted">No open positions</td></tr>';
    }

    const pendCard = document.getElementById("pending-approvals-card");
    const pendList = document.getElementById("pending-approvals-list");
    if (pendCard && pendList) {
      const pending = s.pending_approvals || [];
      pendCard.style.display = pending.length ? "block" : "none";
      pendList.innerHTML = pending
        .map((p) => {
          const word = p.kind === "close" ? "Close" : p.action === "SELL" ? "Sell" : "Buy";
          const amount = p.sized
            ? p.sized.quantity + " shares at $" + Number(p.sized.entry_price).toFixed(2)
            : p.close_quantity + " shares";
          return (
            '<div class="approval-row"><div><b>' + escapeHtml(word) + " " + escapeHtml(p.symbol) +
            "</b> — " + escapeHtml(amount) + '<p class="muted">' + escapeHtml(p.rationale || "") +
            '</p></div><div class="approval-actions">' +
            '<button class="approve-btn" onclick="approveTrade(\'' + p.id + '\')">Approve</button>' +
            '<button class="skip-btn" onclick="skipTrade(\'' + p.id + '\')">Skip</button>' +
            "</div></div>"
          );
        })
        .join("");
    }

    const cyclesBody = document.querySelector("#cycles-table tbody");
    if (cyclesBody && s.cycles) {
      cyclesBody.innerHTML = s.cycles.length
        ? s.cycles
            .slice()
            .reverse()
            .map(
              (c) =>
                "<tr><td>" + escapeHtml(c.cycle_id) + "</td><td>" + escapeHtml(c.started_at) +
                "</td><td>" + (c.focus ? c.focus.length : 0) + "</td><td>" + c.proposals +
                "</td><td>" + c.approved + "</td><td>" + c.rejected + "</td><td>" +
                c.llm_latency_ms + "</td></tr>"
            )
            .join("")
        : '<tr><td colspan="7" class="muted">No cycles yet</td></tr>';
    }
  }

  function escapeHtml(str) {
    return String(str)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");
  }

  function connect() {
    const proto = location.protocol === "https:" ? "wss:" : "ws:";
    const ws = new WebSocket(
      proto + "//" + location.host + "/ws/stream?token=" + encodeURIComponent(token)
    );
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
    await fetch(
      "/control/kill?mode=" + encodeURIComponent(mode) + "&token=" + encodeURIComponent(token),
      { method: "POST" }
    );
  };

  window.resetKill = async function () {
    if (!confirm("Reset the kill switch? Trading will resume on the next cycle.")) return;
    await fetch("/control/reset?token=" + encodeURIComponent(token), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ confirm: true }),
    });
  };

  window.setMode = async function (mode) {
    // Switching to auto hands trading back to the AI unattended, so ask
    // first. Switching to manual is the safer direction — no need to ask.
    const needsConfirm = mode === "auto";
    if (needsConfirm && !confirm("Let the AI trade by itself, without asking you first?")) return;
    await fetch("/control/mode?token=" + encodeURIComponent(token), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ mode: mode, confirm: needsConfirm }),
    });
  };

  window.approveTrade = async function (id) {
    await fetch("/control/approve?token=" + encodeURIComponent(token), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ id: id }),
    });
  };

  window.skipTrade = async function (id) {
    await fetch("/control/reject?token=" + encodeURIComponent(token), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ id: id }),
    });
  };
})();
