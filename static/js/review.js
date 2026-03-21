/**
 * AlphaTemp — Review (Model Autopsy) tab
 *
 * Renders incident cards for model misses and a pattern summary sidebar.
 * Supports date-range and category filtering.
 *
 * Depends on: shared.js (COLORS, fetchAPI, formatPnL, formatEdge, showSkeleton)
 * Depends on: base.html globals (selectedCity, selectedDate)
 */

// -----------------------------------------------------------------------
// State
// -----------------------------------------------------------------------

var reviewRange = "30d";
var reviewFilter = "all";

// Category-to-filter mapping
var CATEGORY_MAP = {
  worst: { label: "Worst", key: "worst_miss" },
  lost: { label: "Lost", key: "lost_bet" },
  missed: { label: "Missed", key: "missed_edge" },
};

// -----------------------------------------------------------------------
// Build Filter Buttons (JS-generated since template has empty divs)
// -----------------------------------------------------------------------

function buildRangeButtons() {
  var el = document.getElementById("range-buttons");
  if (!el) return;
  var ranges = [
    { id: "7d", label: "7d" },
    { id: "30d", label: "30d" },
    { id: "all", label: "All" },
  ];
  var html = "";
  ranges.forEach(function (r) {
    var cls =
      r.id === reviewRange
        ? "at-filter-pill at-filter-pill-active"
        : "at-filter-pill at-filter-pill-inactive";
    html +=
      '<button id="rr-' +
      r.id +
      '" onclick="setReviewRange(\'' +
      r.id +
      '\')" class="' +
      cls +
      '">' +
      r.label +
      "</button>";
  });
  el.innerHTML = html;
}

function buildFilterButtons(incidents) {
  var el = document.getElementById("filter-buttons");
  if (!el) return;

  // Count incidents by filter category
  var counts = { all: 0, worst: 0, lost: 0, missed: 0 };
  if (incidents) {
    counts.all = incidents.length;
    incidents.forEach(function (inc) {
      if (inc.severity >= 0.5) counts.worst++;
      if (inc.type === "lost_bet") counts.lost++;
      if (inc.type === "missed_edge") counts.missed++;
    });
  }

  var filters = [
    { id: "all", label: "All", count: counts.all },
    { id: "worst", label: "Worst", count: counts.worst },
    { id: "lost", label: "Lost", count: counts.lost },
    { id: "missed", label: "Missed", count: counts.missed },
  ];

  // Active style overrides per category
  var activeStyles = {
    worst: "background: var(--loss-bg); color: var(--loss-text);",
    lost: "background: var(--warning-bg); color: var(--warning-text);",
    missed: "background: var(--info-bg); color: var(--info-text);",
  };

  var html = "";
  filters.forEach(function (f) {
    var isActive = f.id === reviewFilter;
    var cls, style;
    if (isActive) {
      cls = "at-filter-pill at-filter-pill-active";
      style = activeStyles[f.id] || "";
    } else {
      cls = "at-filter-pill at-filter-pill-inactive";
      style = "";
    }
    html +=
      '<button id="filter-' + f.id + '" onclick="setFilter(\'' + f.id + "')\"";
    html += ' class="' + cls + '"';
    if (style) html += ' style="' + style + '"';
    html += ">" + f.label;
    if (f.count > 0) {
      html +=
        ' <span style="font-size:9px;opacity:0.8;">' + f.count + "</span>";
    }
    html += "</button>";
  });
  el.innerHTML = html;
}

// -----------------------------------------------------------------------
// Range Toggle
// -----------------------------------------------------------------------

function setReviewRange(r) {
  reviewRange = r;
  buildRangeButtons();
  refreshReview();
}

// -----------------------------------------------------------------------
// Filter Toggle
// -----------------------------------------------------------------------

function setFilter(f) {
  reviewFilter = f;
  // Filter buttons rebuild on data fetch (counts may change with range)
  refreshReview();
}

// -----------------------------------------------------------------------
// Data Fetch & Dispatch
// -----------------------------------------------------------------------

// Cache unfiltered incidents for count badges
var _allIncidents = null;

function refreshReview() {
  Promise.all([
    fetchAPI("/api/review/incidents?range=" + reviewRange + "&filter=all"),
    fetchAPI(
      "/api/review/incidents?range=" + reviewRange + "&filter=" + reviewFilter,
    ),
    fetchAPI("/api/review/patterns?range=" + reviewRange),
  ]).then(function (results) {
    _allIncidents = results[0] ? results[0].incidents : [];
    buildFilterButtons(_allIncidents);
    renderIncidents(results[1]);
    renderPatterns(results[2]);
  });
}

// -----------------------------------------------------------------------
// Render Incidents
// -----------------------------------------------------------------------

function renderIncidents(data) {
  var el = document.getElementById("incident-list");
  if (!el) return;

  if (!data || !data.incidents || data.incidents.length === 0) {
    el.innerHTML =
      '<p style="color: var(--text-muted); font-size: 13px; text-align: center; padding: 40px 0;">' +
      "No incidents in this range. Either the model is perfect or there's no trading data yet." +
      "</p>";
    return;
  }

  var html = "";

  data.incidents.forEach(function (inc) {
    var pnl = formatPnL(inc.pnl);
    var edge = formatEdge(inc.edge);

    // Category pill class
    var categoryPill;
    var categoryLabel;
    if (
      inc.category === "model_miss" ||
      inc.category === "worst_miss" ||
      inc.category === "slow_drift_response" ||
      inc.category === "tail_bracket_underweight"
    ) {
      categoryPill = "at-pill at-pill-loss";
      categoryLabel = "Worst Miss";
    } else if (inc.type === "lost_bet") {
      categoryPill = "at-pill at-pill-warning";
      categoryLabel = "Lost Bet";
    } else if (inc.type === "missed_edge") {
      categoryPill = "at-pill at-pill-info";
      categoryLabel = "Missed Edge";
    } else {
      categoryPill = "at-pill at-pill-neutral";
      categoryLabel = inc.category || "--";
    }

    // Direction pill
    var dirPill;
    if (inc.direction === "YES" || inc.direction === "BUY YES") {
      dirPill = '<span class="at-pill at-pill-info">YES</span>';
    } else if (inc.direction === "NO" || inc.direction === "BUY NO") {
      dirPill = '<span class="at-pill at-pill-orange">NO</span>';
    } else {
      dirPill =
        '<span class="at-pill at-pill-neutral">' +
        (inc.direction || "--") +
        "</span>";
    }

    // P&L color via CSS vars
    var pnlColor = inc.pnl >= 0 ? "var(--profit)" : "var(--loss)";

    html += '<div class="at-card p-4">';

    // Top row: date + category pill | P&L
    html +=
      '<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;">';
    html += '<div style="display:flex;align-items:center;gap:8px;">';
    html +=
      '<span style="font-family:var(--font-data);font-size:11px;color:var(--text-muted);">' +
      (inc.date || "--") +
      "</span>";
    html += '<span class="' + categoryPill + '">' + categoryLabel + "</span>";
    html += "</div>";
    html +=
      '<span style="font-size:16px;font-weight:700;color:' +
      pnlColor +
      ';">' +
      pnl.text +
      "</span>";
    html += "</div>";

    // Middle row: bracket · direction · settled temp
    html +=
      '<div style="display:flex;align-items:center;gap:6px;font-size:11px;color:var(--text-secondary);margin-bottom:10px;">';
    if (inc.bracket) html += "<span>" + inc.bracket + "</span>";
    if (inc.bracket && inc.direction)
      html += '<span style="color:var(--text-muted);">&middot;</span>';
    html += dirPill;
    if (inc.settlement_temp != null) {
      html += '<span style="color:var(--text-muted);">&middot;</span>';
      html += "<span>Settled: " + inc.settlement_temp + "\u00b0F</span>";
    }
    html += "</div>";

    // Metric boxes: Model | Market | Edge
    var modelProb =
      inc.model_prob != null ? (inc.model_prob * 100).toFixed(0) + "%" : "--";
    var marketPrice =
      inc.market_price != null
        ? (inc.market_price * 100).toFixed(0) + "%"
        : "--";
    var edgeBg = inc.edge >= 0 ? "var(--profit-bg)" : "var(--loss-bg)";
    var edgeColor = inc.edge >= 0 ? "var(--profit-text)" : "var(--loss-text)";

    html += '<div style="display:flex;gap:8px;margin-bottom:10px;">';
    // Model
    html +=
      '<div style="flex:1;background:var(--info-bg);border-radius:6px;padding:8px;text-align:center;">';
    html += '<div class="at-card-header">MODEL</div>';
    html +=
      '<div style="font-size:13px;font-weight:700;color:var(--info-text);">' +
      modelProb +
      "</div>";
    html += "</div>";
    // Market
    html +=
      '<div style="flex:1;background:var(--warning-bg);border-radius:6px;padding:8px;text-align:center;">';
    html += '<div class="at-card-header">MARKET</div>';
    html +=
      '<div style="font-size:13px;font-weight:700;color:var(--warning-text);">' +
      marketPrice +
      "</div>";
    html += "</div>";
    // Edge
    html +=
      '<div style="flex:1;background:' +
      edgeBg +
      ';border-radius:6px;padding:8px;text-align:center;">';
    html += '<div class="at-card-header">EDGE</div>';
    html +=
      '<div style="font-size:13px;font-weight:700;color:' +
      edgeColor +
      ';">' +
      edge.text +
      "</div>";
    html += "</div>";
    html += "</div>";

    // Narrative
    if (inc.narrative) {
      html +=
        '<p style="color: var(--text-secondary); font-size: 11px; line-height: 1.4;">' +
        inc.narrative +
        "</p>";
    }

    html += "</div>";
  });

  el.innerHTML = html;
}

// -----------------------------------------------------------------------
// Render Patterns (Sidebar)
// -----------------------------------------------------------------------

function renderPatterns(data) {
  var el = document.getElementById("pattern-summary");
  if (!el) return;

  if (!data || !data.patterns || data.patterns.length === 0) {
    el.innerHTML =
      '<p style="font-size:11px;color:var(--text-muted);">No patterns detected yet</p>';
    return;
  }

  // Category display config
  var categoryConfig = {
    model_miss: { label: "Worst Miss", color: "var(--loss)" },
    worst_miss: { label: "Worst Miss", color: "var(--loss)" },
    slow_drift_response: { label: "Slow Drift", color: "var(--loss)" },
    tail_bracket_underweight: {
      label: "Tail Underweight",
      color: "var(--loss)",
    },
    stale_pricing: { label: "Stale Pricing", color: "var(--loss)" },
    threshold_too_conservative: { label: "Threshold", color: "var(--warning)" },
    lost_bet: { label: "Lost Bet", color: "var(--warning)" },
    missed_edge: { label: "Missed Edge", color: "var(--info)" },
    unknown: { label: "Unknown", color: "var(--text-muted)" },
  };

  var html = "";

  data.patterns.forEach(function (pat) {
    var pnl = formatPnL(pat.total_pnl);
    var cfg = categoryConfig[pat.category] || {
      label: pat.category,
      color: "var(--text-muted)",
    };
    var pnlColor = pat.total_pnl >= 0 ? "var(--profit)" : "var(--loss)";

    html +=
      '<div style="border-left: 3px solid ' +
      cfg.color +
      '; padding-left: 10px; margin-bottom: 14px;">';
    html +=
      '<div style="display:flex;justify-content:space-between;align-items:baseline;">';
    html +=
      '<span class="at-card-header" style="font-weight:600;color:var(--text-primary);">' +
      cfg.label +
      "</span>";
    html +=
      '<span style="font-size:13px;font-weight:700;color:' +
      pnlColor +
      ';">' +
      pnl.text +
      "</span>";
    html += "</div>";
    html +=
      '<div style="font-size:10px;color:var(--text-secondary);margin-top:2px;">' +
      (pat.count || 0) +
      " incidents</div>";
    if (pat.suggested_action) {
      html +=
        '<div style="font-size:10px;color:var(--text-muted);margin-top:4px;line-height:1.3;">' +
        pat.suggested_action +
        "</div>";
    }
    html += "</div>";
  });

  // Key Insight callout — derive from worst pattern
  var worst = data.patterns[0]; // Already sorted by total_pnl ascending (worst first)
  if (worst) {
    var worstCfg = categoryConfig[worst.category] || { label: worst.category };
    var insightText =
      worstCfg.label +
      " accounts for " +
      worst.count +
      " incident" +
      (worst.count !== 1 ? "s" : "") +
      " (" +
      formatPnL(worst.total_pnl).text +
      "). " +
      (worst.suggested_action || "");

    html +=
      '<div style="background:var(--card-bg-alt);border-radius:6px;padding:10px;margin-top:14px;">';
    html +=
      '<div class="at-card-header" style="margin-bottom:4px;">Key Insight</div>';
    html +=
      '<div style="font-size:11px;color:var(--text-primary);line-height:1.4;">' +
      insightText +
      "</div>";
    html += "</div>";
  }

  el.innerHTML = html;
}

// -----------------------------------------------------------------------
// Refresh Hook
// -----------------------------------------------------------------------

function onDateChange() {
  refreshReview();
}

// Build range buttons immediately
buildRangeButtons();

// Show loading skeletons before first fetch
showSkeleton("incident-list", 6);
showSkeleton("pattern-summary", 4);

// Initial load
refreshReview();
