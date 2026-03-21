/**
 * AlphaTemp — Health page: data freshness, pipeline status, DB stats, config.
 * Fetches /api/health/detailed and renders four panels.
 */

var STALE_THRESHOLD = 30;
var NWS_STALE_THRESHOLD = 1440;

function statusDot(ageMin, threshold) {
  if (ageMin === null) return '<span class="text-gray-300">●</span>';
  if (ageMin <= threshold)
    return '<span class="text-emerald-400" style="box-shadow: 0 0 4px #10b981;">●</span>';
  if (ageMin <= threshold * 2) return '<span class="text-amber-400">●</span>';
  return '<span class="text-red-400">●</span>';
}

function formatAge(ageMin) {
  if (ageMin === null) return "never";
  if (ageMin < 1) return "<1 min ago";
  if (ageMin < 60) return Math.round(ageMin) + " min ago";
  if (ageMin < 1440) return Math.round(ageMin / 60) + "h ago";
  return Math.round(ageMin / 1440) + "d ago";
}

var FRESHNESS_LABELS = {
  observations_synoptic: "Observations (Synoptic)",
  observations_awc: "Observations (AWC/IEM)",
  forecasts_hrrr: "HRRR Forecasts",
  market_ticks: "Market Ticks (Kalshi)",
  drift_signals: "Drift Signals",
  nws_cli: "NWS Settlement (CLI)",
  nws_dsm: "NWS Settlement (DSM)",
};

async function refreshHealth() {
  var data = await fetchAPI("/api/health/detailed");
  if (!data) return;

  // Freshness
  var html = "";
  for (var key in FRESHNESS_LABELS) {
    var f = data.freshness[key] || {};
    var threshold = key.startsWith("nws_")
      ? NWS_STALE_THRESHOLD
      : STALE_THRESHOLD;
    html +=
      '<div class="at-kv-row">' +
      '<span style="color: var(--text-muted);">' +
      FRESHNESS_LABELS[key] +
      "</span>" +
      "<span>" +
      formatAge(f.age_minutes) +
      " " +
      statusDot(f.age_minutes, threshold) +
      "</span>" +
      "</div>";
  }
  document.getElementById("freshness-table").innerHTML = html;

  // Pipelines
  html = "";
  var pipelines = data.pipelines || {};
  var pipelineKeys = Object.keys(pipelines);
  if (pipelineKeys.length === 0) {
    html = '<div class="text-gray-300">No heartbeats received yet</div>';
  } else {
    for (var i = 0; i < pipelineKeys.length; i++) {
      var name = pipelineKeys[i];
      var p = pipelines[name];
      var badge;
      if (p.status === "ok") {
        badge = '<span class="at-pill at-pill-profit">Running</span>';
      } else if (p.error) {
        badge = '<span class="at-pill at-pill-loss">Error</span>';
      } else {
        badge = '<span class="at-pill at-pill-neutral">Idle</span>';
      }
      html +=
        '<div class="at-kv-row">' +
        '<span style="color: var(--text-muted);">' +
        name +
        "</span>" +
        '<span style="display:flex; align-items:center; gap:0.5rem;">' +
        '<span style="color: var(--text-muted);">last cycle: ' +
        p.duration_ms +
        "ms</span>" +
        badge +
        "</span>" +
        "</div>";
    }
  }
  document.getElementById("pipeline-table").innerHTML = html;

  // Database
  var tables = data.database.tables || {};
  var tableKeys = Object.keys(tables);
  if (tableKeys.length === 0) {
    html = '<div style="color: var(--text-muted);">No tables found</div>';
  } else {
    html =
      '<table class="at-table">' +
      "<thead><tr><th>Table</th><th>Rows</th></tr></thead>" +
      "<tbody>";
    for (var tbl in tables) {
      html +=
        "<tr>" +
        "<td>" +
        tbl +
        "</td>" +
        "<td>" +
        tables[tbl].rows.toLocaleString() +
        "</td>" +
        "</tr>";
    }
    html += "</tbody></table>";
  }
  document.getElementById("db-table").innerHTML = html;
  document.getElementById("db-size").textContent =
    "DB size: " + (data.database.size || "unknown");

  // Config
  html = "";
  var cfg = data.config || {};
  var configRows = [
    ["Settlement station", cfg.settlement_station],
    ["Neighbor stations", (cfg.neighbor_stations || []).join(", ")],
    ["Stale threshold", cfg.stale_threshold_min + " min"],
    ["Model", cfg.model],
    ["Bias TTL", cfg.bias_ttl_min + " min"],
  ];
  var intervals = cfg.polling_intervals || {};
  for (var svc in intervals) {
    configRows.push(["Poll: " + svc, intervals[svc] + "s"]);
  }
  for (var j = 0; j < configRows.length; j++) {
    html +=
      '<div class="at-kv-row">' +
      '<span style="color: var(--text-muted);">' +
      configRows[j][0] +
      "</span>" +
      "<span>" +
      configRows[j][1] +
      "</span>" +
      "</div>";
  }
  document.getElementById("config-table").innerHTML = html;
}

// Show loading skeletons before first fetch
showSkeleton("freshness-table", 7);
showSkeleton("pipeline-table", 3);
showSkeleton("db-table", 5);
showSkeleton("config-table", 6);

refreshHealth();
setInterval(refreshHealth, 60000);
