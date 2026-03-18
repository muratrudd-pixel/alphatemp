/**
 * AlphaTemp — Operations tab
 *
 * Renders the temperature curve chart, observation feed,
 * forecast runs feed, and bracket spread panel.
 *
 * Depends on: shared.js (COLORS, PLOTLY_LAYOUT, PLOTLY_CONFIG,
 *             toET, getTargetDate, fetchAPI)
 * Depends on: base.html globals (selectedCity, selectedDate)
 */

// -----------------------------------------------------------------------
// Helpers
// -----------------------------------------------------------------------

/**
 * Convert UTC ISO string to full ET ISO string for Plotly x-axis.
 * The shared toET() returns locale-formatted display text (e.g. "14:30"),
 * which Plotly can't parse as a datetime axis. We need full ISO strings.
 */
function toETIso(utcStr) {
  if (!utcStr) return null;
  var input = utcStr;
  if (
    input.indexOf("Z") === -1 &&
    input.indexOf("+") === -1 &&
    input.indexOf("T") !== -1
  ) {
    input = input + "Z";
  }
  var d = new Date(input);
  if (isNaN(d.getTime())) return null;
  var et = new Date(
    d.toLocaleString("en-US", { timeZone: "America/New_York" }),
  );
  return (
    et.getFullYear() +
    "-" +
    String(et.getMonth() + 1).padStart(2, "0") +
    "-" +
    String(et.getDate()).padStart(2, "0") +
    "T" +
    String(et.getHours()).padStart(2, "0") +
    ":" +
    String(et.getMinutes()).padStart(2, "0") +
    ":" +
    String(et.getSeconds()).padStart(2, "0")
  );
}

/**
 * Current time in ET as an ISO-like string (for "now" line on chart).
 */
function nowETIso() {
  var d = new Date();
  var et = new Date(
    d.toLocaleString("en-US", { timeZone: "America/New_York" }),
  );
  return (
    et.getFullYear() +
    "-" +
    String(et.getMonth() + 1).padStart(2, "0") +
    "-" +
    String(et.getDate()).padStart(2, "0") +
    "T" +
    String(et.getHours()).padStart(2, "0") +
    ":" +
    String(et.getMinutes()).padStart(2, "0") +
    ":" +
    String(et.getSeconds()).padStart(2, "0")
  );
}

// -----------------------------------------------------------------------
// Temperature Curve
// -----------------------------------------------------------------------

function refreshTempChart() {
  var dateParam = selectedDate ? "?date=" + selectedDate : "";
  fetchAPI("/api/forecast-curve/" + selectedCity + dateParam).then(
    function (data) {
      if (!data || !data.forecasts || data.forecasts.length === 0) {
        Plotly.react(
          "temp-curve-chart",
          [],
          Object.assign({}, PLOTLY_LAYOUT, {
            annotations: [
              {
                text: "No forecast data",
                showarrow: false,
                font: { size: 14, color: COLORS.slate400 },
                xref: "paper",
                yref: "paper",
                x: 0.5,
                y: 0.5,
              },
            ],
          }),
          PLOTLY_CONFIG,
        );
        return;
      }

      // Extract arrays
      var fcstX = data.forecasts.map(function (p) {
        return toETIso(p.valid_at);
      });
      var fcstY = data.forecasts.map(function (p) {
        return p.temp_f;
      });
      var obsX = data.observations.map(function (p) {
        return toETIso(p.observed_at);
      });
      var obsY = data.observations.map(function (p) {
        return p.temp_f;
      });

      // --- Traces ---

      // 1. Forecast center line (blue, dotted)
      var traceFcstCenter = {
        x: fcstX,
        y: fcstY,
        type: "scatter",
        mode: "lines",
        line: { color: COLORS.blue, width: 2, dash: "dot" },
        name: "HRRR Forecast",
        hovertemplate: "%{y:.1f}\u00b0F<extra></extra>",
      };

      // 2. Observations (white, lines+markers)
      var traceObs = {
        x: obsX,
        y: obsY,
        type: "scatter",
        mode: "lines+markers",
        line: { color: COLORS.white, width: 2 },
        marker: { size: 4, color: COLORS.white },
        name: "Observed",
        hovertemplate: "%{y:.1f}\u00b0F<extra></extra>",
      };

      // 8. 6-hour synoptic max markers (amber, triangle-up)
      var trace6hMax = null;
      if (data.six_hr_maxes && data.six_hr_maxes.length > 0) {
        trace6hMax = {
          x: data.six_hr_maxes.map(function (p) {
            return toETIso(p.observed_at);
          }),
          y: data.six_hr_maxes.map(function (p) {
            return p.temp_f;
          }),
          type: "scatter",
          mode: "markers",
          marker: {
            size: 7,
            symbol: "triangle-up",
            color: "rgba(148,163,184,0.5)",
          },
          name: "6hr High",
          hovertemplate: "%{y:.1f}\u00b0F<extra>6-hour max</extra>",
        };
      }

      // 9. Settlement marker
      var traceSettlement = null;
      if (data.observed_high != null && data.observed_high_at) {
        var src = data.settlement_source;
        var settleName, settleHover, settleSymbol;
        if (src === "nws_cli") {
          settleName = "NWS Settlement (CLI)";
          settleHover = "NWS Settlement: ";
          settleSymbol = "diamond";
        } else if (src === "dsm") {
          settleName = "Settlement (DSM)";
          settleHover = "DSM Settlement: ";
          settleSymbol = "diamond";
        } else {
          settleName = "Running High (est)";
          settleHover = "Running High: ";
          settleSymbol = "circle";
        }
        traceSettlement = {
          x: [toETIso(data.observed_high_at)],
          y: [data.observed_high],
          type: "scatter",
          mode: "markers",
          marker: {
            size: 14,
            color: COLORS.amber,
            symbol: settleSymbol,
            line: { color: "#ffffff", width: 1.5 },
          },
          name: settleName,
          hovertemplate: settleHover + "%{y:.1f}\u00b0F<extra></extra>",
        };
      }

      // 12. Model prediction band (p25-p75 shaded, median line)
      var traceModelBandUpper = null;
      var traceModelBandLower = null;
      var traceModelMedian = null;
      if (data.model_band && data.model_band.median != null) {
        var mb = data.model_band;
        // Use the full x-range of the chart for horizontal bands
        var bandX = [fcstX[0], fcstX[fcstX.length - 1]];
        // Upper bound (invisible, sets top of fill)
        traceModelBandUpper = {
          x: bandX,
          y: [mb.p75, mb.p75],
          type: "scatter",
          mode: "lines",
          line: { color: "transparent", width: 0 },
          showlegend: false,
          hoverinfo: "skip",
        };
        // Lower bound (fill to upper)
        traceModelBandLower = {
          x: bandX,
          y: [mb.p25, mb.p25],
          type: "scatter",
          mode: "lines",
          line: { color: "transparent", width: 0 },
          fill: "tonexty",
          fillcolor: "rgba(168,85,247,0.20)",
          name: "Model 25-75%",
          hoverinfo: "skip",
        };
        // Median line
        traceModelMedian = {
          x: bandX,
          y: [mb.median, mb.median],
          type: "scatter",
          mode: "lines",
          line: { color: "rgba(168,85,247,0.6)", width: 2, dash: "dashdot" },
          name: "Model Median (" + mb.median + "\u00b0F)",
          hovertemplate:
            "Model Median: " + mb.median + "\u00b0F<extra></extra>",
        };
      }

      // --- Assemble traces ---
      var allTraces = [traceFcstCenter, traceObs];
      if (traceModelBandUpper) allTraces.push(traceModelBandUpper);
      if (traceModelBandLower) allTraces.push(traceModelBandLower);
      if (traceModelMedian) allTraces.push(traceModelMedian);
      if (trace6hMax) allTraces.push(trace6hMax);
      if (traceSettlement) allTraces.push(traceSettlement);

      // --- Layout ---
      var dayStr =
        selectedDate ||
        new Date().toLocaleDateString("en-CA", {
          timeZone: "America/New_York",
        });
      var dayStart = dayStr + "T00:00:00";
      var dayEnd = dayStr + "T23:59:59";

      var nowLine = data.now_utc ? toETIso(data.now_utc) : nowETIso();

      var shapes = [
        {
          type: "line",
          x0: nowLine,
          x1: nowLine,
          y0: 0,
          y1: 1,
          yref: "paper",
          line: { color: "rgba(148,163,184,0.4)", width: 1, dash: "dash" },
        },
      ];

      // Compute y-axis range that includes model band
      var yAxisOpts = Object.assign({}, PLOTLY_LAYOUT.yaxis, {
        title: { text: "\u00b0F", standoff: 8, font: { size: 10 } },
      });
      if (data.model_band && data.model_band.p75 != null) {
        var allYVals = fcstY.concat(obsY);
        allYVals.push(data.model_band.p75);
        allYVals.push(data.model_band.p25);
        if (data.observed_high != null) allYVals.push(data.observed_high);
        var yMin = Math.min.apply(null, allYVals) - 2;
        var yMax = Math.max.apply(null, allYVals) + 2;
        yAxisOpts.range = [yMin, yMax];
        yAxisOpts.autorange = false;
      }

      var layout = Object.assign({}, PLOTLY_LAYOUT, {
        showlegend: true,
        legend: {
          x: 1,
          y: 1,
          xanchor: "right",
          bgcolor: "rgba(15,23,42,0.7)",
          font: { size: 9 },
        },
        yaxis: yAxisOpts,
        xaxis: Object.assign({}, PLOTLY_LAYOUT.xaxis, {
          type: "date",
          range: [dayStart, dayEnd],
          dtick: 7200000,
          tickformat: "%H:%M",
          showgrid: true,
        }),
        shapes: shapes,
      });

      Plotly.react("temp-curve-chart", allTraces, layout, PLOTLY_CONFIG);
    },
  );
}

// -----------------------------------------------------------------------
// Observation Feed
// -----------------------------------------------------------------------

function refreshObsFeed() {
  var dateParam = selectedDate ? "?date=" + selectedDate : "";
  fetchAPI("/api/observations/" + selectedCity + dateParam).then(
    function (data) {
      var container = document.getElementById("obs-feed");
      if (!container) return;

      if (!data || !data.observations || data.observations.length === 0) {
        container.innerHTML =
          '<p class="text-xs text-slate-500">No observations</p>';
        return;
      }

      var obs = data.observations.slice(0, 20);

      // Column header
      var header =
        '<div class="grid grid-cols-[60px_70px_1fr] gap-x-2 text-[10px] text-slate-500 uppercase tracking-wider py-1 border-b border-slate-600">' +
        "<span>Time</span><span>Temp</span><span>Details</span></div>";

      var rows = obs
        .map(function (o) {
          var time = toET(o.observed_at);

          // Build detail badges
          var badges;
          if (o.source && o.source.includes("SPECI")) {
            badges =
              '<span class="px-1.5 py-0.5 rounded border text-[10px] bg-amber-900/50 text-amber-400 border-amber-700">SPECI</span>';
          } else if (o.source === "NWS CLI") {
            badges =
              '<span class="px-1.5 py-0.5 rounded border text-[10px] bg-emerald-900/50 text-emerald-400 border-emerald-700">NWS CLI</span>';
          } else {
            badges = '<span class="text-[10px] text-slate-600">METAR</span>';
          }
          if (o.obs_window === "prior 6hrs") {
            badges +=
              ' <span class="px-1.5 py-0.5 rounded border text-[10px] bg-amber-900/50 text-amber-300 border-amber-700">6hr High</span>';
          }
          if (o.is_new_high === true) {
            badges +=
              ' <span class="px-1.5 py-0.5 rounded border text-[10px] bg-emerald-900/50 text-emerald-300 border-emerald-700 font-bold">NEW HIGH</span>';
          }

          return (
            '<div class="grid grid-cols-[60px_70px_1fr] gap-x-2 text-xs py-1 border-b border-slate-700/50 items-center">' +
            '<span class="text-slate-400">' +
            time +
            "</span>" +
            '<span class="text-slate-200">' +
            o.temp_f.toFixed(1) +
            "\u00b0F</span>" +
            '<span class="flex flex-wrap gap-1">' +
            badges +
            "</span>" +
            "</div>"
          );
        })
        .join("");

      container.innerHTML = header + rows;
    },
  );
}

// -----------------------------------------------------------------------
// Forecast Runs Feed
// -----------------------------------------------------------------------

function refreshFcstFeed() {
  var dateParam = selectedDate ? "?date=" + selectedDate : "";
  fetchAPI("/api/forecast-points/" + selectedCity + dateParam).then(
    function (data) {
      var container = document.getElementById("fcst-feed");
      if (!container) return;

      if (!data || !data.runs || data.runs.length === 0) {
        container.innerHTML =
          '<p class="text-xs text-slate-500">No forecast runs</p>';
        return;
      }

      // Only show runs with full coverage (covers through peak heating)
      var fullRuns = data.runs.filter(function (r) {
        return r.coverage === "full";
      });

      if (fullRuns.length === 0) {
        container.innerHTML =
          '<p class="text-xs text-slate-500">No full-coverage runs</p>';
        return;
      }

      // Model badge colors
      var MODEL_COLORS = {
        hrrr: "bg-blue-900/50 text-blue-400 border-blue-700",
        gfs: "bg-emerald-900/50 text-emerald-400 border-emerald-700",
        ecmwf: "bg-purple-900/50 text-purple-400 border-purple-700",
      };

      // Column header
      var header =
        '<div class="grid grid-cols-[3fr_3fr_3fr_2fr] gap-x-2 text-[10px] text-slate-500 uppercase tracking-wider py-1 border-b border-slate-600">' +
        "<span>Run (ET)</span><span>Model</span><span>High</span><span>\u0394</span></div>";

      var rows = fullRuns
        .map(function (r) {
          var runLabel = toET(r.model_run);

          // Model badge
          var modelName = (r.model_name || "hrrr").toLowerCase();
          var modelColor =
            MODEL_COLORS[modelName] ||
            "bg-slate-700 text-slate-300 border-slate-600";
          var modelBadge =
            '<span class="px-1.5 py-0.5 rounded border text-[10px] ' +
            modelColor +
            '">' +
            modelName.toUpperCase() +
            "</span>";

          // Forecast high
          var highStr =
            r.high_temp_f != null ? r.high_temp_f.toFixed(1) + "\u00b0" : "--";

          // Delta from prior run of same model
          var deltaStr = "--";
          var deltaColor = "text-slate-500";
          if (r.temp_change != null && r.temp_change !== 0) {
            var sign = r.temp_change > 0 ? "+" : "";
            deltaStr = sign + r.temp_change.toFixed(1) + "\u00b0";
            deltaColor =
              r.temp_change > 0 ? "text-emerald-400" : "text-red-400";
          } else if (r.temp_change === 0) {
            deltaStr = "\u2014";
            deltaColor = "text-slate-600";
          }

          return (
            '<div class="grid grid-cols-[3fr_3fr_3fr_2fr] gap-x-2 text-xs py-1 border-b border-slate-700/50 items-center">' +
            '<span class="text-slate-400">' +
            runLabel +
            "</span>" +
            "<span>" +
            modelBadge +
            "</span>" +
            '<span class="text-slate-200">' +
            highStr +
            "</span>" +
            '<span class="' +
            deltaColor +
            '">' +
            deltaStr +
            "</span>" +
            "</div>"
          );
        })
        .join("");

      container.innerHTML = header + rows;
    },
  );
}

// -----------------------------------------------------------------------
// Order Book Ladder
// -----------------------------------------------------------------------

function refreshBracketLadder() {
  var dateParam = selectedDate ? "?date=" + selectedDate : "";
  fetchAPI("/api/brackets/" + selectedCity + dateParam).then(function (data) {
    var chartEl = document.getElementById("bracket-chart");
    if (!chartEl) return;

    // Stale liquidity warning
    var staleEl = document.getElementById("bracket-stale");
    if (staleEl && data && data.captured_at) {
      var capturedMs = new Date(data.captured_at).getTime();
      var ageMin = Math.round((Date.now() - capturedMs) / 60000);
      if (ageMin > 30) {
        staleEl.textContent = "STALE (" + ageMin + "m)";
        staleEl.className =
          "ml-2 px-1.5 py-0.5 rounded border text-[10px] bg-amber-900/50 text-amber-400 border-amber-700";
      } else {
        staleEl.textContent = "";
        staleEl.className = "hidden";
      }
    }

    if (!data || !data.brackets || data.brackets.length === 0) {
      chartEl.innerHTML =
        '<p class="text-xs text-slate-500">No bracket data</p>';
      var liqEl = document.getElementById("liquidity-bar");
      if (liqEl) liqEl.innerHTML = "";
      return;
    }

    // Filter to brackets with market data
    var filtered = data.brackets.filter(function (b) {
      return b.yes_bid != null || b.yes_ask != null;
    });

    if (filtered.length === 0) {
      chartEl.innerHTML =
        '<p class="text-xs text-slate-500">No market data</p>';
      return;
    }

    // Build table header
    var html =
      '<table class="w-full text-xs">' +
      '<thead><tr class="text-[10px] text-slate-500 uppercase tracking-wider border-b border-slate-600">' +
      '<th class="py-1 text-left">Bracket</th>' +
      '<th class="py-1 text-right">Bid</th>' +
      '<th class="py-1 text-right">Ask</th>' +
      '<th class="py-1 text-right">Model</th>' +
      '<th class="py-1 text-right">Edge</th>' +
      '<th class="py-1 text-right">EV</th>' +
      "</tr></thead><tbody>";

    filtered.forEach(function (b) {
      // Label
      var label;
      if (b.floor == null) label = "\u2264" + (b.cap - 1) + "\u00b0F";
      else if (b.cap == null) label = "\u2265" + (b.floor + 1) + "\u00b0F";
      else label = b.floor + "-" + b.cap + "\u00b0F";

      // Bid/Ask in cents (integer)
      var bidCents = b.yes_bid != null ? Math.round(b.yes_bid * 100) : null;
      var askCents = b.yes_ask != null ? Math.round(b.yes_ask * 100) : null;

      // Model as percentage
      var modelPct =
        b.model_prob != null ? Math.round(b.model_prob * 10000) / 100 : null;

      // Edge & EV calculations
      var askFrac = b.yes_ask; // 0-1 probability
      var edgePct = null;
      var evCents = null;
      var feeCents = null;

      if (b.model_prob != null && askFrac != null && askFrac > 0) {
        edgePct = Math.round((b.model_prob - askFrac) * 10000) / 100;

        // Fee per contract: max(ceil(0.07 * P * (1-P) * 100), 1)
        var P = askFrac;
        feeCents = Math.max(Math.ceil(0.07 * P * (1 - P) * 100), 1);

        // EV = (model_prob - yes_ask) * 100 - feeCents
        evCents =
          Math.round((b.model_prob - askFrac) * 100 * 100) / 100 - feeCents;
        evCents = Math.round(evCents * 100) / 100;
      }

      // Row highlighting
      var rowClass = "border-b border-slate-700/50";
      if (evCents != null) {
        if (evCents > 0) rowClass += " bg-emerald-900/20";
        else if (evCents < 0) rowClass += " bg-red-900/10";
      }

      // Bold edge text when edge exceeds fee
      var edgeBold =
        edgePct != null && feeCents != null && edgePct > feeCents
          ? " font-bold"
          : "";

      // Edge color
      var edgeColor = "text-slate-500";
      if (edgePct != null) {
        edgeColor = edgePct > 0 ? "text-emerald-400" : "text-red-400";
      }

      // EV color
      var evColor = "text-slate-500";
      if (evCents != null) {
        evColor = evCents > 0 ? "text-emerald-400" : "text-red-400";
      }

      html +=
        '<tr class="' +
        rowClass +
        '">' +
        '<td class="py-1 text-slate-300">' +
        label +
        "</td>" +
        '<td class="py-1 text-right text-slate-400">' +
        (bidCents != null ? bidCents + "\u00a2" : "--") +
        "</td>" +
        '<td class="py-1 text-right text-slate-400">' +
        (askCents != null ? askCents + "\u00a2" : "--") +
        "</td>" +
        '<td class="py-1 text-right text-slate-200">' +
        (modelPct != null ? modelPct.toFixed(1) + "%" : "--") +
        "</td>" +
        '<td class="py-1 text-right ' +
        edgeColor +
        edgeBold +
        '">' +
        (edgePct != null
          ? (edgePct > 0 ? "+" : "") + edgePct.toFixed(1) + "pp"
          : "--") +
        "</td>" +
        '<td class="py-1 text-right ' +
        evColor +
        '">' +
        (evCents != null
          ? (evCents > 0 ? "+" : "") + evCents.toFixed(1) + "\u00a2"
          : "--") +
        "</td>" +
        "</tr>";
    });

    html += "</tbody></table>";
    chartEl.innerHTML = html;

    // Render liquidity info below the table
    var liqEl = document.getElementById("liquidity-bar");
    if (liqEl && data.liquidity) {
      var spreadCents = Math.round(data.liquidity.avg_spread * 100);
      liqEl.innerHTML =
        '<span class="text-slate-400">Vol:</span> ' +
        '<span class="text-slate-200">' +
        data.liquidity.total_volume.toLocaleString() +
        "</span>" +
        '<span class="mx-2 text-slate-600">|</span>' +
        '<span class="text-slate-400">Avg Spread:</span> ' +
        '<span class="text-slate-200">' +
        spreadCents +
        "\u00a2</span>";
    }
  });
}

// -----------------------------------------------------------------------
// Stale Data Check
// -----------------------------------------------------------------------

async function checkStaleness() {
  var health = await fetchAPI("/api/health");
  if (!health) return;
  var warnings = [];
  if (health.obs_stale)
    warnings.push(
      "Observations (" + Math.round(health.obs_age_minutes) + " min)",
    );
  if (health.fcst_stale)
    warnings.push(
      "Forecasts (" + Math.round(health.fcst_age_minutes) + " min)",
    );
  var banner = document.getElementById("stale-banner");
  if (!banner) return;
  if (warnings.length > 0) {
    banner.textContent = "Stale data: " + warnings.join(", ");
    banner.classList.remove("hidden");
  } else {
    banner.classList.add("hidden");
  }
}

// -----------------------------------------------------------------------
// Refresh Loop
// -----------------------------------------------------------------------

function refreshAll() {
  refreshTempChart();
  refreshObsFeed();
  refreshFcstFeed();
  refreshBracketLadder();
  checkStaleness();
  countdown = 60;
}

// Hook into base.html's date toggle
function onDateChange() {
  refreshAll();
}

// Show loading skeletons before first fetch
showSkeleton("obs-feed", 6);
showSkeleton("fcst-feed", 5);

// Initial load
refreshAll();

// Auto-refresh every 60 seconds
setInterval(refreshAll, 60000);
