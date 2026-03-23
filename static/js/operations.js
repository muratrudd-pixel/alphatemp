/**
 * AlphaTemp — Operations tab (Light Theme)
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
                font: { size: 14, color: "#9ca3af" },
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
        line: { color: "#60a5fa", width: 2, dash: "dot" },
        name: "HRRR Forecast",
        hovertemplate: "%{y:.1f}\u00b0F<extra></extra>",
      };

      // 2. Observations (blue line with light blue area fill)
      var traceObs = {
        x: obsX,
        y: obsY,
        type: "scatter",
        mode: "lines",
        line: { color: "#3b82f6", width: 2.5 },
        fill: "tozeroy",
        fillcolor: "rgba(59,130,246,0.08)",
        name: "Observed",
        hovertemplate: "%{y:.1f}\u00b0F<extra></extra>",
      };

      // 6-hour synoptic max markers (gray triangles)
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
            color: "#9ca3af",
          },
          name: "6hr High",
          hovertemplate: "%{y:.1f}\u00b0F<extra>6-hour max</extra>",
        };
      }

      // Settlement marker (amber glowing circle)
      var traceSettlement = null;
      if (data.observed_high != null && data.observed_high_at) {
        var src = data.settlement_source;
        var settleName, settleHover;
        if (src === "nws_cli") {
          settleName = "NWS Settlement (CLI)";
          settleHover = "NWS Settlement: ";
        } else if (src === "dsm") {
          settleName = "Settlement (DSM)";
          settleHover = "DSM Settlement: ";
        } else {
          settleName = "Running High (est)";
          settleHover = "Running High: ";
        }
        traceSettlement = {
          x: [toETIso(data.observed_high_at)],
          y: [data.observed_high],
          type: "scatter",
          mode: "markers+text",
          marker: {
            size: 18,
            color: "#f59e0b",
            symbol: "circle",
            line: { color: "rgba(245,158,11,0.35)", width: 6 },
          },
          text: [data.observed_high.toFixed(0) + "\u00b0"],
          textposition: "top center",
          textfont: {
            size: 11,
            color: "#d97706",
            family: "Space Mono, monospace",
            weight: 600,
          },
          name: settleName,
          hovertemplate: settleHover + "%{y:.1f}\u00b0F<extra></extra>",
        };
      }

      // Model prediction band (p25-p75 shaded, median line)
      var traceModelBandUpper = null;
      var traceModelBandLower = null;
      var traceModelMedian = null;
      if (data.model_band && data.model_band.median != null) {
        var mb = data.model_band;
        var bandX = [fcstX[0], fcstX[fcstX.length - 1]];
        traceModelBandUpper = {
          x: bandX,
          y: [mb.p75, mb.p75],
          type: "scatter",
          mode: "lines",
          line: { color: "transparent", width: 0 },
          showlegend: false,
          hoverinfo: "skip",
        };
        traceModelBandLower = {
          x: bandX,
          y: [mb.p25, mb.p25],
          type: "scatter",
          mode: "lines",
          line: { color: "transparent", width: 0 },
          fill: "tonexty",
          fillcolor: "rgba(168,85,247,0.05)",
          name: "Model 25-75%",
          hoverinfo: "skip",
        };
        traceModelMedian = {
          x: bandX,
          y: [mb.median, mb.median],
          type: "scatter",
          mode: "lines",
          line: { color: "rgba(168,85,247,0.3)", width: 1.5, dash: "dashdot" },
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
          line: { color: "rgba(0,0,0,0.15)", width: 1, dash: "dash" },
        },
      ];

      // Add "Now" annotation at bottom of the line
      var annotations = [
        {
          x: nowLine,
          y: 0,
          yref: "paper",
          text: "Now",
          showarrow: false,
          font: { size: 10, color: "#9ca3af", family: "Space Mono, monospace" },
          yanchor: "top",
          yshift: 5,
        },
      ];

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
        showlegend: false,
        yaxis: yAxisOpts,
        xaxis: Object.assign({}, PLOTLY_LAYOUT.xaxis, {
          type: "date",
          range: [dayStart, dayEnd],
          dtick: 7200000,
          tickformat: "%H:%M",
          showgrid: true,
        }),
        shapes: shapes,
        annotations: annotations,
      });

      Plotly.react("temp-curve-chart", allTraces, layout, PLOTLY_CONFIG);

      // Render HTML legend below chart
      var legendEl = document.getElementById("chart-legend");
      if (legendEl) {
        var items = [
          {
            color: "#f59e0b",
            symbol: "&#9679;",
            label: traceSettlement ? traceSettlement.name : "Settlement",
          },
          { color: "#9ca3af", symbol: "&#9650;", label: "6hr High" },
        ];
        if (traceModelMedian) {
          items.push({
            color: "rgba(168,85,247,0.5)",
            symbol: "- -",
            label: traceModelMedian.name,
          });
        }
        if (traceModelBandLower) {
          items.push({
            color: "rgba(168,85,247,0.12)",
            symbol: "&#9632;",
            label: "Model 25-75%",
          });
        }
        items.push({ color: "#3b82f6", symbol: "&#9644;", label: "Observed" });
        items.push({
          color: "#60a5fa",
          symbol: "&#8943;",
          label: "HRRR Forecast",
        });

        legendEl.innerHTML = items
          .map(function (item) {
            return (
              '<span class="flex items-center gap-1.5">' +
              '<span style="color:' +
              item.color +
              '; font-size: 14px; line-height: 1;">' +
              item.symbol +
              "</span>" +
              "<span>" +
              item.label +
              "</span></span>"
            );
          })
          .join("");
      }
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
          '<p class="text-xs text-gray-400">No observations</p>';
        return;
      }

      // Sort by ingested_at descending (most recently received first)
      var sorted = data.observations.slice().sort(function (a, b) {
        var aT = a.ingested_at || "";
        var bT = b.ingested_at || "";
        return bT.localeCompare(aT);
      });
      var obs = sorted.slice(0, 20);

      var header =
        '<div class="grid grid-cols-[2fr_2fr_2fr_3fr] gap-x-2 text-[10px] text-gray-400 uppercase tracking-wider py-1.5 at-mono">' +
        "<span>Rcvd</span><span>Obs</span><span>Temp</span><span>Details</span></div>";

      var rows = obs
        .map(function (o) {
          var obsTime = toET(o.observed_at);
          var rcvdTime = o.ingested_at ? toET(o.ingested_at) : "--";

          // Only show badges for actionable signals — no generic METAR
          var badges = "";
          if (o.source && o.source.includes("SPECI")) {
            badges =
              '<span class="px-1.5 py-0.5 rounded text-[10px] bg-amber-100 text-amber-700 font-medium">SPECI</span>';
          } else if (o.source === "NWS CLI") {
            badges =
              '<span class="px-1.5 py-0.5 rounded text-[10px] bg-emerald-100 text-emerald-700 font-medium">NWS CLI</span>';
          }
          if (o.obs_window === "prior 6hrs") {
            badges +=
              (badges ? " " : "") +
              '<span class="px-1.5 py-0.5 rounded text-[10px] bg-gray-800 text-white font-medium">6hr High</span>';
          }
          if (o.is_new_high === true) {
            badges +=
              (badges ? " " : "") +
              '<span class="px-1.5 py-0.5 rounded text-[10px] bg-emerald-600 text-white font-bold">NEW HIGH</span>';
          }

          return (
            '<div class="grid grid-cols-[2fr_2fr_2fr_3fr] gap-x-2 text-xs py-1.5 items-center at-mono">' +
            '<span class="text-gray-500">' +
            rcvdTime +
            "</span>" +
            '<span class="text-gray-400">' +
            obsTime +
            "</span>" +
            '<span class="text-gray-900 font-medium">' +
            o.temp_f.toFixed(1) +
            "\u00b0F</span>" +
            '<span class="flex flex-wrap gap-1" style="font-family: \'Space Mono\', monospace;">' +
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
          '<p class="text-xs text-gray-400">No forecast runs</p>';
        return;
      }

      var fullRuns = data.runs;

      if (fullRuns.length === 0) {
        container.innerHTML =
          '<p class="text-xs text-gray-400">No forecast runs</p>';
        return;
      }

      // Sort by ingested_at descending (most recently received first)
      fullRuns.sort(function (a, b) {
        var aT = a.ingested_at || "";
        var bT = b.ingested_at || "";
        return bT.localeCompare(aT);
      });

      var MODEL_COLORS = {
        hrrr: "bg-blue-100 text-blue-700",
        gfs: "bg-emerald-100 text-emerald-700",
        ecmwf: "bg-purple-100 text-purple-700",
      };

      var header =
        '<div class="grid grid-cols-[2fr_2fr_2fr_2fr_1.5fr] gap-x-2 text-[10px] text-gray-400 uppercase tracking-wider py-1.5 at-mono">' +
        "<span>Rcvd</span><span>Run (ET)</span><span>Model</span><span>High</span><span>\u0394</span></div>";

      var rows = fullRuns
        .map(function (r) {
          var runLabel = toET(r.model_run);
          var rcvdLabel = r.ingested_at ? toET(r.ingested_at) : "--";

          var modelName = (r.model_name || "hrrr").toLowerCase();
          var modelColor =
            MODEL_COLORS[modelName] || "bg-gray-100 text-gray-600";
          var modelBadge =
            '<span class="px-2 py-0.5 rounded text-[10px] font-medium ' +
            modelColor +
            '">' +
            modelName.toUpperCase() +
            "</span>";

          var highStr =
            r.high_temp_f != null ? r.high_temp_f.toFixed(1) + "\u00b0" : "--";

          var deltaStr = "--";
          var deltaColor = "text-gray-400";
          if (r.temp_change != null && r.temp_change !== 0) {
            var sign = r.temp_change > 0 ? "+" : "";
            deltaStr = sign + r.temp_change.toFixed(1) + "\u00b0";
            deltaColor =
              r.temp_change > 0 ? "text-emerald-600" : "text-red-600";
          } else if (r.temp_change === 0) {
            deltaStr = "\u2014";
            deltaColor = "text-gray-300";
          }

          var rowOpacity = r.coverage === "full" ? "" : " opacity-40";
          return (
            '<div class="grid grid-cols-[2fr_2fr_2fr_2fr_1.5fr] gap-x-2 text-xs py-1.5 items-center at-mono' +
            rowOpacity +
            '">' +
            '<span class="text-gray-500">' +
            rcvdLabel +
            "</span>" +
            '<span class="text-gray-400">' +
            runLabel +
            "</span>" +
            "<span>" +
            modelBadge +
            "</span>" +
            '<span class="text-gray-900 font-medium">' +
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
        staleEl.textContent = " STALE (" + ageMin + "m)";
        staleEl.className =
          "ml-2 px-1.5 py-0.5 rounded text-[10px] bg-amber-100 text-amber-700 font-medium";
      } else {
        staleEl.textContent = "";
        staleEl.className = "hidden";
      }
    }

    if (!data || !data.brackets || data.brackets.length === 0) {
      chartEl.innerHTML =
        '<p class="text-xs text-gray-400">No bracket data</p>';
      var liqEl = document.getElementById("liquidity-bar");
      if (liqEl) liqEl.innerHTML = "";
      return;
    }

    var filtered = data.brackets.filter(function (b) {
      return b.yes_bid != null || b.yes_ask != null;
    });

    if (filtered.length === 0) {
      chartEl.innerHTML = '<p class="text-xs text-gray-400">No market data</p>';
      return;
    }

    // Find model-favored bracket (highest model_prob)
    var maxModelProb = 0;
    var modelFavIdx = -1;
    filtered.forEach(function (b, i) {
      if (b.model_prob != null && b.model_prob > maxModelProb) {
        maxModelProb = b.model_prob;
        modelFavIdx = i;
      }
    });

    // Build table
    var html =
      '<table class="w-full text-xs at-mono" style="border-collapse: separate; border-spacing: 0;">' +
      '<thead><tr class="text-[10px] text-gray-400 uppercase tracking-wider">' +
      '<th class="py-2 text-left font-medium">Bracket</th>' +
      '<th class="py-2 text-right font-medium">Volume</th>' +
      '<th class="py-2 text-right font-medium">Bid</th>' +
      '<th class="py-2 text-right font-medium">Ask</th>' +
      '<th class="py-2 text-right font-medium">Model %</th>' +
      '<th class="py-2 text-right font-medium">Edge</th>' +
      '<th class="py-2 text-right font-medium">EV</th>' +
      "</tr></thead><tbody>";

    filtered.forEach(function (b, idx) {
      var label;
      if (b.floor == null) label = "\u2264" + (b.cap - 1) + "\u00b0F";
      else if (b.cap == null) label = "\u2265" + (b.floor + 1) + "\u00b0F";
      else label = b.floor + "-" + b.cap + "\u00b0F";

      var bidCents = b.yes_bid != null ? Math.round(b.yes_bid * 100) : null;
      var askCents = b.yes_ask != null ? Math.round(b.yes_ask * 100) : null;

      // Volume display (compact: 1.2k, 54k, etc.)
      var volStr = "--";
      if (b.volume != null) {
        if (b.volume >= 1000) {
          volStr = (b.volume / 1000).toFixed(b.volume >= 10000 ? 0 : 1) + "k";
        } else {
          volStr = b.volume.toString();
        }
      }

      var modelPct =
        b.model_prob != null ? Math.round(b.model_prob * 10000) / 100 : null;

      var askFrac = b.yes_ask;
      var edgePct = null;
      var evCents = null;
      var feeCents = null;

      if (b.model_prob != null && askFrac != null && askFrac > 0) {
        edgePct = Math.round((b.model_prob - askFrac) * 10000) / 100;
        var P = askFrac;
        feeCents = Math.max(Math.ceil(0.07 * P * (1 - P) * 100), 1);
        evCents =
          Math.round((b.model_prob - askFrac) * 100 * 100) / 100 - feeCents;
        evCents = Math.round(evCents * 100) / 100;
      }

      // Pill badges — strict highlighting: only green if EV clears fee drag
      var clearsFees = evCents != null && evCents > 0;

      var edgePill = '<span class="text-gray-300">--</span>';
      if (edgePct != null) {
        var edgePillClass;
        if (edgePct > 0 && clearsFees) {
          edgePillClass = "at-pill at-pill-profit";
        } else if (edgePct < 0) {
          edgePillClass = "at-pill at-pill-loss";
        } else {
          edgePillClass = "at-pill at-pill-neutral";
        }
        edgePill =
          '<span class="' +
          edgePillClass +
          '">' +
          (edgePct > 0 ? "+" : "") +
          edgePct.toFixed(1) +
          "pp</span>";
      }

      var evPill = '<span class="text-gray-300">--</span>';
      if (evCents != null) {
        var evPillClass;
        if (clearsFees) {
          evPillClass = "at-pill at-pill-profit";
        } else if (evCents < 0) {
          evPillClass = "at-pill at-pill-loss";
        } else {
          evPillClass = "at-pill at-pill-neutral";
        }
        evPill =
          '<span class="' +
          evPillClass +
          '">' +
          (evCents > 0 ? "+" : "") +
          evCents.toFixed(1) +
          "\u00a2</span>";
      }

      // Row highlighting: model favorite vs market confirmed
      var rowStyle = "";
      var rowClass = "";
      if (idx === modelFavIdx) {
        // Check if market confirms (yes_bid >= 0.99)
        var marketConfirmed = b.yes_bid != null && b.yes_bid >= 0.99;
        if (marketConfirmed) {
          rowStyle =
            ' style="background: rgba(16,185,129,0.08); outline: 2.5px solid rgba(16,185,129,0.4); outline-offset: -1px; border-radius: 6px;"';
        } else {
          rowStyle =
            ' style="background: rgba(139,92,246,0.06); outline: 2px solid rgba(139,92,246,0.3); outline-offset: -1px; border-radius: 6px;"';
        }
      }

      html +=
        '<tr class="' +
        rowClass +
        '"' +
        rowStyle +
        ">" +
        '<td class="py-2 text-gray-800 font-medium">' +
        label +
        "</td>" +
        '<td class="py-2 text-right text-gray-500">' +
        volStr +
        "</td>" +
        '<td class="py-2 text-right text-gray-500">' +
        (bidCents != null ? bidCents + "\u00a2" : "--") +
        "</td>" +
        '<td class="py-2 text-right text-gray-500">' +
        (askCents != null ? askCents + "\u00a2" : "--") +
        "</td>" +
        '<td class="py-2 text-right text-gray-800">' +
        (modelPct != null ? modelPct.toFixed(1) + "%" : "--") +
        "</td>" +
        '<td class="py-2 text-right">' +
        edgePill +
        "</td>" +
        '<td class="py-2 text-right">' +
        evPill +
        "</td>" +
        "</tr>";
    });

    html += "</tbody></table>";
    chartEl.innerHTML = html;

    // Liquidity bar with visual progress
    var liqEl = document.getElementById("liquidity-bar");
    if (liqEl && data.liquidity) {
      var spreadCents = Math.round(data.liquidity.avg_spread * 100);
      var vol = data.liquidity.total_volume;
      // Rough bar fill: log scale, cap at 500k
      var fillPct = Math.min(
        100,
        Math.round((Math.log10(vol + 1) / Math.log10(500000)) * 100),
      );
      liqEl.innerHTML =
        '<div class="flex items-center gap-3 text-xs text-gray-500 at-mono">' +
        "<span>Liquidity</span>" +
        '<div class="liq-bar flex-1"><div class="liq-bar-fill" style="width:' +
        fillPct +
        '%"></div></div>' +
        '<span class="text-gray-700">Vol: ' +
        vol.toLocaleString() +
        "</span>" +
        '<span class="text-gray-400">|</span>' +
        '<span class="text-gray-700">Avg Spread: ' +
        spreadCents +
        "\u00a2</span>" +
        "</div>";
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
  var staleText = document.getElementById("stale-text");
  if (!banner || !staleText) return;
  if (warnings.length > 0) {
    staleText.textContent = "Stale data: " + warnings.join(", ");
    banner.classList.remove("hidden");
  } else {
    banner.classList.add("hidden");
  }
}

// -----------------------------------------------------------------------
// NWS CLI Report
// -----------------------------------------------------------------------

function refreshCLI() {
  var dateParam = selectedDate ? "?date=" + selectedDate : "";
  fetchAPI("/api/nws-cli/" + selectedCity + dateParam).then(function (data) {
    var statusEl = document.getElementById("cli-status");
    var textEl = document.getElementById("cli-text");
    if (!statusEl || !textEl) return;

    if (!data || !data.raw_text) {
      var src = data && data.source ? " (source: " + data.source + ")" : "";
      statusEl.textContent = "No CLI report available yet" + src;
      textEl.textContent = "Waiting for NWS CLI product (~16:30 ET)...";
      textEl.className =
        "text-[11px] at-mono text-gray-400 bg-gray-50 rounded-lg p-3 italic";
      return;
    }

    var ingestedStr = data.ingested_at
      ? " \u00B7 Received: " + toET(data.ingested_at)
      : "";
    statusEl.innerHTML =
      '<span class="at-pill at-pill-profit">CLI</span> ' +
      '<span class="text-gray-700 font-medium">' +
      data.max_temp_f +
      "\u00B0F</span>" +
      '<span class="text-gray-400">' +
      ingestedStr +
      "</span>";
    textEl.textContent = data.raw_text;
    textEl.className =
      "text-[11px] at-mono text-gray-700 bg-gray-50 rounded-lg p-3 overflow-x-auto whitespace-pre-wrap max-h-64 overflow-y-auto";
  });
}

// -----------------------------------------------------------------------
// Refresh Loop
// -----------------------------------------------------------------------

function refreshAll() {
  refreshTempChart();
  refreshObsFeed();
  refreshFcstFeed();
  refreshBracketLadder();
  refreshCLI();
  checkStaleness();
  countdown = 60;
}

function onDateChange() {
  refreshAll();
}

showSkeleton("obs-feed", 6);
showSkeleton("fcst-feed", 5);

refreshAll();
setInterval(refreshAll, 60000);
