/**
 * AlphaTemp — Operations tab
 *
 * Renders the temperature curve chart, observation feed,
 * forecast runs feed, and placeholder panels for bracket
 * spread and positions.
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
    if (input.indexOf('Z') === -1 && input.indexOf('+') === -1 && input.indexOf('T') !== -1) {
        input = input + 'Z';
    }
    var d = new Date(input);
    if (isNaN(d.getTime())) return null;
    var et = new Date(d.toLocaleString('en-US', { timeZone: 'America/New_York' }));
    return et.getFullYear() + '-' +
        String(et.getMonth() + 1).padStart(2, '0') + '-' +
        String(et.getDate()).padStart(2, '0') + 'T' +
        String(et.getHours()).padStart(2, '0') + ':' +
        String(et.getMinutes()).padStart(2, '0') + ':' +
        String(et.getSeconds()).padStart(2, '0');
}

/**
 * Short ET time display for hover text (e.g. "14:30").
 * Wraps shared toET() with default options.
 */
function formatETShort(utcStr) {
    return toET(utcStr);
}

/**
 * Current time in ET as an ISO-like string (for "now" line on chart).
 */
function nowETIso() {
    var d = new Date();
    var et = new Date(d.toLocaleString('en-US', { timeZone: 'America/New_York' }));
    return et.getFullYear() + '-' +
        String(et.getMonth() + 1).padStart(2, '0') + '-' +
        String(et.getDate()).padStart(2, '0') + 'T' +
        String(et.getHours()).padStart(2, '0') + ':' +
        String(et.getMinutes()).padStart(2, '0') + ':' +
        String(et.getSeconds()).padStart(2, '0');
}

// -----------------------------------------------------------------------
// Neighbor Toggle State
// -----------------------------------------------------------------------

var neighborsActive = { KLGA: false, KEWR: false };
var NEIGHBOR_COLORS = { KLGA: '#a78bfa', KEWR: '#fb923c' };

function toggleNeighbor(station) {
    neighborsActive[station] = !neighborsActive[station];
    var btnId = 'btn-' + station.toLowerCase();
    var btn = document.getElementById(btnId);
    if (!btn) return;
    if (neighborsActive[station]) {
        btn.className = 'px-2 py-0.5 text-[10px] rounded border border-slate-500 bg-slate-600 text-slate-100 uppercase tracking-wider';
    } else {
        btn.className = 'px-2 py-0.5 text-[10px] rounded border border-slate-600 bg-slate-700 text-slate-400 hover:bg-slate-600 uppercase tracking-wider';
    }
    refreshTempChart();
}

// -----------------------------------------------------------------------
// Temperature Curve
// -----------------------------------------------------------------------

function refreshTempChart() {
    var dateParam = selectedDate ? '?date=' + selectedDate : '';
    fetchAPI('/api/forecast-curve/' + selectedCity + dateParam).then(function(data) {
        if (!data || !data.forecasts || data.forecasts.length === 0) {
            Plotly.react('temp-curve-chart', [], Object.assign({}, PLOTLY_LAYOUT, {
                annotations: [{
                    text: 'No forecast data',
                    showarrow: false,
                    font: { size: 14, color: COLORS.slate400 },
                    xref: 'paper', yref: 'paper', x: 0.5, y: 0.5
                }]
            }), PLOTLY_CONFIG);
            return;
        }

        // Extract arrays
        var fcstX = data.forecasts.map(function(p) { return toETIso(p.valid_at); });
        var fcstY = data.forecasts.map(function(p) { return p.temp_f; });
        var obsX = data.observations.map(function(p) { return toETIso(p.observed_at); });
        var obsY = data.observations.map(function(p) { return p.temp_f; });

        // --- Traces ---

        // 1. Forecast center line (blue, dotted)
        var traceFcstCenter = {
            x: fcstX, y: fcstY,
            type: 'scatter', mode: 'lines',
            line: { color: COLORS.blue, width: 2, dash: 'dot' },
            name: 'HRRR Forecast',
            hovertemplate: '%{y:.1f}\u00b0F<extra></extra>'
        };

        // 2. Forecast + Drift line (teal, solid)
        var adjX = (data.bias_adjusted_forecasts || []).map(function(p) { return toETIso(p.valid_at); });
        var adjY = (data.bias_adjusted_forecasts || []).map(function(p) { return p.temp_f; });
        var adjLabel = 'Forecast + Drift';
        if (data.adjustment != null) {
            adjLabel += ' (' + (data.adjustment >= 0 ? '+' : '') + data.adjustment + '\u00b0F)';
        }
        var traceAdj = {
            x: adjX, y: adjY,
            type: 'scatter', mode: 'lines',
            line: { color: COLORS.teal, width: 2 },
            name: adjLabel,
            hovertemplate: '%{y:.1f}\u00b0F<extra>forecast + drift</extra>'
        };

        // 7. Observations (white, lines+markers)
        var traceObs = {
            x: obsX, y: obsY,
            type: 'scatter', mode: 'lines+markers',
            line: { color: COLORS.white, width: 2 },
            marker: { size: 4, color: COLORS.white },
            name: 'Observed',
            hovertemplate: '%{y:.1f}\u00b0F<extra></extra>'
        };

        // 8. 6-hour synoptic max markers (amber, triangle-up)
        var trace6hMax = null;
        if (data.six_hr_maxes && data.six_hr_maxes.length > 0) {
            trace6hMax = {
                x: data.six_hr_maxes.map(function(p) { return toETIso(p.observed_at); }),
                y: data.six_hr_maxes.map(function(p) { return p.temp_f; }),
                type: 'scatter', mode: 'markers',
                marker: { size: 10, symbol: 'triangle-up', color: COLORS.amber },
                name: '6hr High',
                hovertemplate: '%{y:.1f}\u00b0F<extra>6-hour max</extra>'
            };
        }

        // 9. Settlement marker
        var traceSettlement = null;
        if (data.observed_high != null && data.observed_high_at) {
            var src = data.settlement_source;
            var settleName, settleHover, settleSymbol;
            if (src === 'nws_cli') {
                settleName = 'NWS Settlement (CLI)';
                settleHover = 'NWS Settlement: ';
                settleSymbol = 'diamond';
            } else if (src === 'dsm') {
                settleName = 'Settlement (DSM)';
                settleHover = 'DSM Settlement: ';
                settleSymbol = 'diamond';
            } else {
                settleName = 'Running High (est)';
                settleHover = 'Running High: ';
                settleSymbol = 'circle';
            }
            traceSettlement = {
                x: [toETIso(data.observed_high_at)],
                y: [data.observed_high],
                type: 'scatter', mode: 'markers',
                marker: {
                    size: 14,
                    color: COLORS.amber,
                    symbol: settleSymbol,
                    line: { color: '#ffffff', width: 1.5 }
                },
                name: settleName,
                hovertemplate: settleHover + '%{y:.1f}\u00b0F<extra></extra>'
            };
        }

        // 10. Prior runs (faded blue)
        var priorTraces = (data.prior_runs || []).map(function(run, i) {
            return {
                x: run.points.map(function(p) { return toETIso(p.valid_at); }),
                y: run.points.map(function(p) { return p.temp_f; }),
                type: 'scatter', mode: 'lines',
                line: { color: 'rgba(59,130,246,0.2)', width: 1 },
                name: i === 0 ? 'Prior Runs' : undefined,
                showlegend: i === 0,
                legendgroup: 'prior',
                hovertemplate: '%{y:.1f}\u00b0F<extra>' + run.model_run + '</extra>'
            };
        });

        // 11. Neighbor obs when toggled
        var neighborTraces = [];
        if (data.neighbor_obs) {
            Object.keys(data.neighbor_obs).forEach(function(stnId) {
                if (!neighborsActive[stnId]) return;
                var pts = data.neighbor_obs[stnId];
                if (!pts || pts.length === 0) return;
                var color = NEIGHBOR_COLORS[stnId] || '#9ca3af';
                neighborTraces.push({
                    x: pts.map(function(p) { return toETIso(p.observed_at); }),
                    y: pts.map(function(p) { return p.temp_f; }),
                    type: 'scatter', mode: 'lines+markers',
                    line: { color: color, width: 1.5, dash: 'dash' },
                    marker: { size: 3, color: color },
                    name: stnId,
                    hovertemplate: '%{y:.1f}\u00b0F<extra>' + stnId + '</extra>'
                });
            });
        }

        // --- Assemble traces ---
        var allTraces = [].concat(
            priorTraces,
            [traceFcstCenter, traceAdj, traceObs],
            neighborTraces
        );
        if (trace6hMax) allTraces.push(trace6hMax);
        if (traceSettlement) allTraces.push(traceSettlement);

        // --- Layout ---
        var dayStr = selectedDate || new Date().toLocaleDateString('en-CA', { timeZone: 'America/New_York' });
        var dayStart = dayStr + 'T00:00:00';
        var dayEnd = dayStr + 'T23:59:59';

        var nowLine = data.now_utc ? toETIso(data.now_utc) : nowETIso();

        var shapes = [{
            type: 'line',
            x0: nowLine, x1: nowLine,
            y0: 0, y1: 1, yref: 'paper',
            line: { color: 'rgba(148,163,184,0.4)', width: 1, dash: 'dash' }
        }];

        // Settlement horizontal line
        if (data.observed_high != null) {
            var isAuthoritative = data.settlement_source === 'nws_cli' || data.settlement_source === 'dsm';
            shapes.push({
                type: 'line',
                x0: 0, x1: 1, xref: 'paper',
                y0: data.observed_high, y1: data.observed_high,
                line: {
                    color: 'rgba(245,158,11,0.4)',
                    width: 1,
                    dash: isAuthoritative ? 'solid' : 'dot'
                }
            });
        }

        var layout = Object.assign({}, PLOTLY_LAYOUT, {
            showlegend: true,
            legend: { x: 0, y: 1.02, orientation: 'h', font: { size: 10 } },
            yaxis: Object.assign({}, PLOTLY_LAYOUT.yaxis, {
                title: { text: '\u00b0F', standoff: 8, font: { size: 10 } }
            }),
            xaxis: Object.assign({}, PLOTLY_LAYOUT.xaxis, {
                type: 'date',
                range: [dayStart, dayEnd],
                dtick: 7200000,
                tickformat: '%H:%M',
                showgrid: true
            }),
            shapes: shapes
        });

        Plotly.react('temp-curve-chart', allTraces, layout, PLOTLY_CONFIG);
    });
}

// -----------------------------------------------------------------------
// Observation Feed
// -----------------------------------------------------------------------

function refreshObsFeed() {
    var dateParam = selectedDate ? '?date=' + selectedDate : '';
    fetchAPI('/api/observations/' + selectedCity + dateParam).then(function(data) {
        var container = document.getElementById('obs-feed');
        if (!container) return;

        if (!data || !data.observations || data.observations.length === 0) {
            container.innerHTML = '<p class="text-xs text-slate-500">No observations</p>';
            return;
        }

        var obs = data.observations.slice(0, 20);
        var html = obs.map(function(o) {
            var time = toET(o.observed_at);
            var sourceColor;
            if (o.source && o.source.includes('SPECI')) {
                sourceColor = 'text-amber-400';
            } else if (o.source === 'NWS CLI') {
                sourceColor = 'text-emerald-400';
            } else if (o.source === 'Synoptic' || (o.source && o.source.startsWith('AWC'))) {
                sourceColor = 'text-teal-400';
            } else {
                sourceColor = 'text-slate-300';
            }
            return '<div class="flex justify-between text-xs py-1 border-b border-slate-700/50">' +
                '<span class="text-slate-400">' + time + '</span>' +
                '<span class="' + sourceColor + '">' + (o.source || '--') + '</span>' +
                '<span class="text-slate-200">' + o.temp_f.toFixed(1) + '\u00b0F</span>' +
            '</div>';
        }).join('');

        container.innerHTML = html;
    });
}

// -----------------------------------------------------------------------
// Forecast Runs Feed
// -----------------------------------------------------------------------

function refreshFcstFeed() {
    var dateParam = selectedDate ? '?date=' + selectedDate : '';
    fetchAPI('/api/forecast-points/' + selectedCity + dateParam).then(function(data) {
        var container = document.getElementById('fcst-feed');
        if (!container) return;

        if (!data || !data.runs || data.runs.length === 0) {
            container.innerHTML = '<p class="text-xs text-slate-500">No forecast runs</p>';
            return;
        }

        var html = data.runs.map(function(r) {
            // Run hour label (e.g. "14z")
            var runLabel = toET(r.model_run) + 'z';

            // Forecast high
            var highStr = r.high_temp_f != null ? r.high_temp_f.toFixed(1) + '\u00b0' : '--';

            // Delta from prior run
            var deltaStr = '--';
            var deltaColor = 'text-slate-500';
            if (r.temp_change != null && r.temp_change !== 0) {
                var sign = r.temp_change > 0 ? '+' : '';
                deltaStr = sign + r.temp_change.toFixed(1) + '\u00b0';
                deltaColor = r.temp_change > 0 ? 'text-emerald-400' : 'text-red-400';
            } else if (r.temp_change === 0) {
                deltaStr = '\u2014';
                deltaColor = 'text-slate-600';
            }

            return '<div class="flex justify-between text-xs py-1 border-b border-slate-700/50">' +
                '<span class="text-slate-400">' + runLabel + '</span>' +
                '<span class="text-blue-400">' + highStr + '</span>' +
                '<span class="' + deltaColor + '">' + deltaStr + '</span>' +
            '</div>';
        }).join('');

        container.innerHTML = html;
    });
}

// -----------------------------------------------------------------------
// Bracket Chart (placeholder)
// -----------------------------------------------------------------------

function refreshBracketChart() {
    var dateParam = selectedDate ? '?date=' + selectedDate : '';
    fetchAPI('/api/brackets/' + selectedCity + dateParam).then(function(data) {
        var chartEl = document.getElementById('bracket-chart');
        if (!chartEl) return;

        if (!data || !data.brackets || data.brackets.length === 0) {
            Plotly.react('bracket-chart', [], Object.assign({}, PLOTLY_LAYOUT, {
                annotations: [{
                    text: 'No bracket data',
                    showarrow: false,
                    font: { size: 14, color: COLORS.slate400 },
                    xref: 'paper', yref: 'paper', x: 0.5, y: 0.5
                }]
            }), PLOTLY_CONFIG);
            // Clear liquidity info
            var liqEl = document.getElementById('liquidity-bar');
            if (liqEl) liqEl.innerHTML = '';
            return;
        }

        // Show all brackets that have market data
        var filtered = data.brackets.filter(function(b) {
            return b.market_mid != null;
        });

        if (filtered.length === 0) {
            Plotly.react('bracket-chart', [], Object.assign({}, PLOTLY_LAYOUT, {
                annotations: [{
                    text: 'No significant brackets',
                    showarrow: false,
                    font: { size: 14, color: COLORS.slate400 },
                    xref: 'paper', yref: 'paper', x: 0.5, y: 0.5
                }]
            }), PLOTLY_CONFIG);
            return;
        }

        // Build labels: "≤37°F", "38-39°F", "≥46°F", etc.
        var labels = filtered.map(function(b) {
            if (b.floor == null) return '\u2264' + (b.cap - 1) + '\u00b0F';
            if (b.cap == null) return '\u2265' + (b.floor + 1) + '\u00b0F';
            return b.floor + '-' + b.cap + '\u00b0F';
        });

        var modelProbs = filtered.map(function(b) {
            return b.model_prob != null ? Math.round(b.model_prob * 10000) / 100 : 0;
        });
        var marketProbs = filtered.map(function(b) {
            return b.market_mid != null ? Math.round(b.market_mid * 10000) / 100 : 0;
        });

        // Only show model trace if model data exists
        var hasModel = modelProbs.some(function(v) { return v > 0; });

        var traces = [];
        if (hasModel) {
            traces.push({
                y: labels,
                x: modelProbs,
                type: 'bar',
                orientation: 'h',
                name: 'Model',
                marker: { color: COLORS.blue, opacity: 0.8 },
                hovertemplate: '%{x:.1f}%<extra>Model</extra>'
            });
        }

        // Kalshi trace (horizontal bar)
        traces.push({
            y: labels,
            x: marketProbs,
            type: 'bar',
            orientation: 'h',
            name: 'Kalshi',
            marker: { color: COLORS.amber, opacity: 0.8 },
            hovertemplate: '%{x:.1f}%<extra>Kalshi</extra>'
        });

        var layout = Object.assign({}, PLOTLY_LAYOUT, {
            barmode: 'group',
            showlegend: hasModel,
            legend: { x: 1, y: 1, xanchor: 'right', font: { size: 10 } },
            xaxis: Object.assign({}, PLOTLY_LAYOUT.xaxis, {
                title: { text: 'Probability %', standoff: 8, font: { size: 10 } }
            }),
            yaxis: Object.assign({}, PLOTLY_LAYOUT.yaxis, {
                autorange: 'reversed'
            }),
            margin: { t: 10, r: 20, b: 40, l: 70 }
        });

        Plotly.react('bracket-chart', traces, layout, PLOTLY_CONFIG);

        // Render liquidity info below the chart
        var liqEl = document.getElementById('liquidity-bar');
        if (liqEl && data.liquidity) {
            var spreadCents = Math.round(data.liquidity.avg_spread * 100);
            liqEl.innerHTML =
                '<span class="text-slate-400">Vol:</span> ' +
                '<span class="text-slate-200">' + data.liquidity.total_volume.toLocaleString() + '</span>' +
                '<span class="mx-2 text-slate-600">|</span>' +
                '<span class="text-slate-400">Avg Spread:</span> ' +
                '<span class="text-slate-200">' + spreadCents + '\u00a2</span>';
        }
    });
}

// -----------------------------------------------------------------------
// Positions (placeholder)
// -----------------------------------------------------------------------

function refreshPositions() {
    var el = document.getElementById('positions-panel');
    if (!el) return;

    var dateParam = selectedDate ? '?date=' + selectedDate : '';
    var posUrl = '/api/positions/' + selectedCity + dateParam;

    fetchAPI(posUrl).then(function(posData) {
        var html = '';
        var openCount = posData && posData.active ? posData.active.length : 0;

        // --- Summary ---
        var dailyPnl = posData ? formatPnL(posData.daily_pnl) : formatPnL(null);
        var totalPnl = posData ? formatPnL(posData.total_pnl) : formatPnL(null);

        html += '<div class="space-y-3">';
        html += '<div class="flex justify-between items-center text-xs">';
        html += '<span class="text-slate-400">Open Positions</span>';
        html += '<span class="text-slate-100 font-bold">' + openCount + '</span>';
        html += '</div>';

        html += '<div class="flex justify-between items-center pt-2 border-t border-slate-700">';
        html += '<div class="text-center">';
        html += '<div class="text-[10px] text-slate-500 uppercase">Daily P&L</div>';
        html += '<div class="text-sm font-bold ' + dailyPnl.colorClass + '">' + dailyPnl.text + '</div>';
        html += '</div>';
        html += '<div class="text-center">';
        html += '<div class="text-[10px] text-slate-500 uppercase">Total P&L</div>';
        html += '<div class="text-sm font-bold ' + totalPnl.colorClass + '">' + totalPnl.text + '</div>';
        html += '</div>';
        html += '</div>';

        html += '<div class="pt-2 border-t border-slate-700 text-center">';
        html += '<a href="/trading" class="text-xs text-blue-400 hover:text-blue-300">See Trading tab for details</a>';
        html += '</div>';
        html += '</div>';

        el.innerHTML = html;
    });
}

// -----------------------------------------------------------------------
// Stale Data Check
// -----------------------------------------------------------------------

async function checkStaleness() {
    var health = await fetchAPI('/api/health');
    if (!health) return;
    var warnings = [];
    if (health.obs_stale) warnings.push('Observations (' + Math.round(health.obs_age_minutes) + ' min)');
    if (health.fcst_stale) warnings.push('Forecasts (' + Math.round(health.fcst_age_minutes) + ' min)');
    var banner = document.getElementById('stale-banner');
    if (!banner) return;
    if (warnings.length > 0) {
        banner.textContent = 'Stale data: ' + warnings.join(', ');
        banner.classList.remove('hidden');
    } else {
        banner.classList.add('hidden');
    }
}

// -----------------------------------------------------------------------
// Refresh Loop
// -----------------------------------------------------------------------

function refreshAll() {
    refreshTempChart();
    refreshObsFeed();
    refreshFcstFeed();
    refreshBracketChart();
    refreshPositions();
    checkStaleness();
    countdown = 60;
}

// Hook into base.html's date toggle
function onDateChange() {
    refreshAll();
}

// Show loading skeletons before first fetch
showSkeleton('obs-feed', 6);
showSkeleton('fcst-feed', 5);
showSkeleton('positions-panel', 8);

// Initial load
refreshAll();

// Auto-refresh every 60 seconds
setInterval(refreshAll, 60000);
