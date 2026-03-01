/**
 * AlphaTemp — Performance tab
 *
 * Renders cumulative P&L, daily P&L bars, Brier score comparison,
 * edge heatmap, and breakdown stats (summary, by bracket, by edge).
 *
 * Depends on: shared.js (COLORS, PLOTLY_LAYOUT, PLOTLY_CONFIG, fetchAPI, formatPnL)
 * Depends on: base.html globals (selectedCity, selectedDate)
 */

// -----------------------------------------------------------------------
// State
// -----------------------------------------------------------------------

var selectedRange = '30d';

/**
 * Update the selected date range, toggle button styles, and refresh.
 */
function setRange(r) {
    selectedRange = r;

    var ranges = ['7d', '30d', 'all'];
    ranges.forEach(function(id) {
        var btn = document.getElementById('range-' + id);
        if (!btn) return;
        if (id === r) {
            btn.className = 'px-3 py-1 text-xs uppercase tracking-wider rounded border border-slate-600 bg-slate-700 text-slate-100';
        } else {
            btn.className = 'px-3 py-1 text-xs uppercase tracking-wider rounded border border-slate-600 bg-slate-700 text-slate-400 hover:bg-slate-600';
        }
    });

    refreshPerformance();
}

// -----------------------------------------------------------------------
// Data Fetch & Dispatch
// -----------------------------------------------------------------------

function refreshPerformance() {
    Promise.all([
        fetchAPI('/api/performance?range=' + selectedRange),
        fetchAPI('/api/brier-comparison?range=' + selectedRange),
        fetchAPI('/api/edge-heatmap?range=' + selectedRange),
    ]).then(function(results) {
        renderCumulativePnL(results[0]);
        renderDailyBars(results[0]);
        renderBrierComparison(results[1]);
        renderEdgeHeatmap(results[2]);
        renderBreakdown(results[0]);
        renderPaperPnL(results[0]);
    });
}

// -----------------------------------------------------------------------
// Cumulative P&L (line chart)
// -----------------------------------------------------------------------

function renderCumulativePnL(data) {
    var el = document.getElementById('cumulative-pnl-chart');
    if (!el) return;

    if (!data || !data.cumulative_pnl || data.cumulative_pnl.length === 0) {
        Plotly.react('cumulative-pnl-chart', [], Object.assign({}, PLOTLY_LAYOUT, {
            annotations: [{
                text: 'No P&L data yet',
                showarrow: false,
                font: { size: 14, color: COLORS.slate400 },
                xref: 'paper', yref: 'paper', x: 0.5, y: 0.5
            }]
        }), PLOTLY_CONFIG);
        return;
    }

    var dates = data.cumulative_pnl.map(function(p) { return p.date; });
    var pnls = data.cumulative_pnl.map(function(p) { return p.pnl; });

    var lineColor = data.total_net >= 0 ? COLORS.green : COLORS.red;
    var fillColor = data.total_net >= 0 ? 'rgba(16,185,129,0.15)' : 'rgba(239,68,68,0.15)';

    var trace = {
        x: dates,
        y: pnls,
        type: 'scatter',
        mode: 'lines',
        line: { color: lineColor, width: 2 },
        fill: 'tozeroy',
        fillcolor: fillColor,
        hovertemplate: '$%{y:.2f}<extra></extra>'
    };

    var layout = Object.assign({}, PLOTLY_LAYOUT, {
        yaxis: Object.assign({}, PLOTLY_LAYOUT.yaxis, {
            title: { text: 'Net P&L ($)', standoff: 8, font: { size: 10 } }
        }),
        xaxis: Object.assign({}, PLOTLY_LAYOUT.xaxis, {
            type: 'date'
        })
    });

    Plotly.react('cumulative-pnl-chart', [trace], layout, PLOTLY_CONFIG);
}

// -----------------------------------------------------------------------
// Daily P&L (bar chart)
// -----------------------------------------------------------------------

function renderDailyBars(data) {
    var el = document.getElementById('daily-pnl-chart');
    if (!el) return;

    if (!data || !data.daily_bars || data.daily_bars.length === 0) {
        Plotly.react('daily-pnl-chart', [], Object.assign({}, PLOTLY_LAYOUT, {
            annotations: [{
                text: 'No daily data yet',
                showarrow: false,
                font: { size: 14, color: COLORS.slate400 },
                xref: 'paper', yref: 'paper', x: 0.5, y: 0.5
            }]
        }), PLOTLY_CONFIG);
        return;
    }

    var dates = data.daily_bars.map(function(p) { return p.date; });
    var pnls = data.daily_bars.map(function(p) { return p.pnl; });
    var barColors = pnls.map(function(v) { return v >= 0 ? COLORS.green : COLORS.red; });

    var trace = {
        x: dates,
        y: pnls,
        type: 'bar',
        marker: { color: barColors },
        hovertemplate: '$%{y:.2f}<extra></extra>'
    };

    var layout = Object.assign({}, PLOTLY_LAYOUT, {
        yaxis: Object.assign({}, PLOTLY_LAYOUT.yaxis, {
            title: { text: 'Daily P&L ($)', standoff: 8, font: { size: 10 } }
        }),
        xaxis: Object.assign({}, PLOTLY_LAYOUT.xaxis, {
            type: 'date'
        })
    });

    Plotly.react('daily-pnl-chart', [trace], layout, PLOTLY_CONFIG);
}

// -----------------------------------------------------------------------
// Brier Score Comparison (dual line chart)
// -----------------------------------------------------------------------

function renderBrierComparison(data) {
    var el = document.getElementById('brier-chart');
    if (!el) return;

    if (!data || !data.by_date || data.by_date.length === 0) {
        var noteText = (data && data.note) ? data.note : 'No Brier data yet';
        Plotly.react('brier-chart', [], Object.assign({}, PLOTLY_LAYOUT, {
            annotations: [{
                text: noteText,
                showarrow: false,
                font: { size: 12, color: COLORS.slate400 },
                xref: 'paper', yref: 'paper', x: 0.5, y: 0.5
            }]
        }), PLOTLY_CONFIG);
        return;
    }

    var dates = data.by_date.map(function(p) { return p.date; });
    var marketScores = data.by_date.map(function(p) { return p.market_brier; });
    var modelScores = data.by_date.map(function(p) { return p.model_brier; });

    var traces = [];

    // Market Brier — always present for settled markets
    traces.push({
        x: dates,
        y: marketScores,
        type: 'scatter',
        mode: 'lines+markers',
        line: { color: COLORS.amber, width: 2 },
        marker: { size: 4 },
        name: 'Market',
        hovertemplate: '%{y:.3f}<extra>Market</extra>'
    });

    // Model Brier — only if any non-null values exist
    var hasModel = modelScores.some(function(v) { return v != null; });
    if (hasModel) {
        traces.push({
            x: dates,
            y: modelScores,
            type: 'scatter',
            mode: 'lines+markers',
            line: { color: COLORS.blue, width: 2 },
            marker: { size: 4 },
            name: 'Model',
            hovertemplate: '%{y:.3f}<extra>Model</extra>'
        });
    }

    var layout = Object.assign({}, PLOTLY_LAYOUT, {
        showlegend: true,
        legend: { x: 0, y: 1.15, orientation: 'h', font: { size: 10 } },
        yaxis: Object.assign({}, PLOTLY_LAYOUT.yaxis, {
            autorange: 'reversed',
            title: { text: 'Brier Score (lower = better)', standoff: 8, font: { size: 10 } }
        }),
        xaxis: Object.assign({}, PLOTLY_LAYOUT.xaxis, {
            type: 'date'
        })
    });

    Plotly.react('brier-chart', traces, layout, PLOTLY_CONFIG);
}

// -----------------------------------------------------------------------
// Edge Heatmap
// -----------------------------------------------------------------------

function renderEdgeHeatmap(data) {
    var el = document.getElementById('edge-heatmap');
    if (!el) return;

    if (!data || !data.cells || data.cells.length === 0) {
        Plotly.react('edge-heatmap', [], Object.assign({}, PLOTLY_LAYOUT, {
            annotations: [{
                text: 'No edge data yet',
                showarrow: false,
                font: { size: 14, color: COLORS.slate400 },
                xref: 'paper', yref: 'paper', x: 0.5, y: 0.5
            }]
        }), PLOTLY_CONFIG);
        return;
    }

    // Extract unique hours and brackets
    var hourSet = {};
    var bracketSet = {};
    data.cells.forEach(function(c) {
        hourSet[c.hour_et] = true;
        bracketSet[c.bracket] = true;
    });

    var hours = Object.keys(hourSet).map(Number).sort(function(a, b) { return a - b; });
    var brackets = Object.keys(bracketSet).sort();

    // Build hour labels (e.g. "08", "09")
    var hourLabels = hours.map(function(h) { return String(h).padStart(2, '0'); });

    // Build lookup map: bracket -> hour -> avg_edge
    var lookup = {};
    data.cells.forEach(function(c) {
        if (!lookup[c.bracket]) lookup[c.bracket] = {};
        lookup[c.bracket][c.hour_et] = c.avg_edge;
    });

    // Build z matrix: brackets (rows) x hours (columns)
    var z = brackets.map(function(b) {
        return hours.map(function(h) {
            var val = lookup[b] && lookup[b][h] != null ? lookup[b][h] * 100 : null;
            return val;
        });
    });

    var trace = {
        x: hourLabels,
        y: brackets,
        z: z,
        type: 'heatmap',
        colorscale: [
            [0, COLORS.slate700],
            [0.5, COLORS.amber],
            [1, COLORS.green]
        ],
        colorbar: {
            title: { text: 'Edge %', font: { size: 10 } },
            ticksuffix: '%',
            len: 0.8
        },
        hovertemplate: 'Bracket: %{y}<br>Hour: %{x}:00 ET<br>Edge: %{z:.1f}%<extra></extra>'
    };

    var layout = Object.assign({}, PLOTLY_LAYOUT, {
        xaxis: Object.assign({}, PLOTLY_LAYOUT.xaxis, {
            title: { text: 'Hour (ET)', standoff: 8, font: { size: 10 } }
        }),
        yaxis: Object.assign({}, PLOTLY_LAYOUT.yaxis, {
            title: { text: 'Bracket', standoff: 8, font: { size: 10 } }
        })
    });

    Plotly.react('edge-heatmap', [trace], layout, PLOTLY_CONFIG);
}

// -----------------------------------------------------------------------
// Breakdown Stats (Summary, By Bracket, By Edge)
// -----------------------------------------------------------------------

function renderBreakdown(data) {
    var el = document.getElementById('breakdown-stats');
    if (!el) return;

    if (!data) {
        el.innerHTML = '<p class="text-xs text-slate-500 col-span-3">No data</p>';
        return;
    }

    var html = '';

    // --- Column 1: Summary ---
    html += '<div>';
    html += '<h3 class="text-xs text-slate-400 uppercase tracking-wide mb-2">Summary</h3>';
    html += '<div class="text-xs space-y-1">';

    html += '<div class="flex justify-between">';
    html += '<span class="text-slate-400">Total Trades</span>';
    html += '<span class="text-slate-200">' + (data.total_trades || 0) + '</span>';
    html += '</div>';

    var winPct = data.win_rate != null ? (data.win_rate * 100).toFixed(1) + '%' : '--';
    html += '<div class="flex justify-between">';
    html += '<span class="text-slate-400">Win Rate</span>';
    html += '<span class="text-slate-200">' + winPct + '</span>';
    html += '</div>';

    var grossFmt = formatPnL(data.total_gross);
    html += '<div class="flex justify-between">';
    html += '<span class="text-slate-400">Gross</span>';
    html += '<span class="' + grossFmt.colorClass + '">' + grossFmt.text + '</span>';
    html += '</div>';

    var feesFmt = formatPnL(data.total_fees != null ? -Math.abs(data.total_fees) : null);
    html += '<div class="flex justify-between">';
    html += '<span class="text-slate-400">Fees</span>';
    html += '<span class="text-red-400">' + (data.total_fees != null ? '-$' + Math.abs(data.total_fees).toFixed(2) : '--') + '</span>';
    html += '</div>';

    var netFmt = formatPnL(data.total_net);
    html += '<div class="flex justify-between font-bold">';
    html += '<span class="text-slate-300">Net</span>';
    html += '<span class="' + netFmt.colorClass + '">' + netFmt.text + '</span>';
    html += '</div>';

    html += '</div></div>';

    // --- Column 2: By Bracket ---
    html += '<div>';
    html += '<h3 class="text-xs text-slate-400 uppercase tracking-wide mb-2">By Bracket</h3>';
    html += '<div class="text-xs space-y-1">';

    if (data.by_bracket && data.by_bracket.length > 0) {
        data.by_bracket.forEach(function(b) {
            var pnlFmt = formatPnL(b.pnl);
            html += '<div class="flex justify-between">';
            html += '<span class="text-slate-300 font-mono">' + b.bracket + '\u00b0F</span>';
            html += '<span class="' + pnlFmt.colorClass + '">' + pnlFmt.text + ' <span class="text-slate-500">(' + b.trades + ')</span></span>';
            html += '</div>';
        });
    } else {
        html += '<p class="text-slate-500">No bracket data</p>';
    }

    html += '</div></div>';

    // --- Column 3: By Edge ---
    html += '<div>';
    html += '<h3 class="text-xs text-slate-400 uppercase tracking-wide mb-2">By Edge</h3>';
    html += '<div class="text-xs space-y-1">';

    if (data.by_edge && data.by_edge.length > 0) {
        data.by_edge.forEach(function(e) {
            var pnlFmt = formatPnL(e.pnl);
            var wr = (e.win_rate * 100).toFixed(0) + '%';
            html += '<div class="flex justify-between">';
            html += '<span class="text-slate-300 font-mono">' + e.bucket + '</span>';
            html += '<span class="' + pnlFmt.colorClass + '">' + pnlFmt.text + ' <span class="text-slate-500">(' + wr + ' WR)</span></span>';
            html += '</div>';
        });
    } else {
        html += '<p class="text-slate-500">No edge data</p>';
    }

    html += '</div></div>';

    el.innerHTML = html;
}

// -----------------------------------------------------------------------
// Paper P&L (cumulative line chart from /api/performance)
// -----------------------------------------------------------------------

function renderPaperPnL(data) {
    var el = document.getElementById('paper-pnl-chart');
    if (!el) return;

    if (!data || !data.cumulative_pnl || data.cumulative_pnl.length === 0) {
        Plotly.react('paper-pnl-chart', [], Object.assign({}, PLOTLY_LAYOUT, {
            annotations: [{
                text: 'No paper trading data yet',
                showarrow: false,
                font: { size: 14, color: COLORS.slate400 },
                xref: 'paper', yref: 'paper', x: 0.5, y: 0.5
            }]
        }), PLOTLY_CONFIG);
        updatePaperPnLStats(null);
        return;
    }

    var dates = data.cumulative_pnl.map(function(p) { return p.date; });
    var pnls = data.cumulative_pnl.map(function(p) { return p.pnl; });

    var lineColor = data.total_net >= 0 ? COLORS.green : COLORS.red;
    var fillColor = data.total_net >= 0 ? 'rgba(16,185,129,0.15)' : 'rgba(239,68,68,0.15)';

    var trace = {
        x: dates,
        y: pnls,
        type: 'scatter',
        mode: 'lines',
        line: { color: lineColor, width: 2 },
        fill: 'tozeroy',
        fillcolor: fillColor,
        hovertemplate: '$%{y:.2f}<extra></extra>'
    };

    var layout = Object.assign({}, PLOTLY_LAYOUT, {
        yaxis: Object.assign({}, PLOTLY_LAYOUT.yaxis, {
            title: { text: 'Cumulative P&L ($)', standoff: 8, font: { size: 10 } }
        }),
        xaxis: Object.assign({}, PLOTLY_LAYOUT.xaxis, {
            type: 'date'
        })
    });

    Plotly.react('paper-pnl-chart', [trace], layout, PLOTLY_CONFIG);
    updatePaperPnLStats(data);
}

function updatePaperPnLStats(data) {
    var el = document.getElementById('paper-pnl-stats');
    if (!el) return;

    if (!data || data.total_trades === 0) {
        el.textContent = 'No trades yet';
        return;
    }

    var netFmt = formatPnL(data.total_net);
    var winPct = data.win_rate != null ? (data.win_rate * 100).toFixed(1) + '%' : '--';

    el.innerHTML =
        '<span class="mr-4">Trades: <span class="text-slate-200">' + data.total_trades + '</span></span>' +
        '<span class="mr-4">Win Rate: <span class="text-slate-200">' + winPct + '</span></span>' +
        '<span>Net P&L: <span class="' + netFmt.colorClass + '">' + netFmt.text + '</span></span>';
}

// -----------------------------------------------------------------------
// Refresh Hook
// -----------------------------------------------------------------------

function onDateChange() {
    refreshPerformance();
}

// Initial load
refreshPerformance();
