/**
 * AlphaTemp — Review (Model Autopsy) tab
 *
 * Renders incident cards for model misses and a pattern summary.
 * Supports date-range and category filtering.
 *
 * Depends on: shared.js (COLORS, fetchAPI, formatPnL, formatEdge)
 * Depends on: base.html globals (selectedCity, selectedDate)
 */

// -----------------------------------------------------------------------
// State
// -----------------------------------------------------------------------

var reviewRange = '30d';
var reviewFilter = 'all';

// -----------------------------------------------------------------------
// Range Toggle
// -----------------------------------------------------------------------

/**
 * Update the selected date range, toggle button styles, and refresh.
 */
function setReviewRange(r) {
    reviewRange = r;

    var ranges = ['7d', '30d', 'all'];
    ranges.forEach(function(id) {
        var btn = document.getElementById('rr-' + id);
        if (!btn) return;
        if (id === r) {
            btn.className = 'px-3 py-1 text-xs uppercase tracking-wider rounded border border-slate-600 bg-slate-700 text-slate-100';
        } else {
            btn.className = 'px-3 py-1 text-xs uppercase tracking-wider rounded border border-slate-600 bg-slate-700 text-slate-400 hover:bg-slate-600';
        }
    });

    refreshReview();
}

// -----------------------------------------------------------------------
// Filter Toggle
// -----------------------------------------------------------------------

/**
 * Update the selected filter, toggle button styles, and refresh.
 */
function setFilter(f) {
    reviewFilter = f;

    var filters = ['all', 'worst', 'lost', 'missed'];
    filters.forEach(function(id) {
        var btn = document.getElementById('filter-' + id);
        if (!btn) return;
        if (id === f) {
            btn.className = 'px-3 py-1 text-xs uppercase tracking-wider rounded border border-slate-600 bg-slate-700 text-slate-100';
        } else {
            btn.className = 'px-3 py-1 text-xs uppercase tracking-wider rounded border border-slate-600 bg-slate-700 text-slate-400 hover:bg-slate-600';
        }
    });

    refreshReview();
}

// -----------------------------------------------------------------------
// Data Fetch & Dispatch
// -----------------------------------------------------------------------

function refreshReview() {
    Promise.all([
        fetchAPI('/api/review/incidents?range=' + reviewRange + '&filter=' + reviewFilter),
        fetchAPI('/api/review/patterns?range=' + reviewRange),
    ]).then(function(results) {
        renderIncidents(results[0]);
        renderPatterns(results[1]);
    });
}

// -----------------------------------------------------------------------
// Render Incidents
// -----------------------------------------------------------------------

function renderIncidents(data) {
    var el = document.getElementById('incident-list');
    if (!el) return;

    if (!data || !data.incidents || data.incidents.length === 0) {
        el.innerHTML = '<p class="text-sm text-slate-500 py-8 text-center">'
            + 'No incidents in this range. Either the model is perfect or there\'s no trading data yet.'
            + '</p>';
        return;
    }

    var html = '';

    data.incidents.forEach(function(inc) {
        var pnl = formatPnL(inc.pnl);
        var edge = formatEdge(inc.edge);

        // Severity bar color
        var severityColor;
        if (inc.severity >= 0.7) {
            severityColor = COLORS.red;
        } else if (inc.severity >= 0.4) {
            severityColor = COLORS.amber;
        } else {
            severityColor = COLORS.green;
        }
        var severityWidth = Math.max(5, Math.min(100, (inc.severity || 0) * 100));

        html += '<div class="bg-slate-800 rounded-lg p-4 border border-slate-700">';

        // Top row: date + category | P&L + severity bar
        html += '<div class="flex items-center justify-between mb-2">';
        html += '<div class="flex items-center gap-2">';
        html += '<span class="font-bold text-slate-100">' + (inc.date || '--') + '</span>';
        html += '<span class="px-2 py-0.5 rounded text-xs bg-slate-700 text-slate-300">' + (inc.category || '') + '</span>';
        html += '</div>';
        html += '<div class="flex items-center gap-2">';
        html += '<span class="font-bold ' + pnl.colorClass + '">' + pnl.text + '</span>';
        html += '<div class="w-16 h-1 bg-slate-700 rounded">';
        html += '<div class="h-1 rounded" style="width: ' + severityWidth + '%; background: ' + severityColor + ';"></div>';
        html += '</div>';
        html += '</div>';
        html += '</div>';

        // Middle: bracket, direction, settlement temp
        html += '<div class="text-xs text-slate-400 mb-1">';
        var middleParts = [];
        if (inc.bracket) middleParts.push(inc.bracket);
        if (inc.direction) middleParts.push(inc.direction);
        if (inc.settlement_temp != null) middleParts.push('Settled: ' + inc.settlement_temp + '\u00b0F');
        html += middleParts.join(' &middot; ');
        html += '</div>';

        // Bottom: model prob, market price, edge
        html += '<div class="text-xs flex gap-4 mb-1">';
        html += '<span>Model: <span class="text-blue-400">' + (inc.model_prob != null ? (inc.model_prob * 100).toFixed(1) + '%' : '--') + '</span></span>';
        html += '<span>Market: <span class="text-amber-400">' + (inc.market_price != null ? (inc.market_price * 100).toFixed(1) + '%' : '--') + '</span></span>';
        html += '<span>Edge: <span class="' + edge.colorClass + '">' + edge.text + '</span></span>';
        html += '</div>';

        // Narrative
        if (inc.narrative) {
            html += '<p class="text-slate-300 text-xs mt-1">' + inc.narrative + '</p>';
        }

        html += '</div>';
    });

    el.innerHTML = html;
}

// -----------------------------------------------------------------------
// Render Patterns
// -----------------------------------------------------------------------

function renderPatterns(data) {
    var el = document.getElementById('pattern-summary');
    if (!el) return;

    if (!data || !data.patterns || data.patterns.length === 0) {
        el.innerHTML = '<p class="text-sm text-slate-500">No patterns detected yet</p>';
        return;
    }

    var html = '<ol class="space-y-2 text-sm">';

    data.patterns.forEach(function(pat, idx) {
        var pnl = formatPnL(pat.total_pnl);
        var categoryName = (pat.category || '').replace(/_/g, ' ');
        categoryName = categoryName.charAt(0).toUpperCase() + categoryName.slice(1);

        html += '<li>';
        html += '<div class="flex items-center gap-2">';
        html += '<span class="text-slate-500 font-mono">' + (idx + 1) + '.</span>';
        html += '<span class="text-slate-200 font-bold">' + categoryName + '</span>';
        html += '<span class="' + pnl.colorClass + '">' + pnl.text + '</span>';
        html += '<span class="text-slate-500">(' + (pat.count || 0) + ' incidents)</span>';
        html += '</div>';
        if (pat.suggested_action) {
            html += '<p class="text-xs text-slate-400 ml-6 mt-0.5">' + pat.suggested_action + '</p>';
        }
        html += '</li>';
    });

    html += '</ol>';
    el.innerHTML = html;
}

// -----------------------------------------------------------------------
// Refresh Hook
// -----------------------------------------------------------------------

function onDateChange() {
    refreshReview();
}

// Initial load
refreshReview();
