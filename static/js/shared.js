/**
 * AlphaTemp — Shared utilities for all dashboard tabs.
 *
 * Provides: time conversion, date helpers, fetch wrapper,
 * formatting helpers, Plotly defaults, and color palette.
 */

// -----------------------------------------------------------------------
// Color Palette
// -----------------------------------------------------------------------
var COLORS = {
    green:    '#10b981',
    red:      '#ef4444',
    amber:    '#f59e0b',
    blue:     '#3b82f6',
    teal:     '#2dd4bf',
    white:    '#f1f5f9',
    slate400: '#94a3b8',
    slate500: '#64748b',
    slate700: '#334155',
};

// -----------------------------------------------------------------------
// Plotly Defaults
// -----------------------------------------------------------------------
var PLOTLY_LAYOUT = {
    paper_bgcolor: '#0f172a',
    plot_bgcolor:  '#0f172a',
    font: {
        family: 'Space Mono, monospace',
        color:  '#94a3b8',
        size:   11,
    },
    margin: { t: 10, r: 20, b: 40, l: 50 },
    xaxis: {
        gridcolor: 'rgba(51,65,85,0.5)',
        zeroline: false,
    },
    yaxis: {
        gridcolor: 'rgba(51,65,85,0.5)',
        zeroline: false,
    },
};

var PLOTLY_CONFIG = {
    displayModeBar: false,
    responsive: true,
};

// -----------------------------------------------------------------------
// Time Conversion
// -----------------------------------------------------------------------

/**
 * Convert a UTC ISO string to an Eastern Time display string.
 *
 * @param {string} utcIso  — ISO timestamp (with or without trailing Z)
 * @param {object} opts    — Intl.DateTimeFormat options override
 * @returns {string}       — Formatted ET string
 */
function toET(utcIso, opts) {
    if (!utcIso) return '--';
    // Ensure the string is treated as UTC
    var input = utcIso;
    if (input.indexOf('Z') === -1 && input.indexOf('+') === -1 && input.indexOf('T') !== -1) {
        input = input + 'Z';
    }
    var d = new Date(input);
    if (isNaN(d.getTime())) return '--';

    var defaults = {
        timeZone: 'America/New_York',
        hour: '2-digit',
        minute: '2-digit',
        hour12: false,
    };
    var merged = Object.assign({}, defaults, opts || {});
    return d.toLocaleString('en-US', merged);
}

// -----------------------------------------------------------------------
// Date Helpers
// -----------------------------------------------------------------------

/**
 * Get a YYYY-MM-DD string for 'today' or 'tomorrow' in Eastern Time.
 *
 * @param {string} which — 'today' or 'tomorrow'
 * @returns {string}     — YYYY-MM-DD in ET
 */
function getTargetDate(which) {
    var d = new Date();
    if (which === 'tomorrow') {
        d.setDate(d.getDate() + 1);
    }
    return d.toLocaleDateString('en-CA', { timeZone: 'America/New_York' });
}

// -----------------------------------------------------------------------
// Fetch Helper
// -----------------------------------------------------------------------

/**
 * Fetch JSON from a URL with error handling. Returns null on failure.
 *
 * @param {string} url
 * @returns {Promise<object|null>}
 */
function fetchAPI(url) {
    return fetch(url)
        .then(function(r) {
            if (!r.ok) {
                console.error('fetchAPI non-OK:', r.status, url);
                return null;
            }
            return r.json();
        })
        .catch(function(e) {
            console.error('fetchAPI error:', url, e);
            return null;
        });
}

// -----------------------------------------------------------------------
// Formatting Helpers
// -----------------------------------------------------------------------

/**
 * Format a P&L dollar amount with color class.
 *
 * @param {number} amount — Dollar amount (positive = profit)
 * @returns {{text: string, colorClass: string}}
 */
function formatPnL(amount) {
    if (amount == null) return { text: '--', colorClass: 'text-slate-500' };
    var sign = amount >= 0 ? '+' : '';
    var text = sign + '$' + Math.abs(amount).toFixed(2);
    var colorClass = amount >= 0 ? 'text-emerald-400' : 'text-red-400';
    return { text: text, colorClass: colorClass };
}

/**
 * Format an edge percentage with color class.
 * Green >= 10%, amber >= 0%, red < 0%.
 *
 * @param {number} edge — Edge as a decimal (e.g. 0.12 = 12%)
 * @returns {{text: string, colorClass: string}}
 */
function formatEdge(edge) {
    if (edge == null) return { text: '--', colorClass: 'text-slate-500' };
    var pct = (edge * 100).toFixed(1);
    var sign = edge >= 0 ? '+' : '';
    var text = sign + pct + '%';
    var colorClass;
    if (edge >= 0.10) {
        colorClass = 'text-emerald-400';
    } else if (edge >= 0) {
        colorClass = 'text-amber-400';
    } else {
        colorClass = 'text-red-400';
    }
    return { text: text, colorClass: colorClass };
}

// -----------------------------------------------------------------------
// DOM Helpers
// -----------------------------------------------------------------------

/**
 * Set the text content of an element by ID.
 *
 * @param {string} id   — Element ID
 * @param {*}      text — Value to display
 */
function setText(id, text) {
    var el = document.getElementById(id);
    if (el) el.textContent = text;
}

/**
 * Set text content and CSS class of an element by ID.
 *
 * @param {string} id         — Element ID
 * @param {string} text       — Value to display
 * @param {string} colorClass — Tailwind color class
 */
function setColoredText(id, text, colorClass) {
    var el = document.getElementById(id);
    if (el) {
        el.textContent = text;
        el.className = el.className.replace(/text-\S+/g, '').trim() + ' ' + colorClass;
    }
}

// -----------------------------------------------------------------------
// KPI Header Refresh
// -----------------------------------------------------------------------

/**
 * Fetch KPI summary and update all header bar elements.
 * Called on page load and every 60s from base.html.
 */
async function refreshKPI() {
    var data = await fetchAPI('/api/kpi-summary?city=' + (window.selectedCity || 'nyc'));
    if (!data) return;

    var dot = document.getElementById('kpi-status-dot');
    if (dot) {
        dot.style.background = data.system_status === 'green' ? '#10b981' :
                               data.system_status === 'amber' ? '#f59e0b' : '#ef4444';
    }

    setText('kpi-model-high', data.model_high ? data.model_high + '\u00B0F' : '--');

    var stl = data.settlement || {};
    if (stl.source === 'NWS_CLI') {
        setText('kpi-settlement', stl.temp + '\u00B0F (CLI)');
    } else if (stl.source === 'DSM') {
        setText('kpi-settlement', stl.temp + '\u00B0F (DSM)');
    } else {
        setText('kpi-settlement', 'Pending');
    }

    setText('kpi-consensus', data.market_consensus || '--');
    setText('kpi-drift', data.drift !== null ? (data.drift > 0 ? '+' : '') + data.drift + '\u00B0F' : '--');
    setText('kpi-open-pos', data.open_positions);

    var dayPnl = formatPnL(data.day_pnl || 0);
    var totalPnl = formatPnL(data.total_pnl || 0);
    setColoredText('kpi-day-pnl', dayPnl.text, dayPnl.colorClass);
    setColoredText('kpi-total-pnl', totalPnl.text, totalPnl.colorClass);
}
