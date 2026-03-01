/**
 * AlphaTemp — Blotter page: settlement countdown + bracket grid with positions.
 *
 * Depends on: shared.js (fetchAPI, formatPnL, formatEdge, toET, getTargetDate)
 */

var countdownInterval = null;
var closeTimes = {};

function startCountdownTicker() {
    if (countdownInterval) clearInterval(countdownInterval);
    countdownInterval = setInterval(function() {
        var now = Date.now();
        Object.keys(closeTimes).forEach(function(suffix) {
            var el = document.getElementById('countdown-timer-' + suffix);
            if (!el) return;
            var diff = new Date(closeTimes[suffix]).getTime() - now;
            if (diff <= 0) {
                el.textContent = 'Settled';
                return;
            }
            var h = Math.floor(diff / 3600000);
            var m = Math.floor((diff % 3600000) / 60000);
            var s = Math.floor((diff % 60000) / 1000);
            el.textContent = h + 'h ' + m + 'm ' + s + 's';
        });
    }, 1000);
}

function renderCountdown(cd, elId, suffix) {
    var el = document.getElementById(elId);
    if (!el || !cd) { if (el) el.innerHTML = ''; return; }

    closeTimes[suffix] = cd.close_time;

    var settlementText = 'Pending';
    if (cd.settlement && cd.settlement.source === 'NWS_CLI') {
        settlementText = cd.settlement.temp + '\u00B0F (CLI \u2014 Final)';
    } else if (cd.settlement && cd.settlement.source === 'DSM') {
        settlementText = cd.settlement.temp + '\u00B0F (DSM)';
    }

    var obsMaxText = cd.running_obs_max !== null
        ? cd.running_obs_max + '\u00B0F'
        : '--';
    if (cd.obs_max_time) {
        obsMaxText += ' (as of ' + toET(cd.obs_max_time) + ')';
    }

    el.innerHTML =
        '<div class="flex items-baseline gap-4 mb-2">'
        + '<span class="text-slate-300 font-bold">KXHIGHNY</span>'
        + '<span class="text-slate-500">' + cd.event_date + '</span>'
        + '<span class="text-white">Settles in: <strong id="countdown-timer-' + suffix + '">--</strong></span>'
        + '</div>'
        + '<div class="grid grid-cols-2 gap-x-8 gap-y-1 text-slate-400">'
        + '<span>Model High: <strong class="text-white">'
            + (cd.model_high ? cd.model_high + '\u00B0F' : '--') + '</strong>'
            + (cd.model_bracket ? ' \u2192 ' + cd.model_bracket + '\u00B0F ('
                + Math.round((cd.model_bracket_prob || 0) * 100) + '%)' : '')
        + '</span>'
        + '<span>Running Obs Max: <strong class="text-white">' + obsMaxText + '</strong></span>'
        + '<span>Settlement: <strong class="text-white">' + settlementText + '</strong></span>'
        + '</div>';

    startCountdownTicker();
}

function renderBlotter(brackets, positions) {
    var body = document.getElementById('blotter-body');
    if (!body) return;

    // Index positions by bracket key "floor-cap"
    var posMap = {};
    (positions || []).forEach(function(p) {
        posMap[p.bracket_floor + '-' + p.bracket_cap] = p;
    });

    var html = '';
    var bracketList = (brackets && brackets.brackets) ? brackets.brackets : [];
    bracketList.forEach(function(b) {
        var key = b.floor + '-' + b.cap;
        var pos = posMap[key];
        var hasPos = pos && pos.status === 'open';
        var rowClass = hasPos ? 'text-white' : 'text-slate-500';
        var indicator = hasPos ? '<span class="text-emerald-400 mr-1">\u25CF</span>' : '';

        var modelProb = b.model_prob || 0;
        // market_mid is a decimal (0-1); display as cents
        var marketMid = b.market_mid;
        var marketDisplay = marketMid !== null && marketMid !== undefined
            ? Math.round(marketMid * 100) + '\u00A2' : '--';

        // edge is model_prob - market_mid (decimal); formatEdge expects decimal
        var edgeFmt = formatEdge(b.edge);

        html += '<tr class="' + rowClass + ' border-b border-slate-800/50">'
            + '<td class="py-1.5">' + indicator + b.floor + '-' + b.cap + '\u00B0F</td>'
            + '<td class="text-right">' + (modelProb * 100).toFixed(1) + '%</td>'
            + '<td class="text-right">' + marketDisplay + '</td>'
            + '<td class="text-right ' + edgeFmt.colorClass + '">' + edgeFmt.text + '</td>'
            + '<td class="text-center">' + (pos ? pos.direction : '\u2014') + '</td>'
            + '<td class="text-right">' + (pos ? pos.contracts : '\u2014') + '</td>'
            + '<td class="text-right">' + (pos && pos.entry_price != null ? Math.round(pos.entry_price * 100) + '\u00A2' : '\u2014') + '</td>'
            + '<td class="text-right">'
                + (pos ? formatPnL(pos.pnl || 0).text : '\u2014')
            + '</td>'
            + '</tr>';
    });
    body.innerHTML = html;
}

function renderSummary(summary) {
    var el = document.getElementById('blotter-summary');
    if (!el || !summary) return;
    var pnl = formatPnL(summary.day_pnl || 0);
    el.innerHTML =
        'Open: <strong class="text-white">' + summary.open_count + '</strong>'
        + ' \u00B7 Exposure: <strong class="text-white">$' + (summary.day_exposure || 0).toFixed(2) + '</strong>'
        + ' \u00B7 Day P&L: <strong class="' + pnl.colorClass + '">' + pnl.text + '</strong>';
}

async function refreshBlotter() {
    var date = getTargetDate(window.selectedDate || 'today');
    var data = await fetchAPI('/api/blotter/nyc?date=' + date);
    if (!data) return;

    renderCountdown(data.countdown, 'countdown-today', 'today');
    renderBlotter(data.brackets, data.positions);
    renderSummary(data.positions_summary);

    // Also fetch tomorrow for the second countdown row
    var tmrw = getTargetDate('tomorrow');
    var tmrwData = await fetchAPI('/api/blotter/nyc?date=' + tmrw);
    if (tmrwData && tmrwData.countdown) {
        renderCountdown(tmrwData.countdown, 'countdown-tomorrow', 'tomorrow');
    }
}

function onDateChange() {
    refreshBlotter();
}

// Show loading skeletons before first fetch
showSkeleton('countdown-today', 2);
showSkeleton('blotter-body', 8);

refreshBlotter();
setInterval(refreshBlotter, 60000);
