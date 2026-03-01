/**
 * AlphaTemp — Mobile tab
 *
 * Dashboard + Trade toggle with compact data views
 * optimized for phone screens.
 *
 * Depends on: shared.js (COLORS, toET, fetchAPI, formatPnL, formatEdge)
 * Depends on: base.html globals (selectedCity, selectedDate)
 */

// -----------------------------------------------------------------------
// Tab Toggle
// -----------------------------------------------------------------------

function setMobileTab(tab) {
    var dashboardEl = document.getElementById('mobile-dashboard');
    var tradeEl = document.getElementById('mobile-trade');
    var btnDash = document.getElementById('mob-dashboard');
    var btnTrade = document.getElementById('mob-trade');

    if (tab === 'dashboard') {
        dashboardEl.classList.remove('hidden');
        tradeEl.classList.add('hidden');
        btnDash.className = 'flex-1 py-2 text-sm font-bold uppercase tracking-wider rounded bg-slate-700 text-slate-100';
        btnTrade.className = 'flex-1 py-2 text-sm font-bold uppercase tracking-wider rounded text-slate-400';
    } else {
        dashboardEl.classList.add('hidden');
        tradeEl.classList.remove('hidden');
        btnDash.className = 'flex-1 py-2 text-sm font-bold uppercase tracking-wider rounded text-slate-400';
        btnTrade.className = 'flex-1 py-2 text-sm font-bold uppercase tracking-wider rounded bg-slate-700 text-slate-100';
    }
}

// -----------------------------------------------------------------------
// Refresh
// -----------------------------------------------------------------------

function refreshMobile() {
    var dateParam = selectedDate ? '?date=' + selectedDate : '';

    Promise.all([
        fetchAPI('/api/observations/' + selectedCity + dateParam),
        fetchAPI('/api/forecast-curve/' + selectedCity + dateParam),
        fetchAPI('/api/positions/' + selectedCity + dateParam),
        fetchAPI('/api/market-swings/' + selectedCity + dateParam),
        fetchAPI('/api/brackets/' + selectedCity + dateParam),
    ]).then(function(results) {
        var obsData = results[0];
        var fcstData = results[1];
        var posData = results[2];
        var swingsData = results[3];
        var bracketsData = results[4];

        // -----------------------------------------------------------
        // Current State Card
        // -----------------------------------------------------------

        // Latest obs
        var latestObsEl = document.getElementById('mob-latest-obs');
        if (obsData && obsData.running_high != null) {
            latestObsEl.textContent = obsData.running_high.toFixed(1) + '\u00b0F';
        } else if (obsData && obsData.observations && obsData.observations.length > 0) {
            latestObsEl.textContent = obsData.observations[0].temp_f.toFixed(1) + '\u00b0F';
        } else {
            latestObsEl.textContent = '--';
        }

        // Model high
        var modelHighEl = document.getElementById('mob-model-high');
        if (fcstData && fcstData.forecast_high != null) {
            modelHighEl.textContent = fcstData.forecast_high.toFixed(1) + '\u00b0F';
        } else {
            modelHighEl.textContent = '--';
        }

        // Settlement
        var settlementEl = document.getElementById('mob-settlement');
        if (fcstData && fcstData.observed_high != null) {
            settlementEl.textContent = fcstData.observed_high.toFixed(1) + '\u00b0F';
        } else {
            settlementEl.textContent = '--';
        }

        // Daily P&L (card)
        var dailyPnlEl = document.getElementById('mob-daily-pnl');
        var dailyPnl = posData ? formatPnL(posData.daily_pnl) : formatPnL(null);
        dailyPnlEl.textContent = dailyPnl.text;
        dailyPnlEl.className = 'text-xl font-bold ' + dailyPnl.colorClass;

        // -----------------------------------------------------------
        // Alerts (market swings)
        // -----------------------------------------------------------
        var alertsEl = document.getElementById('mob-alerts');
        if (swingsData && swingsData.swings && swingsData.swings.length > 0) {
            var swings = swingsData.swings.slice(0, 5);
            var alertHtml = swings.map(function(s) {
                var arrow = s.change >= 0 ? '\u2191' : '\u2193';
                var colorCls = s.change >= 0 ? 'text-emerald-400' : 'text-red-400';
                var changeCents = Math.abs(Math.round(s.change * 100));
                return '<div class="flex justify-between text-xs py-1 border-b border-slate-700/50">' +
                    '<span class="text-slate-300">\u26a1 ' + s.bracket + '</span>' +
                    '<span class="' + colorCls + '">' + arrow + ' ' + changeCents + '\u00a2</span>' +
                '</div>';
            }).join('');
            alertsEl.innerHTML = alertHtml;
        } else {
            alertsEl.innerHTML = '<p class="text-xs text-slate-500">No alerts</p>';
        }

        // -----------------------------------------------------------
        // Recent Observations (last 5)
        // -----------------------------------------------------------
        var obsFeedEl = document.getElementById('mob-obs-feed');
        if (obsData && obsData.observations && obsData.observations.length > 0) {
            var recent = obsData.observations.slice(0, 5);
            var obsHtml = recent.map(function(o) {
                return '<div class="flex justify-between text-xs py-1 border-b border-slate-700/50">' +
                    '<span class="text-slate-400">' + toET(o.observed_at) + '</span>' +
                    '<span class="text-slate-400">' + (o.source || '--') + '</span>' +
                    '<span class="text-slate-200">' + o.temp_f.toFixed(1) + '\u00b0F</span>' +
                '</div>';
            }).join('');
            obsFeedEl.innerHTML = obsHtml;
        } else {
            obsFeedEl.innerHTML = '<p class="text-xs text-slate-500">No observations</p>';
        }

        // -----------------------------------------------------------
        // Suggested Bets (edge >= 0.10)
        // -----------------------------------------------------------
        var suggestedEl = document.getElementById('mob-suggested-bets');
        if (bracketsData && bracketsData.brackets && bracketsData.brackets.length > 0) {
            var bets = bracketsData.brackets
                .filter(function(b) { return b.edge != null && b.edge >= 0.10; })
                .sort(function(a, b) { return b.edge - a.edge; });

            if (bets.length > 0) {
                var betsHtml = bets.map(function(b) {
                    var label = b.floor + '-' + b.cap + '\u00b0F';
                    var confidence = b.edge >= 0.15 ? 'HIGH' : 'MED';
                    var confColor = b.edge >= 0.15 ? 'text-emerald-400' : 'text-amber-400';
                    var modelPct = (b.model_prob * 100).toFixed(1);
                    var kalshiPct = b.market_mid != null ? (b.market_mid * 100).toFixed(1) : '--';
                    var edgePct = (b.edge * 100).toFixed(1);
                    var netEdge = ((b.edge - 0.11) * 100).toFixed(1);
                    var askPrice = b.ask != null ? (b.ask * 100).toFixed(0) + '\u00a2' : '--';
                    var volume = b.volume != null ? b.volume : 0;

                    var html = '<div class="bg-slate-700/50 rounded border border-slate-600 p-3 mb-2">';
                    html += '<div class="flex justify-between items-center mb-2">';
                    html += '<span class="text-slate-100 font-mono text-sm font-bold">' + label + ' BUY YES</span>';
                    html += '<span class="text-xs font-bold ' + confColor + '">' + confidence + '</span>';
                    html += '</div>';
                    html += '<div class="grid grid-cols-2 gap-1 text-[10px]">';
                    html += '<div><span class="text-slate-500">Model:</span> <span class="text-slate-200">' + modelPct + '%</span></div>';
                    html += '<div><span class="text-slate-500">Kalshi:</span> <span class="text-slate-200">' + kalshiPct + '%</span></div>';
                    html += '<div><span class="text-slate-500">Edge:</span> <span class="text-emerald-400">' + edgePct + '%</span></div>';
                    html += '<div><span class="text-slate-500">Net edge:</span> <span class="text-slate-200">' + netEdge + '%</span></div>';
                    html += '<div><span class="text-slate-500">Ask:</span> <span class="text-slate-200">' + askPrice + '</span></div>';
                    html += '<div><span class="text-slate-500">Vol:</span> <span class="text-slate-200">' + volume + '</span></div>';
                    html += '</div>';
                    if (volume < 100) {
                        html += '<div class="text-[10px] text-amber-400 mt-1">\u26a0 Thin liquidity</div>';
                    }
                    html += '</div>';
                    return html;
                }).join('');
                suggestedEl.innerHTML = betsHtml;
            } else {
                suggestedEl.innerHTML = '<p class="text-xs text-slate-500">No bets meeting threshold</p>';
            }
        } else {
            suggestedEl.innerHTML = '<p class="text-xs text-slate-500">No bets meeting threshold</p>';
        }

        // -----------------------------------------------------------
        // Near Threshold
        // -----------------------------------------------------------
        var nearEl = document.getElementById('mob-near-threshold');
        if (posData && posData.near_misses && posData.near_misses.length > 0) {
            var nearHtml = posData.near_misses.map(function(nm) {
                var edgePct = (nm.edge * 100).toFixed(1);
                var threshPct = (nm.threshold * 100).toFixed(0);
                return '<div class="flex justify-between text-xs py-1 border-b border-slate-700/50">' +
                    '<span class="text-amber-400 font-mono">' + nm.bracket + '</span>' +
                    '<span class="text-amber-400">' + edgePct + '% / ' + threshPct + '%</span>' +
                '</div>';
            }).join('');
            nearEl.innerHTML = nearHtml;
        } else {
            nearEl.innerHTML = '<p class="text-xs text-slate-500">No near-misses</p>';
        }

        // -----------------------------------------------------------
        // Active Positions
        // -----------------------------------------------------------
        var activeEl = document.getElementById('mob-active-positions');
        if (posData && posData.active && posData.active.length > 0) {
            var activeHtml = posData.active.map(function(bet) {
                var entryStr = bet.entry_price != null ? (bet.entry_price * 100).toFixed(0) + '\u00a2' : '--';
                return '<div class="flex justify-between text-xs py-1 border-b border-slate-700/50">' +
                    '<span class="text-slate-100 font-mono">' + bet.bracket + ' ' + (bet.direction || '--') + '</span>' +
                    '<span class="text-slate-400">Entry: ' + entryStr + '</span>' +
                '</div>';
            }).join('');
            activeEl.innerHTML = activeHtml;
        } else {
            activeEl.innerHTML = '<p class="text-xs text-slate-500">No active positions</p>';
        }

        // -----------------------------------------------------------
        // P&L Footer
        // -----------------------------------------------------------
        var footerDailyEl = document.getElementById('mob-footer-daily-pnl');
        var footerTotalEl = document.getElementById('mob-footer-total-pnl');

        var fDaily = posData ? formatPnL(posData.daily_pnl) : formatPnL(null);
        var fTotal = posData ? formatPnL(posData.total_pnl) : formatPnL(null);

        footerDailyEl.textContent = fDaily.text;
        footerDailyEl.className = 'text-sm font-bold ' + fDaily.colorClass;

        footerTotalEl.textContent = fTotal.text;
        footerTotalEl.className = 'text-sm font-bold ' + fTotal.colorClass;

        // Reset countdown
        countdown = 60;
    });
}

// -----------------------------------------------------------------------
// Blotter Summary
// -----------------------------------------------------------------------

async function refreshMobileBlotter() {
    var date = getTargetDate(window.selectedDate || 'today');
    var data = await fetchAPI('/api/blotter/nyc?date=' + date);
    if (!data) return;
    var el = document.getElementById('mobile-blotter');
    if (!el) return;
    var positions = data.positions || [];
    var open = positions.filter(function(p) { return p.status === 'open'; });
    if (open.length === 0) {
        el.innerHTML = '<div class="text-slate-600 text-xs">No open positions</div>';
        return;
    }
    var html = '';
    open.forEach(function(p) {
        var pnl = formatPnL(p.pnl || 0);
        html += '<div class="flex justify-between py-1 border-b border-slate-800/50 text-xs">'
            + '<span>' + p.bracket_floor + '-' + p.bracket_cap + '\u00b0F ' + p.direction + '</span>'
            + '<span class="' + pnl.colorClass + '">' + pnl.text + '</span>'
            + '</div>';
    });
    el.innerHTML = html;
}

// -----------------------------------------------------------------------
// Hooks
// -----------------------------------------------------------------------

function onDateChange() {
    refreshMobile();
    refreshMobileBlotter();
}

// Show loading skeletons before first fetch
showSkeleton('mob-alerts', 3);
showSkeleton('mob-obs-feed', 4);
showSkeleton('mob-suggested-bets', 3);
showSkeleton('mob-active-positions', 3);

// Initial load
refreshMobile();
refreshMobileBlotter();
