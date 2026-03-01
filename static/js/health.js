/**
 * AlphaTemp — Health page: data freshness, pipeline status, DB stats, config.
 * Fetches /api/health/detailed and renders four panels.
 */

var STALE_THRESHOLD = 30;
var NWS_STALE_THRESHOLD = 1440;

function statusDot(ageMin, threshold) {
    if (ageMin === null) return '<span class="text-slate-600">●</span>';
    if (ageMin <= threshold) return '<span class="text-emerald-400">●</span>';
    if (ageMin <= threshold * 2) return '<span class="text-amber-400">●</span>';
    return '<span class="text-red-400">●</span>';
}

function formatAge(ageMin) {
    if (ageMin === null) return 'never';
    if (ageMin < 1) return '<1 min ago';
    if (ageMin < 60) return Math.round(ageMin) + ' min ago';
    if (ageMin < 1440) return Math.round(ageMin / 60) + 'h ago';
    return Math.round(ageMin / 1440) + 'd ago';
}

var FRESHNESS_LABELS = {
    observations_synoptic: 'Observations (Synoptic)',
    observations_awc: 'Observations (AWC/IEM)',
    forecasts_hrrr: 'HRRR Forecasts',
    market_ticks: 'Market Ticks (Kalshi)',
    drift_signals: 'Drift Signals',
    nws_cli: 'NWS Settlement (CLI)',
    nws_dsm: 'NWS Settlement (DSM)',
};

async function refreshHealth() {
    var data = await fetchAPI('/api/health/detailed');
    if (!data) return;

    // Freshness
    var html = '';
    for (var key in FRESHNESS_LABELS) {
        var f = data.freshness[key] || {};
        var threshold = key.startsWith('nws_') ? NWS_STALE_THRESHOLD : STALE_THRESHOLD;
        html += '<div class="flex justify-between items-center py-1 border-b border-slate-700/50">'
            + '<span class="text-slate-400">' + FRESHNESS_LABELS[key] + '</span>'
            + '<span>' + formatAge(f.age_minutes) + ' ' + statusDot(f.age_minutes, threshold) + '</span>'
            + '</div>';
    }
    document.getElementById('freshness-table').innerHTML = html;

    // Pipelines
    html = '';
    var pipelines = data.pipelines || {};
    var pipelineKeys = Object.keys(pipelines);
    if (pipelineKeys.length === 0) {
        html = '<div class="text-slate-600">No heartbeats received yet</div>';
    } else {
        for (var i = 0; i < pipelineKeys.length; i++) {
            var name = pipelineKeys[i];
            var p = pipelines[name];
            var pDot = p.status === 'ok'
                ? '<span class="text-emerald-400">●</span>'
                : '<span class="text-red-400">●</span>';
            html += '<div class="flex justify-between items-center py-1 border-b border-slate-700/50">'
                + '<span class="text-slate-400">' + name + '</span>'
                + '<span class="text-slate-500">last cycle: ' + p.duration_ms + 'ms</span>'
                + '<span>' + (p.status === 'ok' ? 'Running' : (p.error || 'Error')) + ' ' + pDot + '</span>'
                + '</div>';
        }
    }
    document.getElementById('pipeline-table').innerHTML = html;

    // Database
    html = '';
    var tables = data.database.tables || {};
    for (var tbl in tables) {
        html += '<div class="flex justify-between items-center py-1 border-b border-slate-700/50">'
            + '<span class="text-slate-400">' + tbl + '</span>'
            + '<span class="text-slate-300">' + tables[tbl].rows.toLocaleString() + ' rows</span>'
            + '</div>';
    }
    document.getElementById('db-table').innerHTML = html;
    document.getElementById('db-size').textContent = 'DB size: ' + (data.database.size || 'unknown');

    // Config
    html = '';
    var cfg = data.config || {};
    var configRows = [
        ['Settlement station', cfg.settlement_station],
        ['Neighbor stations', (cfg.neighbor_stations || []).join(', ')],
        ['Stale threshold', cfg.stale_threshold_min + ' min'],
        ['Model', cfg.model],
        ['Bias TTL', cfg.bias_ttl_min + ' min'],
    ];
    var intervals = cfg.polling_intervals || {};
    for (var svc in intervals) {
        configRows.push(['Poll: ' + svc, intervals[svc] + 's']);
    }
    for (var j = 0; j < configRows.length; j++) {
        html += '<div class="flex justify-between items-center py-1 border-b border-slate-700/50">'
            + '<span class="text-slate-400">' + configRows[j][0] + '</span>'
            + '<span class="text-slate-300">' + configRows[j][1] + '</span>'
            + '</div>';
    }
    document.getElementById('config-table').innerHTML = html;
}

// Show loading skeletons before first fetch
showSkeleton('freshness-table', 7);
showSkeleton('pipeline-table', 3);
showSkeleton('db-table', 5);
showSkeleton('config-table', 6);

refreshHealth();
setInterval(refreshHealth, 60000);
