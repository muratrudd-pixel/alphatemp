/**
 * AlphaTemp — Blotter: calendar P&L + risk console + sortable positions.
 *
 * Depends on: shared.js (fetchAPI, formatPnL, toET, getTargetDate)
 */

// -----------------------------------------------------------------------
// State
// -----------------------------------------------------------------------

var blotterRange = "30d";
var calendarMonth = new Date();
var calendarData = {};
var selectedBlotterDate = null;
var countdownInterval = null;
var closeTimeBlotter = null;

// Cached position data for sorting without re-fetching
var cachedLivePositions = [];
var cachedClosedPositions = [];
var cachedBrackets = null;

// Sort state per table — default to entry_time descending (newest first)
var sortState = {
  live: { key: "entry_time", dir: "desc" },
  closed: { key: "entry_time", dir: "desc" },
};

// -----------------------------------------------------------------------
// Helpers
// -----------------------------------------------------------------------

function pnlColor(val) {
  if (val > 0) return "text-emerald-600";
  if (val < 0) return "text-red-600";
  return "text-gray-400";
}

function fmtDollars(val) {
  if (val == null) return "--";
  var sign = val >= 0 ? "+" : "";
  return sign + "$" + Math.abs(val).toFixed(2);
}

function fmtCents(val) {
  if (val == null) return "--";
  // entry_price/exit_price are stored in cents (0-100), not fractions
  return Math.round(val) + "\u00a2";
}

function sideClass(dir) {
  if (dir === "YES") return "text-blue-600 font-medium";
  if (dir === "NO") return "text-orange-600 font-medium";
  return "text-gray-500";
}

function todayET() {
  return new Date().toLocaleDateString("en-CA", {
    timeZone: "America/New_York",
  });
}

function fmtTime(isoStr) {
  if (!isoStr) return "--";
  return toET(isoStr);
}

function getETDate(isoStr) {
  // Extract YYYY-MM-DD in ET from an ISO timestamp
  if (!isoStr) return null;
  var input = isoStr;
  if (
    input.indexOf("Z") === -1 &&
    input.indexOf("+") === -1 &&
    input.indexOf("T") !== -1
  ) {
    input = input + "Z";
  }
  var d = new Date(input);
  if (isNaN(d.getTime())) return null;
  return d.toLocaleDateString("en-CA", { timeZone: "America/New_York" });
}

// -----------------------------------------------------------------------
// Sorting
// -----------------------------------------------------------------------

function sortBy(table, key) {
  var s = sortState[table];
  if (s.key === key) {
    s.dir = s.dir === "asc" ? "desc" : "asc";
  } else {
    s.key = key;
    s.dir = "asc";
  }

  // Update header indicators
  var tableEl = document.getElementById(table + "-table");
  if (tableEl) {
    tableEl.querySelectorAll("th.sortable").forEach(function (th) {
      th.classList.remove("sort-asc", "sort-desc");
      if (th.dataset.key === key) {
        th.classList.add(s.dir === "asc" ? "sort-asc" : "sort-desc");
      }
    });
  }

  if (table === "live") {
    renderLivePositions(cachedLivePositions, cachedBrackets);
  } else {
    renderClosedPositions(cachedClosedPositions);
  }
}

function applySortToList(list, table) {
  var s = sortState[table];
  var sorted = list.slice();

  // Always pin prior-day entries to the bottom
  sorted.sort(function (a, b) {
    var aPrior = a._sort && a._sort._isPriorDay ? 1 : 0;
    var bPrior = b._sort && b._sort._isPriorDay ? 1 : 0;
    if (aPrior !== bPrior) return aPrior - bPrior;

    if (!s.key) return 0;
    var key = s.key;
    var mult = s.dir === "asc" ? 1 : -1;
    var aVal =
      a._sort && a._sort[key] != null
        ? a._sort[key]
        : a[key] != null
          ? a[key]
          : "";
    var bVal =
      b._sort && b._sort[key] != null
        ? b._sort[key]
        : b[key] != null
          ? b[key]
          : "";
    if (typeof aVal === "string") return mult * aVal.localeCompare(bVal);
    return mult * ((aVal || 0) - (bVal || 0));
  });
  return sorted;
}

// Attach click handlers to sortable headers
document.addEventListener("DOMContentLoaded", function () {
  document.querySelectorAll("th.sortable").forEach(function (th) {
    th.addEventListener("click", function () {
      sortBy(th.dataset.table, th.dataset.key);
    });
  });
});

// -----------------------------------------------------------------------
// Range Selector
// -----------------------------------------------------------------------

function setBlotterRange(r) {
  blotterRange = r;
  ["7d", "30d", "90d", "all"].forEach(function (id) {
    var btn = document.getElementById("br-" + id);
    if (!btn) return;
    btn.className =
      id === r
        ? "px-3 py-1 text-xs rounded bg-gray-900 text-white"
        : "px-3 py-1 text-xs rounded bg-gray-100 text-gray-500";
  });
  refreshCalendar();
}

// -----------------------------------------------------------------------
// Calendar Widget
// -----------------------------------------------------------------------

function calendarPrev() {
  calendarMonth.setMonth(calendarMonth.getMonth() - 1);
  renderCalendar();
}

function calendarNext() {
  calendarMonth.setMonth(calendarMonth.getMonth() + 1);
  renderCalendar();
}

function selectCalendarDate(dateStr) {
  selectedBlotterDate = dateStr;
  renderCalendar();
  refreshBlotterData(dateStr);
}

function renderCalendar() {
  var el = document.getElementById("calendar-grid");
  var labelEl = document.getElementById("cal-month-label");
  var pnlEl = document.getElementById("cal-month-pnl");
  if (!el) return;

  var year = calendarMonth.getFullYear();
  var month = calendarMonth.getMonth();
  var monthNames = [
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
  ];
  labelEl.textContent = monthNames[month] + " " + year;

  var monthPnl = 0;
  Object.keys(calendarData).forEach(function (d) {
    var parts = d.split("-");
    if (parseInt(parts[0]) === year && parseInt(parts[1]) === month + 1) {
      monthPnl += calendarData[d].net_pnl || 0;
    }
  });
  pnlEl.textContent = fmtDollars(monthPnl);
  pnlEl.className = "text-sm font-bold at-mono " + pnlColor(monthPnl);

  var firstDay = new Date(year, month, 1).getDay();
  var daysInMonth = new Date(year, month + 1, 0).getDate();
  var today = todayET();

  var html =
    '<div class="grid grid-cols-7 gap-0 text-center text-[10px] text-gray-400 mb-1">';
  ["S", "M", "T", "W", "T", "F", "S"].forEach(function (d) {
    html += "<div>" + d + "</div>";
  });
  html += "</div>";
  html += '<div class="grid grid-cols-7 gap-0.5">';

  for (var i = 0; i < firstDay; i++) html += '<div class="h-9"></div>';

  for (var day = 1; day <= daysInMonth; day++) {
    var dateStr =
      year +
      "-" +
      String(month + 1).padStart(2, "0") +
      "-" +
      String(day).padStart(2, "0");
    var dayData = calendarData[dateStr];
    var isToday = dateStr === today;
    var isSelected = dateStr === selectedBlotterDate;

    var bgClass = "";
    var pnlText = "";
    if (dayData && dayData.net_pnl !== 0) {
      if (dayData.net_pnl > 0) {
        bgClass = isSelected ? "bg-emerald-200" : "bg-emerald-50";
        pnlText =
          '<div class="text-[9px] text-emerald-700 font-medium at-mono">+' +
          Math.abs(dayData.net_pnl).toFixed(0) +
          "</div>";
      } else {
        bgClass = isSelected ? "bg-red-200" : "bg-red-50";
        pnlText =
          '<div class="text-[9px] text-red-700 font-medium at-mono">\u2212' +
          Math.abs(dayData.net_pnl).toFixed(0) +
          "</div>";
      }
    } else if (isSelected) {
      bgClass = "bg-gray-200";
    }

    var outline = isToday ? " ring-1 ring-gray-400" : "";
    html +=
      '<div class="h-9 flex flex-col items-center justify-center rounded ' +
      bgClass +
      outline +
      ' cursor-pointer hover:bg-gray-100"' +
      " onclick=\"selectCalendarDate('" +
      dateStr +
      "')\">" +
      '<div class="text-xs text-gray-700">' +
      day +
      "</div>" +
      pnlText +
      "</div>";
  }

  html += "</div>";
  el.innerHTML = html;
}

async function refreshCalendar() {
  var data = await fetchAPI("/api/blotter/calendar/NYC?range=" + blotterRange);
  if (!data) return;

  calendarData = {};
  (data.days || []).forEach(function (d) {
    calendarData[d.date] = d;
  });

  var s = data.summary || {};
  document.getElementById("stat-record").textContent =
    (s.wins || 0) + "W\u2013" + (s.losses || 0) + "L";
  var wrEl = document.getElementById("stat-winrate");
  wrEl.textContent = ((s.win_rate || 0) * 100).toFixed(1) + "%";
  wrEl.className =
    "font-bold " + (s.win_rate >= 0.5 ? "text-emerald-600" : "text-red-600");
  document.getElementById("stat-wagered").textContent =
    "$" + (s.total_wagered || 0).toFixed(2);
  var profEl = document.getElementById("stat-profit");
  profEl.textContent = fmtDollars(s.total_profit);
  profEl.className = "font-bold " + pnlColor(s.total_profit);
  var roiEl = document.getElementById("stat-roi");
  roiEl.textContent = ((s.roi || 0) * 100).toFixed(1) + "%";
  roiEl.className = "font-bold " + pnlColor(s.roi);

  renderCalendar();
}

// -----------------------------------------------------------------------
// Settlement Banner
// -----------------------------------------------------------------------

function startCountdownTicker() {
  if (countdownInterval) clearInterval(countdownInterval);
  countdownInterval = setInterval(function () {
    var el = document.getElementById("settle-countdown");
    if (!el || !closeTimeBlotter) return;
    var diff = new Date(closeTimeBlotter).getTime() - Date.now();
    if (diff <= 0) {
      el.textContent = "Settled";
      return;
    }
    var h = Math.floor(diff / 3600000);
    var m = Math.floor((diff % 3600000) / 60000);
    var s = Math.floor((diff % 60000) / 1000);
    el.textContent = h + "h " + m + "m " + s + "s";
  }, 1000);
}

function renderSettleBanner(cd) {
  var el = document.getElementById("settle-info");
  if (!el || !cd) return;
  closeTimeBlotter = cd.close_time;

  var settlementText = "Pending";
  var settleBadge = "pill pill-neutral";
  if (cd.settlement && cd.settlement.source === "NWS_CLI") {
    settlementText = cd.settlement.temp + "\u00b0F (CLI)";
    settleBadge = "pill pill-positive";
  } else if (cd.settlement && cd.settlement.source === "DSM") {
    settlementText = cd.settlement.temp + "\u00b0F (DSM)";
    settleBadge = "pill pill-positive";
  }

  var obsMax =
    cd.running_obs_max != null ? cd.running_obs_max + "\u00b0F" : "--";
  el.innerHTML =
    '<span class="text-gray-900 font-semibold">' +
    cd.event_date +
    "</span>" +
    '<span class="text-gray-400">|</span>' +
    '<span class="text-gray-500">Max: <span class="text-gray-900 font-medium">' +
    obsMax +
    "</span></span>" +
    '<span class="text-gray-400">|</span>' +
    '<span class="text-gray-500">Settlement: <span class="' +
    settleBadge +
    '">' +
    settlementText +
    "</span></span>" +
    '<span class="text-gray-400">|</span>' +
    '<span class="text-gray-500">In: <span id="settle-countdown" class="text-gray-900 font-medium">--</span></span>';
  startCountdownTicker();
}

// -----------------------------------------------------------------------
// Risk Grid
// -----------------------------------------------------------------------

function renderRiskGrid(summary, positions, brackets) {
  document.getElementById("risk-capital").textContent =
    "$" + (summary.capital_locked || 0).toFixed(2);

  var netEv = 0;
  var openPositions = (positions || []).filter(function (p) {
    return p.status === "open";
  });
  var bracketMap = {};
  if (brackets && brackets.brackets) {
    brackets.brackets.forEach(function (b) {
      bracketMap[b.floor + "-" + b.cap] = b;
    });
  }
  openPositions.forEach(function (p) {
    var b = bracketMap[p.bracket_floor + "-" + p.bracket_cap];
    if (!b || p.entry_price == null) return;
    // entry_price is in cents (0-100), convert to fraction for EV math
    var entryFrac = p.entry_price / 100;
    var modelProb = b.model_prob || 0;
    var qty = p.contracts || 1;
    var feeCents = Math.max(
      Math.ceil(0.07 * entryFrac * (1 - entryFrac) * 100),
      1,
    );
    if (p.direction === "YES") {
      // EV = (prob * (100 - fee) - entryCents) * qty / 100
      netEv += ((modelProb * (100 - feeCents) - p.entry_price) * qty) / 100;
    } else {
      var noProb = 1 - modelProb;
      netEv +=
        ((noProb * (100 - feeCents) - (100 - p.entry_price)) * qty) / 100;
    }
  });

  var evEl = document.getElementById("risk-ev");
  evEl.textContent = fmtDollars(netEv);
  evEl.className = "text-lg font-bold at-mono " + pnlColor(netEv);

  var lossEl = document.getElementById("risk-max-loss");
  lossEl.textContent = "-$" + (summary.max_loss || 0).toFixed(2);
  lossEl.className = "text-lg font-bold at-mono text-red-600";

  var pnlEl = document.getElementById("risk-pnl");
  pnlEl.textContent = fmtDollars(summary.day_pnl || 0);
  pnlEl.className = "text-lg font-bold at-mono " + pnlColor(summary.day_pnl);
}

// -----------------------------------------------------------------------
// Live Positions (with sort + time column)
// -----------------------------------------------------------------------

function renderLivePositions(positions, brackets) {
  var tbody = document.getElementById("live-positions-body");
  var emptyEl = document.getElementById("live-positions-empty");
  if (!tbody) return;

  var open = (positions || []).filter(function (p) {
    return p.status === "open";
  });

  if (open.length === 0) {
    tbody.innerHTML = "";
    if (emptyEl) emptyEl.classList.remove("hidden");
    return;
  }
  if (emptyEl) emptyEl.classList.add("hidden");

  var bracketMap = {};
  if (brackets && brackets.brackets) {
    brackets.brackets.forEach(function (b) {
      bracketMap[b.floor + "-" + b.cap] = b;
    });
  }

  // Enrich with computed fields for sorting
  open.forEach(function (p) {
    var b = bracketMap[p.bracket_floor + "-" + p.bracket_cap];
    // mark: brackets API returns 0-1 fractions, convert to cents for display
    var markCents = null;
    if (b) {
      var markFrac = p.direction === "YES" ? b.yes_bid : 1 - (b.yes_ask || 0);
      markCents = Math.round((markFrac || 0) * 100);
    }
    var unrealPnl = p.unrealized_pnl || 0;
    var roc = null;
    // entry_price is in cents, capital at risk = entry cents for YES, (100 - entry) for NO
    if (p.entry_price != null && p.entry_price > 0) {
      var capCents =
        p.direction === "YES" ? p.entry_price : 100 - p.entry_price;
      if (capCents > 0)
        roc = (unrealPnl / ((capCents * (p.contracts || 1)) / 100)) * 100;
    }
    // Model confidence: model_prob for YES, inverted for NO
    var modelConf = null;
    if (p.model_prob != null) {
      modelConf = p.direction === "NO" ? 1 - p.model_prob : p.model_prob;
    }
    // Prior-day detection: compare entry_time ET date to the selected blotter date
    var entryDateET = getETDate(p.entry_time);
    var viewDate = selectedBlotterDate || todayET();
    var isPriorDay = entryDateET != null && entryDateET < viewDate;
    p._sort = {
      bracket: p.bracket_floor || 0,
      roc: roc,
      mark: markCents,
      model_conf: modelConf,
      entry_time: p.entry_time || "",
      _isPriorDay: isPriorDay,
    };
    p._computed = {
      markCents: markCents,
      roc: roc,
      modelConf: modelConf,
      isPriorDay: isPriorDay,
    };
  });

  var sorted = applySortToList(open, "live");

  var html = "";
  sorted.forEach(function (p) {
    var label;
    if (p.bracket_floor == null)
      label = "\u2264" + (p.bracket_cap - 1) + "\u00b0F";
    else if (p.bracket_cap == null)
      label = "\u2265" + (p.bracket_floor + 1) + "\u00b0F";
    else label = p.bracket_floor + "-" + p.bracket_cap + "\u00b0F";

    var markCents = p._computed.markCents;
    var roc = p._computed.roc;
    var modelConf = p._computed.modelConf;
    var isPriorDay = p._computed.isPriorDay;
    var unrealPnl = p.unrealized_pnl || 0;
    var rowClass = "at-border-faint";

    html +=
      '<tr class="' +
      rowClass +
      '">' +
      '<td class="py-2 text-gray-800 font-medium">' +
      label +
      "</td>" +
      '<td class="py-2 text-center ' +
      sideClass(p.direction) +
      '">' +
      p.direction +
      "</td>" +
      '<td class="py-2 text-right text-gray-600">' +
      (modelConf != null ? (modelConf * 100).toFixed(1) + "%" : "--") +
      "</td>" +
      '<td class="py-2 text-right text-gray-700">' +
      (p.contracts || 1) +
      "</td>" +
      '<td class="py-2 text-right text-gray-700">' +
      fmtCents(p.entry_price) +
      "</td>" +
      '<td class="py-2 text-right text-gray-700">' +
      (markCents != null ? markCents + "\u00a2" : "--") +
      "</td>" +
      '<td class="py-2 text-right ' +
      pnlColor(unrealPnl) +
      '">' +
      fmtDollars(unrealPnl) +
      "</td>" +
      '<td class="py-2 text-right ' +
      (roc != null ? pnlColor(roc) : "text-gray-400") +
      '">' +
      (roc != null ? (roc > 0 ? "+" : "") + roc.toFixed(1) + "%" : "--") +
      "</td>" +
      '<td class="py-2 text-right text-gray-500">' +
      fmtTime(p.entry_time) +
      "</td>" +
      "</tr>";
  });
  tbody.innerHTML = html;
}

// -----------------------------------------------------------------------
// Closed Positions (with sort + time columns)
// -----------------------------------------------------------------------

function renderClosedPositions(positions) {
  var tbody = document.getElementById("closed-positions-body");
  var emptyEl = document.getElementById("closed-positions-empty");
  if (!tbody) return;

  var closed = (positions || []).filter(function (p) {
    return p.status !== "open";
  });

  if (closed.length === 0) {
    tbody.innerHTML = "";
    if (emptyEl) emptyEl.classList.remove("hidden");
    return;
  }
  if (emptyEl) emptyEl.classList.add("hidden");

  closed.forEach(function (p) {
    var modelConf = null;
    if (p.model_prob != null) {
      modelConf = p.direction === "NO" ? 1 - p.model_prob : p.model_prob;
    }
    var entryDateET = getETDate(p.entry_time);
    var viewDate = selectedBlotterDate || todayET();
    var isPriorDay = entryDateET != null && entryDateET < viewDate;
    var heldToSettle = p.exit_reason === "settlement";
    p._sort = {
      bracket: p.bracket_floor || 0,
      model_conf: modelConf,
      entry_time: p.entry_time || "",
      exit_time: p.exit_time || "",
      _isPriorDay: isPriorDay,
    };
    p._computed = {
      modelConf: modelConf,
      isPriorDay: isPriorDay,
      heldToSettle: heldToSettle,
    };
  });

  var sorted = applySortToList(closed, "closed");

  var html = "";
  sorted.forEach(function (p) {
    var label;
    if (p.bracket_floor == null)
      label = "\u2264" + (p.bracket_cap - 1) + "\u00b0F";
    else if (p.bracket_cap == null)
      label = "\u2265" + (p.bracket_floor + 1) + "\u00b0F";
    else label = p.bracket_floor + "-" + p.bracket_cap + "\u00b0F";

    var modelConf = p._computed.modelConf;
    var heldToSettle = p._computed.heldToSettle;
    var isPriorDay = p._computed.isPriorDay;
    // Subtle left border for held-to-settlement, dimmed for prior-day
    var rowClass = "at-border-faint";
    if (heldToSettle) rowClass += " border-l-2 border-l-amber-400";

    html +=
      '<tr class="' +
      rowClass +
      '">' +
      '<td class="py-2 text-gray-800 font-medium">' +
      label +
      (heldToSettle
        ? ' <span class="text-[9px] text-amber-500 font-normal" title="Held to settlement">STL</span>'
        : "") +
      "</td>" +
      '<td class="py-2 text-center ' +
      sideClass(p.direction) +
      '">' +
      p.direction +
      "</td>" +
      '<td class="py-2 text-right text-gray-600">' +
      (modelConf != null ? (modelConf * 100).toFixed(1) + "%" : "--") +
      "</td>" +
      '<td class="py-2 text-right text-gray-700">' +
      (p.contracts || 1) +
      "</td>" +
      '<td class="py-2 text-right text-gray-700">' +
      fmtCents(p.entry_price) +
      "</td>" +
      '<td class="py-2 text-right text-gray-700">' +
      fmtCents(p.exit_price) +
      "</td>" +
      '<td class="py-2 text-right ' +
      pnlColor(p.gross_pnl) +
      '">' +
      fmtDollars(p.gross_pnl) +
      "</td>" +
      '<td class="py-2 text-right text-red-600">' +
      (p.fees ? "-$" + Math.abs(p.fees).toFixed(2) : "--") +
      "</td>" +
      '<td class="py-2 text-right ' +
      pnlColor(p.net_pnl) +
      ' font-medium">' +
      fmtDollars(p.net_pnl) +
      "</td>" +
      '<td class="py-2 text-right text-gray-500">' +
      fmtTime(p.entry_time) +
      "</td>" +
      '<td class="py-2 text-right text-gray-500">' +
      fmtTime(p.exit_time) +
      "</td>" +
      "</tr>";
  });
  tbody.innerHTML = html;
}

// -----------------------------------------------------------------------
// Refresh
// -----------------------------------------------------------------------

async function refreshBlotterData(dateStr) {
  var date = dateStr || selectedBlotterDate || getTargetDate("today");
  var data = await fetchAPI("/api/blotter/nyc?date=" + date);
  if (!data) return;

  var labelEl = document.getElementById("positions-date-label");
  if (labelEl) labelEl.textContent = "(" + date + ")";

  // Cache for sorting
  cachedLivePositions = data.positions || [];
  cachedClosedPositions = data.positions || [];
  cachedBrackets = data.brackets;

  renderSettleBanner(data.countdown);
  renderRiskGrid(data.positions_summary, data.positions, data.brackets);
  renderLivePositions(data.positions, data.brackets);
  renderClosedPositions(data.positions);
}

function onDateChange() {
  selectedBlotterDate = window.selectedDate;
  refreshBlotterData();
  refreshCalendar();
}

showSkeleton("live-positions-body", 4);

calendarMonth = new Date();
refreshCalendar();
refreshBlotterData();
setInterval(function () {
  refreshBlotterData();
}, 60000);

// Wire up sortable headers after DOM is ready
setTimeout(function () {
  document.querySelectorAll("th.sortable").forEach(function (th) {
    th.addEventListener("click", function () {
      sortBy(th.dataset.table, th.dataset.key);
    });
  });
}, 100);
