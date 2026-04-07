// ── CONFIGURATION ────────────────────────────────

var COLORS = [
  '#0d1520',   // 0 detections (almost black)
  '#0d2030',   // very few
  '#0a4d3a',   // few
  '#0d7a56',   // moderate
  '#00b874',   // many
  '#00d4aa'    // most (bright accent)
];

var MONTHS = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'];


// ── HELPERS ──────────────────────────────────────

// Returns the right colour for a cell given its value and the dataset maximum
function cellColor(val, max) {
  if (val === 0) return COLORS[0];
  var idx = Math.ceil(val / max * 5);
  return COLORS[Math.min(idx, 5)];
}

// Generates 364 days of fake detection data
// Replace this function later with real DAS data
function generateData() {
  var today = new Date();
  today.setHours(0, 0, 0, 0);

  var days = [];

  for (var i = 363; i >= 0; i--) {
    var d = new Date(today);
    d.setDate(d.getDate() - i);
    var mo = d.getMonth();

    // Blue/fin whales more active spring–autumn in the Mediterranean
    var seasonal = (mo >= 3 && mo <= 9) ? 1.8 : 0.6;

    var val = Math.random() < 0.15
      ? Math.floor(Math.random() * 18 * seasonal)  // occasional busy day
      : Math.floor(Math.random() * 6  * seasonal);

    days.push({ date: new Date(d), val: val });
  }

  return days;
}


// ── RENDERING ────────────────────────────────────

// Builds the 52×7 grid of coloured cells
function buildWeeks(days) {
  var max = Math.max.apply(null, days.map(function(d) { return d.val; }));
  var startDOW = days[0].date.getDay();  // which weekday does week 1 start on?
  var weeksEl = document.getElementById('hm-weeks');
  weeksEl.innerHTML = '';

  for (var w = 0; w < 52; w++) {
    var weekEl = document.createElement('div');
    weekEl.className = 'hm-week';

    for (var d = 0; d < 7; d++) {
      var dayIdx = w * 7 + d - startDOW;
      var cell = document.createElement('div');
      cell.className = 'hm-cell';

      if (dayIdx < 0 || dayIdx >= days.length) {
        // Outside the 52-week window — leave transparent
        cell.style.background = 'transparent';
        cell.style.cursor = 'default';
      } else {
        var day = days[dayIdx];
        cell.style.background = cellColor(day.val, max);

        // Attach tooltip events — wrapped in IIFE to capture the correct day
        ;(function(day) {
          cell.addEventListener('mouseenter', function(e) { showTip(e, day.date, day.val); });
          cell.addEventListener('mouseleave', hideTip);
        })(day);
      }

      weekEl.appendChild(cell);
    }

    weeksEl.appendChild(weekEl);
  }
}

// Builds the month name labels above the grid
function buildMonthLabels(startDate) {
  var row = document.getElementById('hm-months');
  row.innerHTML = '';

  var d = new Date(startDate);
  var prev = -1;
  var segs = [];

  for (var w = 0; w < 52; w++) {
    var mo = d.getMonth();
    if (mo !== prev) {
      segs.push({ label: MONTHS[mo], weeks: 1 });
      prev = mo;
    } else {
      segs[segs.length - 1].weeks++;
    }
    d.setDate(d.getDate() + 7);
  }

  segs.forEach(function(s) {
    var el = document.createElement('div');
    el.className = 'hm-month';
    el.style.flex = s.weeks;
    el.textContent = s.label;
    row.appendChild(el);
  });
}


// ── TOOLTIP ──────────────────────────────────────

var tip = document.getElementById('tooltip');

function showTip(e, date, val) {
  var dateStr = date.toLocaleDateString('en-GB', { day: 'numeric', month: 'short', year: 'numeric' });
  tip.innerHTML = '<strong>' + val + ' detection' + (val !== 1 ? 's' : '') + '</strong> · ' + dateStr;
  tip.style.display = 'block';
  positionTip(e);
}

function hideTip() {
  tip.style.display = 'none';
}

function positionTip(e) {
  tip.style.left = (e.clientX + 12) + 'px';
  tip.style.top  = (e.clientY - 32) + 'px';
}

document.addEventListener('mousemove', function(e) {
  if (tip.style.display !== 'none') positionTip(e);
});


// ── PUBLIC API ───────────────────────────────────

// This is the only function main.js needs to call
function buildCalendar() {
  var days = generateData();
  buildWeeks(days);
  buildMonthLabels(days[0].date);
}
