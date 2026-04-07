// Wait for the full HTML to be ready before running anything
document.addEventListener('DOMContentLoaded', function () {

  // map.js already ran and set up the Leaflet map — nothing to call here.

  // Build the calendar heatmap (defined in calendar.js)
  buildCalendar();

});
