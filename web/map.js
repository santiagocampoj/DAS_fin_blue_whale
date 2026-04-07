var map = L.map('map', {
  center: [38.5, 14.0],   // Mediterranean Sea
  zoom: 5
});

L.tileLayer(
  'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
  {
    attribution: 'Tiles © Esri',
    maxZoom: 18
  }
).addTo(map);
