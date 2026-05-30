// =====================================================================
//  CetaWatch — map.js
//  Dibuja los dos cables (Norte y Sur) de 2021 y pinta todas las
//  detecciones reales del modelo, coloreadas por clase (HF / LF).
// =====================================================================

// Geometria de cada cable (canal -> coordenada). Se rellena al cargar.
const geom = { north: null, south: null };

// Colores por clase del modelo
const CLASS_STYLE = {
  HF: { color: "#22c55e", label: "HF" }, // verde
  LF: { color: "#ef4444", label: "LF" }, // rojo
};

// Estilo de la linea de cada cable
const CABLE_STYLE = {
  north: { color: "#38bdf8" }, // cian
  south: { color: "#2dd4bf" }, // turquesa
};

/**
 * Distancia a lo largo del cable (METROS) -> coordenada [lat, lng] de Leaflet.
 *   canal  = round(d / dc)
 *   indice = canal - channel_min   (canales consecutivos)
 *   coord  = lonlat[indice]        (el JSON guarda [lon, lat])
 */
function distanceToLatLng(d_meters, cable) {
  const g = geom[cable];
  if (!g) return null;
  const channel = Math.round(d_meters / g.dc);
  let idx = channel - g.channel_min;
  idx = Math.max(0, Math.min(idx, g.lonlat.length - 1));
  const [lon, lat] = g.lonlat[idx];
  return [lat, lon];
}

/** Formatea el timestamp ISO a algo legible en el popup. */
function fmtTime(iso) {
  const d = new Date(iso);
  return d.toISOString().replace("T", " ").replace("Z", " UTC");
}

document.addEventListener("DOMContentLoaded", async () => {
  // --- 1. Mapa base: satelite de Esri (sin API key) ---
  const map = L.map("map", { zoomControl: true });

  L.tileLayer(
    "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
    {
      attribution:
        "Imagery &copy; Esri, Maxar, Earthstar Geographics | Cable: OOI RCA (2021)",
      maxZoom: 18,
    }
  ).addTo(map);

  // --- 2. Dibujar los dos cables y ajustar la vista a ambos ---
  const allBounds = [];
  for (const cable of ["north", "south"]) {
    try {
      const resp = await fetch(`data/cable_${cable}.geojson`);
      const data = await resp.json();
      const layer = L.geoJSON(data, {
        style: { color: CABLE_STYLE[cable].color, weight: 3, opacity: 0.85 },
      }).addTo(map);
      allBounds.push(layer.getBounds());
    } catch (err) {
      console.error(`No se pudo cargar el cable ${cable}:`, err);
    }
  }
  if (allBounds.length) {
    let bounds = allBounds[0];
    for (let i = 1; i < allBounds.length; i++) bounds = bounds.extend(allBounds[i]);
    map.fitBounds(bounds, { padding: [50, 50] });
  } else {
    map.setView([45.18, -124.55], 9);
  }

  // --- 3. Cargar geometria (canal -> coordenada) de cada cable ---
  for (const cable of ["north", "south"]) {
    try {
      const resp = await fetch(`data/channel_coords_${cable}.json`);
      geom[cable] = await resp.json();
    } catch (err) {
      console.error(`No se pudo cargar channel_coords_${cable}.json:`, err);
    }
  }

  // --- 4. Cargar y pintar las DETECCIONES reales ---
  let detections = [];
  try {
    const resp = await fetch("data/detections.json");
    detections = await resp.json();
  } catch (err) {
    console.error("No se pudo cargar detections.json:", err);
    return;
  }

  let placed = 0;
  detections.forEach((det) => {
    const latlng = distanceToLatLng(det.d_mid, det.cable);
    if (!latlng) return;

    const style = CLASS_STYLE[det.cls] || { color: "#94a3b8", label: det.cls };

    const marker = L.circleMarker(latlng, {
      radius: 6,
      color: style.color,
      fillColor: style.color,
      fillOpacity: 0.75,
      weight: 2,
    }).addTo(map);

    marker.bindPopup(`
      <div class="popup">
        <div class="popup-title" style="color:${style.color}">
          ${style.label} · ${det.cable.toUpperCase()} cable
        </div>
        <div class="popup-row"><b>Tiempo:</b> ${fmtTime(det.timestamp_utc)}</div>
        <div class="popup-row"><b>Distancia:</b> ${(det.d_mid / 1000).toFixed(2)} km a lo largo del cable</div>
        <div class="popup-row"><b>Duración:</b> ${det.dur} s</div>
        <div class="popup-row popup-id">${det.id}</div>
      </div>
    `);

    placed++;
  });

  console.log(`Detecciones pintadas: ${placed} / ${detections.length}`);
});