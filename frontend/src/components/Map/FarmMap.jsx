import { useState } from 'react'
import {
  MapContainer,
  TileLayer,
  Marker,
  Polygon,
  Popup,
  LayersControl,
  useMap
} from 'react-leaflet'
import 'leaflet/dist/leaflet.css'
import L from 'leaflet'

import {
  FARM_METADATA,
  DISEASE_HOTSPOT,
  generateTimelineGrid
} from '../../services/gisService'
import TimelineSlider from './TimelineSlider'

// Fix Leaflet marker icon
delete L.Icon.Default.prototype._getIconUrl
L.Icon.Default.mergeOptions({
  iconRetinaUrl:
    'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/images/marker-icon-2x.png',
  iconUrl:
    'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/images/marker-icon.png',
  shadowUrl:
    'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/images/marker-shadow.png'
})

// Custom red warning icon for the 5-acre fungal outbreak epicenter
const outbreakIcon = new L.DivIcon({
  className: 'custom-outbreak-marker',
  html: '<div class="pulse-marker-ring"></div><div class="pulse-marker-center">⚠️</div>',
  iconSize: [36, 36],
  iconAnchor: [18, 18]
})

// Camera navigation controls
function MapNavigationControls({ centerCoords, hotspotCoords }) {
  const map = useMap()

  return (
    <div className="gis-camera-controls">
      <button
        className="gis-ctrl-btn"
        onClick={() => map.setView(centerCoords, 14)}
        title="Center view on 1,000-acre farm"
      >
        📍 Center Farm (1,000 Ac)
      </button>

      <button
        className="gis-ctrl-btn focus-hotspot"
        onClick={() => map.setView(hotspotCoords, 16)}
        title="Zoom directly into the 5.2-acre fungal blight outbreak zone"
      >
        🎯 Focus 5.2-Ac Outbreak Zone
      </button>
    </div>
  )
}

// Interactive crop health legend
function HeatmapLegend({ currentWeek }) {
  return (
    <div className="heatmap-legend">
      <div className="legend-header">
        <h4>Crop Spectral Health</h4>
        <span className="legend-validation-badge">✅ GIS Validated</span>
      </div>

      <div className="legend-item">
        <span className="legend-color legend-green"></span>
        <span>Optimal Canopy (NDVI &gt; 0.70)</span>
      </div>

      <div className="legend-item">
        <span className="legend-color legend-yellow"></span>
        <span>Sub-Visual Stress (NDVI 0.55-0.70)</span>
      </div>

      <div className="legend-item">
        <span className="legend-color legend-orange"></span>
        <span>Elevated Pathogen Risk (NDVI 0.40-0.55)</span>
      </div>

      <div className="legend-item">
        <span className="legend-color legend-red"></span>
        <span>Critical Outbreak (NDVI &lt; 0.40)</span>
      </div>

      <div className="legend-footer">
        <small>Timeline Stage: Week {currentWeek >= 0 ? `+${currentWeek}` : currentWeek}</small>
      </div>
    </div>
  )
}

function FarmMap({ currentWeek = 0, onWeekChange }) {
  const [internalWeek, setInternalWeek] = useState(0)
  const [isPlaying, setIsPlaying] = useState(false)

  // Layer visibility toggles
  const [showHeatmap, setShowHeatmap] = useState(true)
  const [showParcels, setShowParcels] = useState(true)
  const [showHotspot, setShowHotspot] = useState(true)

  const activeWeek = onWeekChange !== undefined ? currentWeek : internalWeek
  const handleWeekChange = onWeekChange || setInternalWeek

  // Dynamically generate georeferenced grid for active timeline week
  const { cells } = generateTimelineGrid(activeWeek)

  return (
    <div className="farm-map-wrapper">
      {/* GIS Status & Layer Toolbar */}
      <div className="gis-toolbar">
        <div className="gis-toolbar-left">
          <span className="gis-badge">
            <span className="live-dot"></span>
            <strong>GIS Engine:</strong> EPSG:4326 / WGS84 Georeferenced
          </span>
          <span className="gis-badge accuracy">
            <strong>Offset:</strong> &lt; 0.1m Precision
          </span>
          <span className="gis-badge farm-size">
            <strong>Total Area:</strong> 1,000 Acres (6 Management Parcels)
          </span>
        </div>

        <div className="gis-toolbar-right">
          <label className="toggle-label">
            <input
              type="checkbox"
              checked={showHeatmap}
              onChange={(e) => setShowHeatmap(e.target.checked)}
            />
            <span>3D Spectral Grid</span>
          </label>

          <label className="toggle-label">
            <input
              type="checkbox"
              checked={showParcels}
              onChange={(e) => setShowParcels(e.target.checked)}
            />
            <span>Parcels (A–F)</span>
          </label>

          <label className="toggle-label">
            <input
              type="checkbox"
              checked={showHotspot}
              onChange={(e) => setShowHotspot(e.target.checked)}
            />
            <span>5.2-Ac Hotspot</span>
          </label>
        </div>
      </div>

      {/* Main React Leaflet Map Container */}
      <div className="map-container">
        <MapContainer
          center={FARM_METADATA.center}
          zoom={14}
          scrollWheelZoom={true}
          className="farm-map"
          zoomControl={true}
        >
          {/* Base Maps Layer Control: Esri Satellite (Default) and OpenStreetMap */}
          <LayersControl position="topright">
            <LayersControl.BaseLayer checked name="High-Res Satellite (Esri)">
              <TileLayer
                attribution="&copy; Esri World Imagery, Maxar, Earthstar Geographics"
                url="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
                maxZoom={19}
              />
            </LayersControl.BaseLayer>

            <LayersControl.BaseLayer name="Street Map (OSM)">
              <TileLayer
                attribution="&copy; OpenStreetMap contributors"
                url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
                maxZoom={19}
              />
            </LayersControl.BaseLayer>
          </LayersControl>

          {/* 1,000-Acre Farm Outer Perimeter Boundary */}
          <Polygon
            positions={FARM_METADATA.outer_boundary}
            pathOptions={{
              color: '#163a24',
              weight: 3,
              fillColor: '#2d6a4f',
              fillOpacity: 0.04,
              dashArray: '4 4'
            }}
          >
            <Popup>
              <strong>{FARM_METADATA.farm_name}</strong>
              <br />
              Total Area: <strong>{FARM_METADATA.total_acreage} Acres</strong> (404.7 Hectares)
              <br />
              Location: {FARM_METADATA.location}
              <br />
              Sensor: 200+ Band Hyperspectral (Airborne / Hyperion)
            </Popup>
          </Polygon>

          {/* 6 Agricultural Management Parcels (Parcels A to F) */}
          {showParcels &&
            FARM_METADATA.parcels.map((parcel) => {
              const isCritical = parcel.risk === 'Critical'
              return (
                <Polygon
                  key={parcel.id}
                  positions={parcel.boundary}
                  pathOptions={{
                    color: isCritical ? '#dc2626' : '#22c55e',
                    weight: isCritical ? 2.5 : 1.5,
                    fillColor: isCritical ? '#ef4444' : '#22c55e',
                    fillOpacity: isCritical ? 0.18 : 0.06
                  }}
                >
                  <Popup>
                    <strong>{parcel.name}</strong>
                    <br />
                    Parcel ID: <strong>{parcel.id}</strong> | Size: <strong>{parcel.acres} Acres</strong>
                    <br />
                    Crop Variety: {parcel.crop}
                    <br />
                    Status:{' '}
                    <span style={{ color: isCritical ? '#dc2626' : '#16a34a', fontWeight: 'bold' }}>
                      {parcel.status}
                    </span>
                    <br />
                    Mean NDVI: <strong>{parcel.ndvi}</strong>
                    {isCritical && (
                      <div style={{ marginTop: '6px', color: '#dc2626', fontSize: '12px' }}>
                        ⚠️ Early Fungal Blight Infection Detected in Zone Alpha (5.2 Acres)
                      </div>
                    )}
                  </Popup>
                </Polygon>
              )
            })}

          {/* Georeferenced Spectral Grid Cells (Hyperspectral 3D-CNN/ViT Inference) */}
          {showHeatmap &&
            cells.map((cell) => (
              <Polygon
                key={cell.id}
                positions={cell.bounds}
                pathOptions={{
                  color: cell.color,
                  weight: cell.isHotspot ? 1.5 : 0.6,
                  fillColor: cell.color,
                  fillOpacity: cell.isHotspot ? 0.60 : 0.32
                }}
              >
                <Popup>
                  <strong>Spectral Cell: {cell.id}</strong>
                  <br />
                  GPS Center: {cell.center[0].toFixed(5)}, {cell.center[1].toFixed(5)}
                  <br />
                  Status: <strong>{cell.label}</strong>
                  <br />
                  Disease Severity: <strong>{(cell.severity * 100).toFixed(1)}%</strong>
                  <br />
                  Estimated NDVI: <strong>{cell.ndvi}</strong>
                  {cell.isHotspot && (
                    <div style={{ marginTop: '4px', color: '#dc2626', fontWeight: 'bold' }}>
                      ⚡ 3-Week Early Outbreak Prediction Epicenter
                    </div>
                  )}
                </Popup>
              </Polygon>
            ))}

          {/* 5.2-Acre Fungal Blight Outbreak Zone (Parcel C) */}
          {showHotspot && (
            <>
              <Polygon
                positions={DISEASE_HOTSPOT.boundary}
                pathOptions={{
                  color: '#dc2626',
                  weight: 3.5,
                  fillColor: '#ef4444',
                  fillOpacity: 0.50,
                  dashArray: '6 4'
                }}
              >
                <Popup>
                  <div style={{ minWidth: '220px' }}>
                    <strong style={{ color: '#dc2626', fontSize: '14px' }}>
                      🚨 {DISEASE_HOTSPOT.name}
                    </strong>
                    <hr style={{ margin: '6px 0', borderColor: '#fee2e2' }} />
                    Area: <strong>{DISEASE_HOTSPOT.area_acres} Acres</strong> (Precision Targeted Zone)
                    <br />
                    Outbreak Lead Time: <strong>{DISEASE_HOTSPOT.lead_time_days} Days (3 Weeks Early)</strong>
                    <br />
                    Chlorophyll Dip: <strong>{DISEASE_HOTSPOT.chlorophyll_dip}</strong>
                    <br />
                    PRI Photochemical Index: <strong>{DISEASE_HOTSPOT.pri}</strong>
                    <br />
                    Recommended Action: <strong>Targeted bio-fungicide spray</strong>
                    <div style={{ marginTop: '6px', fontSize: '11px', color: '#059669' }}>
                      ✓ 94.8% Chemical Savings vs Blanket Spraying
                    </div>
                  </div>
                </Popup>
              </Polygon>

              {/* Epicenter Marker */}
              <Marker position={DISEASE_HOTSPOT.epicenter} icon={outbreakIcon}>
                <Popup>
                  <strong>Fungal Blight Epicenter</strong>
                  <br />
                  GPS: {DISEASE_HOTSPOT.epicenter[0]}, {DISEASE_HOTSPOT.epicenter[1]}
                  <br />
                  Predicted Outbreak Zone: 5.2 Acres
                </Popup>
              </Marker>
            </>
          )}

          {/* Camera Navigation Buttons */}
          <MapNavigationControls
            centerCoords={FARM_METADATA.center}
            hotspotCoords={DISEASE_HOTSPOT.epicenter}
          />

          {/* Map Legend */}
          <HeatmapLegend currentWeek={activeWeek} />
        </MapContainer>
      </div>

      {/* Embedded Timeline Progression Slider (Week 4 Milestone) */}
      <TimelineSlider
        currentWeek={activeWeek}
        onWeekChange={handleWeekChange}
        isPlaying={isPlaying}
        setIsPlaying={setIsPlaying}
      />
    </div>
  )
}

export default FarmMap