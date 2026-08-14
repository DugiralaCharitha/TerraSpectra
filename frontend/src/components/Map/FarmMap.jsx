import { useState } from 'react'

import {
  MapContainer,
  TileLayer,
  Marker,
  Polygon,
  Popup,
  LayersControl,
  Circle,
  useMap
} from 'react-leaflet'

import 'leaflet/dist/leaflet.css'
import L from 'leaflet'

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

// Farm location
const farmPosition = [20.7505, 76.6061]

// Farm boundary
const farmBoundary = [
  [20.7512, 76.6055],
  [20.7515, 76.6070],
  [20.7498, 76.6074],
  [20.7495, 76.6058]
]

// Mock crop-health data
const heatmapData = [
  { position: [20.7510, 76.6058], severity: 0.2 },
  { position: [20.7510, 76.6063], severity: 0.4 },
  { position: [20.7510, 76.6068], severity: 0.7 },
  { position: [20.7510, 76.6072], severity: 0.9 },

  { position: [20.7505, 76.6058], severity: 0.3 },
  { position: [20.7505, 76.6063], severity: 0.5 },
  { position: [20.7505, 76.6068], severity: 0.8 },
  { position: [20.7505, 76.6072], severity: 0.6 },

  { position: [20.7500, 76.6058], severity: 0.1 },
  { position: [20.7500, 76.6063], severity: 0.3 },
  { position: [20.7500, 76.6068], severity: 0.6 },
  { position: [20.7500, 76.6072], severity: 0.8 },

  { position: [20.7497, 76.6060], severity: 0.2 },
  { position: [20.7497, 76.6065], severity: 0.4 },
  { position: [20.7497, 76.6070], severity: 0.7 }
]

// Convert severity to color
function getHeatmapColor(severity) {
  if (severity < 0.3) {
    return 'green'
  }

  if (severity < 0.5) {
    return 'yellow'
  }

  if (severity < 0.7) {
    return 'orange'
  }

  return 'red'
}

// Center Farm button
function CenterFarmButton() {
  const map = useMap()

  const centerFarm = () => {
    map.setView(farmPosition, 15)
  }

  return (
    <div className="center-farm-control">
      <button onClick={centerFarm}>
        📍 Center Farm
      </button>
    </div>
  )
}

// Heatmap toggle
function HeatmapToggle({ showHeatmap, setShowHeatmap }) {
  return (
    <div className="heatmap-toggle">
      <button onClick={() => setShowHeatmap(!showHeatmap)}>
        {showHeatmap ? 'Hide Heatmap' : 'Show Heatmap'}
      </button>
    </div>
  )
}

// Crop health legend
function HeatmapLegend() {
  return (
    <div className="heatmap-legend">
      <h4>Crop Health</h4>

      <div className="legend-item">
        <span className="legend-color legend-green"></span>
        <span>Low Stress</span>
      </div>

      <div className="legend-item">
        <span className="legend-color legend-yellow"></span>
        <span>Moderate Stress</span>
      </div>

      <div className="legend-item">
        <span className="legend-color legend-orange"></span>
        <span>Elevated Stress</span>
      </div>

      <div className="legend-item">
        <span className="legend-color legend-red"></span>
        <span>High Stress</span>
      </div>
    </div>
  )
}

// Heatmap circles
function CropHealthHeatmap() {
  return (
    <>
      {heatmapData.map((cell, index) => {
        const color = getHeatmapColor(cell.severity)

        return (
          <Circle
            key={index}
            center={cell.position}
            radius={45}
            pathOptions={{
              color: color,
              fillColor: color,
              fillOpacity: 0.45,
              weight: 1
            }}
          >
            <Popup>
              <strong>Crop Health Zone</strong>
              <br />
              Severity: {Math.round(cell.severity * 100)}%
            </Popup>
          </Circle>
        )
      })}
    </>
  )
}

function FarmMap() {
  const [showHeatmap, setShowHeatmap] = useState(true)

  return (
    <MapContainer
      center={farmPosition}
      zoom={15}
      scrollWheelZoom={true}
      className="farm-map"
      zoomControl={true}
    >

      {/* Base Maps */}
      <LayersControl position="topright">

        <LayersControl.BaseLayer
          checked
          name="Street Map"
        >
          <TileLayer
            attribution="&copy; OpenStreetMap contributors"
            url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
          />
        </LayersControl.BaseLayer>

        <LayersControl.BaseLayer
          name="Satellite"
        >
          <TileLayer
            attribution="&copy; Esri"
            url="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
          />
        </LayersControl.BaseLayer>

      </LayersControl>


      {/* Farm Marker */}
      <Marker position={farmPosition}>
        <Popup>
          <strong>Wadgaon Farm</strong>
          <br />
          Wheat Farm
        </Popup>
      </Marker>


      {/* Farm Boundary */}
      <Polygon
        positions={farmBoundary}
        pathOptions={{
          color: 'green',
          fillColor: 'green',
          fillOpacity: 0.25
        }}
      >
        <Popup>
          <strong>Wadgaon Farm</strong>
          <br />
          Area: 2.5 Acres
        </Popup>
      </Polygon>


      {/* Crop Health Heatmap */}
      {showHeatmap && <CropHealthHeatmap />}


      {/* Custom Controls */}
      <CenterFarmButton />

      <HeatmapToggle
        showHeatmap={showHeatmap}
        setShowHeatmap={setShowHeatmap}
      />

      <HeatmapLegend />

    </MapContainer>
  )
}

export default FarmMap