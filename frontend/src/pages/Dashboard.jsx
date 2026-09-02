import { useState } from 'react'
import FarmMap from '../components/Map/FarmMap'
import SpectralAnalyticsPanel from '../components/Analytics/SpectralAnalyticsPanel'

function Dashboard({ onLogout }) {
  const [currentWeek, setCurrentWeek] = useState(0)

  const scrollToSection = (id) => {
    document.getElementById(id)?.scrollIntoView({
      behavior: 'smooth'
    })
  }

  return (
    <main className="dashboard">
      {/* ================= HEADER ================= */}
      <header className="dashboard-header">
        <div className="dashboard-header-content">
          <div>
            <div className="title-row">
              <h1>TerraSpectra</h1>
              <span className="header-gis-pill">GIS &amp; 3D Geospatial Engine</span>
            </div>
            <p>Hyperspectral Crop Disease Forecasting &amp; 1,000-Acre Farm Intelligence</p>
          </div>

          <div className="header-actions">
            <span className="validation-status-tag">
              <span className="green-circle"></span> GIS Validated (&lt; 0.1m Precision)
            </span>
            <button className="logout-button" onClick={onLogout}>
              Logout
            </button>
          </div>
        </div>

        {/* ================= DASHBOARD NAVIGATION ================= */}
        <nav className="dashboard-nav">
          <button onClick={() => scrollToSection('farm-map')}>Farm Map &amp; Timeline</button>
          <button onClick={() => scrollToSection('analytics-panel')}>Spectral Analytics</button>
          <button onClick={() => scrollToSection('farm-information')}>Farm Information</button>
          <button onClick={() => scrollToSection('crop-health')}>Crop Health</button>
          <button onClick={() => scrollToSection('health-trend')}>Health Trend</button>
          <button onClick={() => scrollToSection('farm-monitoring')}>Monitoring</button>
          <button onClick={() => scrollToSection('alerts')}>Alerts</button>
        </nav>
      </header>

      <section className="dashboard-content">
        {/* ================= FARM MAP & GIS VISUALIZATION ================= */}
        <div id="farm-map" className="map-card">
          <div className="card-header-flex">
            <div>
              <h2>Interactive Farm GIS Map</h2>
              <p className="card-subtitle">
                React-Leaflet Satellite Topography with 200+ Band Hyperspectral Grid Overlay
              </p>
            </div>
            <div className="card-header-badge">
              <span>Monitored Extent: <strong>1,000 Acres</strong></span>
            </div>
          </div>

          <FarmMap currentWeek={currentWeek} onWeekChange={setCurrentWeek} />
        </div>

        {/* ================= SPECTRAL ANALYTICS PANEL (WEEK 3 MILESTONE) ================= */}
        <SpectralAnalyticsPanel currentWeek={currentWeek} />

        {/* ================= FARM INFORMATION ================= */}
        <div id="farm-information" className="farm-info-card">
          <h2>Farm Information &amp; Geospatial Metadata</h2>

          <div className="farm-info-grid">
            <div className="info-item">
              <span>Farm Name</span>
              <strong>Wadgaon Research Site</strong>
            </div>

            <div className="info-item">
              <span>Location</span>
              <strong>Wadgaon, Maharashtra (20.75°N, 76.61°E)</strong>
            </div>

            <div className="info-item">
              <span>Total Farm Area</span>
              <strong>1,000.14 Acres (404.7 Ha)</strong>
            </div>

            <div className="info-item">
              <span>Management Parcels</span>
              <strong>6 Parcels (A through F)</strong>
            </div>

            <div className="info-item">
              <span>Primary Crop</span>
              <strong>Wheat (Sharbati &amp; Lokwan)</strong>
            </div>

            <div className="info-item">
              <span>Soil Classification</span>
              <strong>Deep Vertisol (Black Cotton Soil)</strong>
            </div>

            <div className="info-item">
              <span>Spatial Reference</span>
              <strong>EPSG:4326 (WGS84 / CRS84)</strong>
            </div>

            <div className="info-item">
              <span>Current Status</span>
              <strong className="status-critical">Early Warning Active (Parcel C)</strong>
            </div>
          </div>
        </div>

        {/* ================= CROP HEALTH ================= */}
        <div id="crop-health" className="crop-health-card">
          <h2>Crop Health &amp; Biophysical Parameters</h2>

          <div className="crop-health-grid">
            <div className="health-item">
              <span>Canopy Health</span>
              <strong className="status-warning">Early Anomaly (Parcel C)</strong>
            </div>

            <div className="health-item">
              <span>Farm Mean NDVI</span>
              <strong>0.72</strong>
            </div>

            <div className="health-item">
              <span>Hotspot NDVI</span>
              <strong className="status-critical">0.46</strong>
            </div>

            <div className="health-item">
              <span>Red Edge Chlorophyll Dip</span>
              <strong className="status-critical">-28.4% (705nm)</strong>
            </div>

            <div className="health-item">
              <span>Soil Moisture</span>
              <strong>68%</strong>
            </div>

            <div className="health-item">
              <span>Canopy Temperature</span>
              <strong>28.4°C</strong>
            </div>

            <div className="health-item">
              <span>Predicted Lead Time</span>
              <strong className="status-lead">21 Days Early</strong>
            </div>
          </div>
        </div>

        {/* ================= CROP HEALTH TREND ================= */}
        <div id="health-trend" className="trend-card">
          <div className="trend-header">
            <div>
              <h2>Crop Health Trend</h2>
              <p>Mean NDVI variation across the 1,000-acre farm over the last 7 days</p>
            </div>

            <div className="trend-current">
              <span>Current Mean NDVI</span>
              <strong>0.72</strong>
            </div>
          </div>

          <div className="trend-chart">
            <div className="chart-y-axis">
              <span>1.0</span>
              <span>0.8</span>
              <span>0.6</span>
              <span>0.4</span>
              <span>0.2</span>
              <span>0</span>
            </div>

            <div className="chart-area">
              <div className="chart-grid-line line-1"></div>
              <div className="chart-grid-line line-2"></div>
              <div className="chart-grid-line line-3"></div>
              <div className="chart-grid-line line-4"></div>
              <div className="chart-grid-line line-5"></div>

              <div className="trend-line">
                <span className="chart-point point-1"></span>
                <span className="chart-point point-2"></span>
                <span className="chart-point point-3"></span>
                <span className="chart-point point-4"></span>
                <span className="chart-point point-5"></span>
                <span className="chart-point point-6"></span>
                <span className="chart-point point-7"></span>
              </div>

              <div className="chart-labels">
                <span>Mon</span>
                <span>Tue</span>
                <span>Wed</span>
                <span>Thu</span>
                <span>Fri</span>
                <span>Sat</span>
                <span>Sun</span>
              </div>
            </div>
          </div>
        </div>

        {/* ================= FARM MONITORING ================= */}
        <div id="farm-monitoring" className="status-card">
          <h2>Farm Sub-System Monitoring</h2>

          <div className="status-grid">
            <div className="monitor-item">
              <div className="monitor-icon warning-dot"></div>
              <div>
                <span>Hyperspectral Disease Risk</span>
                <strong className="status-warning">High in Parcel C (5.2 Ac)</strong>
              </div>
            </div>

            <div className="monitor-item">
              <div className="monitor-icon healthy-dot"></div>
              <div>
                <span>Soil Moisture Status</span>
                <strong>Optimal (68% Capacity)</strong>
              </div>
            </div>

            <div className="monitor-item">
              <div className="monitor-icon healthy-dot"></div>
              <div>
                <span>Weather &amp; Humidity Risk</span>
                <strong>Moderate (Warm / Humid)</strong>
              </div>
            </div>

            <div className="monitor-item">
              <div className="monitor-icon healthy-dot"></div>
              <div>
                <span>GIS Pipeline Status</span>
                <strong>Operational (Port 8080)</strong>
              </div>
            </div>
          </div>
        </div>

        {/* ================= ALERTS & RECOMMENDATIONS ================= */}
        <div id="alerts" className="alerts-card">
          <div className="alerts-header">
            <div>
              <h2>Early Warning Alerts &amp; Precision Prescriptions</h2>
              <p>Actionable AI insights generated by 3D-CNN &amp; Vision Transformer</p>
            </div>

            <span className="alert-count">3 Active Notices</span>
          </div>

          <div className="alerts-list">
            {/* Alert 1: Urgent Early Fungal Blight Notice */}
            <div className="alert-item alert-critical">
              <div className="alert-icon">🚨</div>
              <div className="alert-content">
                <div className="alert-title-row">
                  <h3>Targeted Preventative Action: Fungal Blight Outbreak</h3>
                  <span className="alert-urgent-tag">URGENT ACTION (21-Day Window)</span>
                </div>
                <p>
                  The 3D-CNN &amp; ViT model detected a subtle chlorophyll reflection dip (-28.4% at 705nm) across a <strong>5.2-acre zone in Parcel C</strong>. Leaves are currently green and show zero macroscopic symptoms. Apply targeted bio-fungicide within 48–72 hours to neutralize fungal hyphae before foliar necrosis occurs.
                </p>
                <div className="alert-meta-row">
                  <span className="alert-meta-tag">Affected Area: 5.2 / 1,000 Acres</span>
                  <span className="alert-meta-tag">Estimated Savings: $38,500 (94.8% reduction vs full farm spray)</span>
                  <span className="alert-time">Forecast Horizon: 3 Weeks Early</span>
                </div>
              </div>
            </div>

            {/* Alert 2 */}
            <div className="alert-item alert-warning">
              <div className="alert-icon">⚠️</div>
              <div className="alert-content">
                <h3>Monitor Soil Moisture in Northeast Parcels</h3>
                <p>
                  High relative humidity combined with 68% soil moisture elevates sporulation likelihood. Maintain current drip irrigation schedule without overwatering.
                </p>
                <span className="alert-time">Irrigation Advisory</span>
              </div>
            </div>

            {/* Alert 3 */}
            <div className="alert-item alert-info">
              <div className="alert-icon">✅</div>
              <div className="alert-content">
                <h3>Parcels A, B, D, E, F Operating at Peak Vegetative Health</h3>
                <p>
                  976.4 acres of wheat crops display robust chlorophyll synthesis and optimal photochemical reflectance (PRI &gt; +0.035).
                </p>
                <span className="alert-time">Vegetative Health Status</span>
              </div>
            </div>
          </div>
        </div>
      </section>
    </main>
  )
}

export default Dashboard