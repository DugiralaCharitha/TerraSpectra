import FarmMap from '../components/Map/FarmMap'

function Dashboard({ onLogout }) {
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
            <h1>TerraSpectra</h1>
            <p>Geospatial Crop Intelligence Platform</p>
          </div>

          <button
            className="logout-button"
            onClick={onLogout}
          >
            Logout
          </button>

        </div>

        {/* ================= DASHBOARD NAVIGATION ================= */}
        <nav className="dashboard-nav">

          <button onClick={() => scrollToSection('farm-map')}>
            Farm Map
          </button>

          <button onClick={() => scrollToSection('farm-information')}>
            Farm Information
          </button>

          <button onClick={() => scrollToSection('crop-health')}>
            Crop Health
          </button>

          <button onClick={() => scrollToSection('health-trend')}>
            Health Trend
          </button>

          <button onClick={() => scrollToSection('farm-monitoring')}>
            Monitoring
          </button>

          <button onClick={() => scrollToSection('alerts')}>
            Alerts
          </button>

        </nav>

      </header>


      <section className="dashboard-content">

        {/* ================= FARM MAP ================= */}
        <div
          id="farm-map"
          className="map-card"
        >
          <h2>Farm Map</h2>

          <div className="map-container">
            <FarmMap />
          </div>
        </div>


        {/* ================= FARM INFORMATION ================= */}
        <div
          id="farm-information"
          className="farm-info-card"
        >
          <h2>Farm Information</h2>

          <div className="farm-info-grid">

            <div className="info-item">
              <span>Farm Name</span>
              <strong>Wadgaon Farm</strong>
            </div>

            <div className="info-item">
              <span>Location</span>
              <strong>Wadgaon, Maharashtra</strong>
            </div>

            <div className="info-item">
              <span>Farm Area</span>
              <strong>2.5 Acres</strong>
            </div>

            <div className="info-item">
              <span>Crop</span>
              <strong>Wheat</strong>
            </div>

            <div className="info-item">
              <span>Soil Type</span>
              <strong>Black Soil</strong>
            </div>

            <div className="info-item">
              <span>Farm Status</span>
              <strong className="status-healthy">
                Healthy
              </strong>
            </div>

          </div>
        </div>


        {/* ================= CROP HEALTH ================= */}
        <div
          id="crop-health"
          className="crop-health-card"
        >
          <h2>Crop Health</h2>

          <div className="crop-health-grid">

            <div className="health-item">
              <span>Crop Health</span>
              <strong className="status-healthy">
                Healthy
              </strong>
            </div>

            <div className="health-item">
              <span>NDVI</span>
              <strong>0.72</strong>
            </div>

            <div className="health-item">
              <span>Soil Moisture</span>
              <strong>68%</strong>
            </div>

            <div className="health-item">
              <span>Temperature</span>
              <strong>28°C</strong>
            </div>

            <div className="health-item">
              <span>Rainfall</span>
              <strong>42 mm</strong>
            </div>

          </div>
        </div>


        {/* ================= CROP HEALTH TREND ================= */}
        <div
          id="health-trend"
          className="trend-card"
        >

          <div className="trend-header">

            <div>
              <h2>Crop Health Trend</h2>
              <p>NDVI variation over the last 7 days</p>
            </div>

            <div className="trend-current">
              <span>Current NDVI</span>
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
        <div
          id="farm-monitoring"
          className="status-card"
        >

          <h2>Farm Monitoring</h2>

          <div className="status-grid">

            <div className="monitor-item">

              <div className="monitor-icon healthy-dot"></div>

              <div>
                <span>Crop Condition</span>
                <strong>Healthy</strong>
              </div>

            </div>


            <div className="monitor-item">

              <div className="monitor-icon healthy-dot"></div>

              <div>
                <span>Soil Condition</span>
                <strong>Good</strong>
              </div>

            </div>


            <div className="monitor-item">

              <div className="monitor-icon warning-dot"></div>

              <div>
                <span>Weather Risk</span>
                <strong>Low</strong>
              </div>

            </div>


            <div className="monitor-item">

              <div className="monitor-icon healthy-dot"></div>

              <div>
                <span>Farm Status</span>
                <strong>Operational</strong>
              </div>

            </div>

          </div>

        </div>


        {/* ================= ALERTS & RECOMMENDATIONS ================= */}
        <div
          id="alerts"
          className="alerts-card"
        >

          <div className="alerts-header">

            <div>
              <h2>Alerts & Recommendations</h2>
              <p>Important observations for your farm</p>
            </div>

            <span className="alert-count">
              2 Alerts
            </span>

          </div>


          <div className="alerts-list">

            {/* Alert 1 */}
            <div className="alert-item alert-warning">

              <div className="alert-icon">
                ⚠️
              </div>

              <div className="alert-content">

                <h3>Monitor Soil Moisture</h3>

                <p>
                  Soil moisture is currently at 68%.
                  Continue monitoring irrigation levels.
                </p>

                <span className="alert-time">
                  Recommendation
                </span>

              </div>

            </div>


            {/* Alert 2 */}
            <div className="alert-item alert-info">

              <div className="alert-icon">
                💡
              </div>

              <div className="alert-content">

                <h3>Crop Health Looks Good</h3>

                <p>
                  Current NDVI is 0.72, indicating healthy
                  crop vegetation.
                </p>

                <span className="alert-time">
                  Positive observation
                </span>

              </div>

            </div>

          </div>

        </div>

      </section>

    </main>
  )
}

export default Dashboard