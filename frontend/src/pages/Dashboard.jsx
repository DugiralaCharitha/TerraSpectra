import { useState } from 'react'
import FarmMap from '../components/Map/FarmMap'
import SpectralAnalyticsPanel from '../components/Analytics/SpectralAnalyticsPanel'

const API_BASE_URL = 'http://localhost:8000'

function Dashboard({ onLogout }) {
  const [currentWeek, setCurrentWeek] = useState(0)
  const [selectedFile, setSelectedFile] = useState(null)
  const [previewUrl, setPreviewUrl] = useState('')
  const [prediction, setPrediction] = useState(null)

  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [uploadStatus, setUploadStatus] = useState('')

  const scrollToSection = (id) => {
    document.getElementById(id)?.scrollIntoView({
      behavior: 'smooth'
    })
  }

  // =========================================================
  // IMAGE SELECTION
  // =========================================================

  const handleFileChange = (event) => {
    const file = event.target.files?.[0]

    setPrediction(null)
    setError('')
    setUploadStatus('')

    if (!file) {
      setSelectedFile(null)
      setPreviewUrl('')
      return
    }

    const isNpy = file.name.toLowerCase().endsWith('.npy')

    if (!isNpy) {
      setSelectedFile(null)
      setPreviewUrl('')
      setError(
        'Invalid file type. Please select a .npy hyperspectral cube.'
      )
      return
    }

    const maxSize = 50 * 1024 * 1024

    if (file.size > maxSize) {
      setSelectedFile(null)
      setPreviewUrl('')
      setError(
        'File is too large. Please select a file smaller than 50 MB.'
      )
      return
    }

    setSelectedFile(file)
    setPreviewUrl('')
  }

  // =========================================================
  // REMOVE IMAGE
  // =========================================================

  const removeSelectedFile = () => {
    setSelectedFile(null)
    setPreviewUrl('')
    setPrediction(null)
    setError('')
    setUploadStatus('')

    const input = document.getElementById('crop-image')

    if (input) {
      input.value = ''
    }
  }

  // =========================================================
  // PREDICTION
  // =========================================================

  const handlePrediction = async () => {
    if (!selectedFile) {
      setError('Please select a hyperspectral image first.')
      return
    }

    setLoading(true)
    setError('')
    setPrediction(null)
    setUploadStatus('Uploading hyperspectral image...')

    try {
      // -------------------------------------------------------
      // STEP 1: UPLOAD IMAGE
      // -------------------------------------------------------

      const formData = new FormData()

      formData.append('file', selectedFile)

      console.log('Uploading:', selectedFile.name)

      const uploadResponse = await fetch(
        `${API_BASE_URL}/upload`,
        {
          method: 'POST',
          body: formData
        }
      )

      const uploadText = await uploadResponse.text()

      let uploadData = {}

      try {
        uploadData = uploadText
          ? JSON.parse(uploadText)
          : {}
      } catch {
        uploadData = {
          raw_response: uploadText
        }
      }

      console.log(
        'UPLOAD STATUS:',
        uploadResponse.status
      )

      console.log(
        'UPLOAD RESPONSE:',
        uploadData
      )

      if (!uploadResponse.ok) {
        throw new Error(
          uploadData.detail ||
          uploadData.message ||
          uploadData.raw_response ||
          `Image upload failed (${uploadResponse.status}).`
        )
      }

      // -------------------------------------------------------
      // GET IMAGE PATH FROM BACKEND
      // -------------------------------------------------------

      const imagePath =
        uploadData.file_path ||
        uploadData.image_path ||
        uploadData.path ||
        uploadData.filename

      if (!imagePath) {
        throw new Error(
          'Image was uploaded, but the backend did not return an image path.'
        )
      }

      setUploadStatus(
        'Image uploaded successfully. Running AI prediction...'
      )

      console.log(
        'IMAGE PATH:',
        imagePath
      )

      // -------------------------------------------------------
      // STEP 2: CALL PREDICTION API
      // -------------------------------------------------------

      const predictionUrl =
        `${API_BASE_URL}/predict?image_path=${encodeURIComponent(
          imagePath
        )}`

      console.log(
        'PREDICTION URL:',
        predictionUrl
      )

      const predictResponse = await fetch(
        predictionUrl,
        {
          method: 'POST'
        }
      )

      const predictionText =
        await predictResponse.text()

      console.log(
        'PREDICTION STATUS:',
        predictResponse.status
      )

      console.log(
        'PREDICTION RESPONSE:',
        predictionText
      )

      let predictionData = {}

      try {
        predictionData = predictionText
          ? JSON.parse(predictionText)
          : {}
      } catch {
        throw new Error(
          'Prediction service returned an invalid response.'
        )
      }

      if (!predictResponse.ok) {
        throw new Error(
          predictionData.detail ||
          predictionData.message ||
          `Prediction failed (${predictResponse.status}).`
        )
      }

      // -------------------------------------------------------
      // STEP 3: DISPLAY RESULT
      // -------------------------------------------------------

      setPrediction(predictionData)

      setUploadStatus(
        'AI crop analysis completed successfully.'
      )

    } catch (err) {
      console.error(
        'FULL PREDICTION ERROR:',
        err
      )

      setError(
        err?.message ||
        'Unable to connect to the TerraSpectra prediction service.'
      )

      setUploadStatus('')

    } finally {
      setLoading(false)
    }
  }

  // =========================================================
  // CONFIDENCE FORMATTER
  // =========================================================

  const formatConfidence = (confidence) => {
    if (
      confidence === undefined ||
      confidence === null ||
      confidence === ''
    ) {
      return null
    }

    const numericConfidence =
      Number(confidence)

    if (Number.isNaN(numericConfidence)) {
      return confidence
    }

    const percentage =
      numericConfidence <= 1
        ? numericConfidence * 100
        : numericConfidence

    return `${percentage.toFixed(1)}%`
  }

  // =========================================================
  // RETURN
  // =========================================================

  return (
    <main className="dashboard">
      {/* ================= HEADER ================= */}

      {/* =====================================================
          HEADER
      ===================================================== */}
      <header className="dashboard-header">
        <div className="dashboard-header-content">
          <div>
            <div className="title-row">
              <h1>TerraSpectra</h1>
              <span className="header-gis-pill">GIS &amp; 3D Geospatial Engine</span>
            </div>
            <p>Hyperspectral Crop Disease Forecasting &amp; 1,000-Acre Farm Intelligence</p>
            <h1>TerraSpectra</h1>

            <p>
              Geospatial Crop Intelligence Platform
            </p>
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

        {/* ===================================================
            NAVIGATION
        =================================================== */}

        <nav className="dashboard-nav">
          <button onClick={() => scrollToSection('farm-map')}>Farm Map &amp; Timeline</button>
          <button onClick={() => scrollToSection('analytics-panel')}>Spectral Analytics</button>
          <button onClick={() => scrollToSection('farm-information')}>Farm Information</button>
          <button onClick={() => scrollToSection('crop-health')}>Crop Health</button>
          <button onClick={() => scrollToSection('health-trend')}>Health Trend</button>
          <button onClick={() => scrollToSection('farm-monitoring')}>Monitoring</button>
          <button onClick={() => scrollToSection('alerts')}>Alerts</button>

          <button
            onClick={() =>
              scrollToSection('farm-map')
            }
          >
            Farm Map
          </button>

          <button
            onClick={() =>
              scrollToSection('farm-information')
            }
          >
            Farm Information
          </button>

          <button
            onClick={() =>
              scrollToSection('prediction')
            }
          >
            Image Prediction
          </button>

          <button
            onClick={() =>
              scrollToSection('crop-health')
            }
          >
            Crop Health
          </button>

          <button
            onClick={() =>
              scrollToSection('health-trend')
            }
          >
            Health Trend
          </button>

          <button
            onClick={() =>
              scrollToSection('farm-monitoring')
            }
          >
            Monitoring
          </button>

          <button
            onClick={() =>
              scrollToSection('alerts')
            }
          >
            Alerts
          </button>

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

        {/* =====================================================
            FARM MAP
        ===================================================== */}

        <div
          id="farm-map"
          className="map-card"
        >

          <h2>Farm Map</h2>

          <div className="map-container">
            <FarmMap />
          </div>

        </div>

        {/* =====================================================
            FARM INFORMATION
        ===================================================== */}

        <div
          id="farm-information"
          className="farm-info-card"
        >

          <h2>Farm Information</h2>

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

              <strong>
                Wadgaon, Maharashtra
              </strong>

            </div>

            <div className="info-item">

              <span>Farm Area</span>

              <strong>
                2.5 Acres
              </strong>

            </div>

            <div className="info-item">

              <span>Crop</span>

              <strong>
                Wheat
              </strong>

            </div>

            <div className="info-item">

              <span>Soil Type</span>

              <strong>
                Black Soil
              </strong>

            </div>

            <div className="info-item">

              <span>Farm Status</span>

              <strong className="status-healthy">
                Healthy
              </strong>

            </div>

          </div>

        </div>

        {/* =====================================================
            CROP IMAGE PREDICTION
        ===================================================== */}

        <div
          id="prediction"
          className="prediction-card"
        >

          <div className="prediction-header">

            <div>

              <div className="prediction-title-row">

                <h2>
                  Crop Image Prediction
                </h2>

                <span className="prediction-badge">
                  AI Analysis
                </span>

              </div>

              <p>
                Upload a hyperspectral crop scan to
                assess crop health using TerraSpectra
                intelligence.
              </p>

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

          <div className="prediction-body">

            {/* =================================================
                PROFESSIONAL UPLOAD AREA
            ================================================= */}

            {!selectedFile && (

              <div className="prediction-upload-area">

                <div className="prediction-upload-icon">
                  <span>↑</span>
                </div>

                <div className="prediction-upload-content">

                  <h3>
                    Upload Hyperspectral Image
                  </h3>

                  <p>
                    Select a crop scan for AI-powered
                    health analysis
                  </p>

                  <span className="prediction-upload-supported">
                    NPY file • 125 spectral bands • Max 50 MB
                  </span>

                </div>

                <label
                  htmlFor="crop-image"
                  className="prediction-browse-button"
                >
                  Browse Files
                </label>

                <input
                  id="crop-image"
                  type="file"
                  accept=".npy"
                  onChange={handleFileChange}
                  hidden
                />

              </div>

            )}

            {/* =================================================
                SELECTED FILE
            ================================================= */}

            {selectedFile && (

              <div className="prediction-preview-section">

                <div className="prediction-preview">

                  <div
                    style={{
                      padding: '2rem',
                      textAlign: 'center'
                    }}
                  >

                    <div
                      style={{
                        width: '64px',
                        height: '64px',
                        margin: '0 auto 1rem',
                        borderRadius: '16px',
                        background: '#e2e8f0',
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'center',
                        color: '#334155'
                      }}
                    >

                      <div
                        style={{
                          fontSize: '20px',
                          fontWeight: '700',
                          letterSpacing: '0.5px'
                        }}
                      >
                        NPY
                      </div>

                    </div>

                    <strong>
                      Hyperspectral Cube
                    </strong>

                    <p
                      style={{
                        fontSize: '0.85rem',
                        color: '#64748b',
                        marginTop: '0.4rem'
                      }}
                    >
                      125 Spectral Bands
                    </p>

                  </div>

                </div>

                <div className="prediction-file">

                  <div className="prediction-file-icon">
                    NPY
                  </div>

                  <div className="prediction-file-info">

                    <strong>
                      {selectedFile.name}
                    </strong>

                    <span>
                      {(selectedFile.size / 1024).toFixed(1)} KB
                    </span>

                  </div>

                  <button
                    type="button"
                    className="prediction-remove"
                    onClick={removeSelectedFile}
                    disabled={loading}
                  >
                    Remove
                  </button>

                </div>

              </div>

            )}

            {/* =================================================
                ANALYZE BUTTON
            ================================================= */}

            <button
              type="button"
              onClick={handlePrediction}
              disabled={
                !selectedFile ||
                loading
              }
              className="prediction-button"
            >

              {loading
                ? 'Analyzing crop image...'
                : 'Analyze Crop Image'}

            </button>

            {/* =================================================
                STATUS
            ================================================= */}

            {uploadStatus && (

              <div className="prediction-status">

                <span className="status-dot"></span>

                <span>
                  {uploadStatus}
                </span>

              </div>

            )}

            {/* =================================================
                ERROR
            ================================================= */}

            {error && (

              <div className="prediction-error">

                <strong>
                  Analysis could not be completed
                </strong>

                <p>
                  {error}
                </p>

              </div>

            )}

            {/* =================================================
                AI RESULT
            ================================================= */}

            {prediction && (

              <div className="prediction-result">

                <div className="prediction-result-header">

                  <div>

                    <span className="result-label">
                      ANALYSIS RESULT
                    </span>

                    <h3>
                      Crop Health Assessment
                    </h3>

                  </div>

                  <div className="result-success">
                    ✓ Complete
                  </div>

                </div>

                <div className="prediction-result-grid">

                  {/* STATUS */}

                  {prediction.status && (

                    <div className="prediction-metric">

                      <span>
                        Status
                      </span>

                      <strong>
                        {prediction.status}
                      </strong>

                    </div>

                  )}

                  {/* PREDICTION */}

                  {prediction.prediction && (

                    <div className="prediction-metric">

                      <span>
                        Prediction
                      </span>

                      <strong>
                        {prediction.prediction}
                      </strong>

                    </div>

                  )}

                  {/* LABEL */}

                  {prediction.label && (

                    <div className="prediction-metric">

                      <span>
                        Classification
                      </span>

                      <strong>
                        {prediction.label}
                      </strong>

                    </div>

                  )}

                  {/* CLASS NAME */}

                  {prediction.class_name && (

                    <div className="prediction-metric">

                      <span>
                        Crop Condition
                      </span>

                      <strong>
                        {prediction.class_name}
                      </strong>

                    </div>

                  )}

                  {/* CONFIDENCE */}

                  {prediction.confidence !== undefined && (

                    <div className="prediction-metric">

                      <span>
                        Confidence
                      </span>

                      <strong>
                        {formatConfidence(
                          prediction.confidence
                        )}
                      </strong>

                    </div>

                  )}

                </div>

                {/* RECOMMENDATION */}

                {prediction.message && (

                  <div className="prediction-message">

                    <span>
                      Recommendation
                    </span>

                    <p>
                      {prediction.message}
                    </p>

                  </div>

                )}

              </div>

            )}

          </div>

        </div>

        {/* ================= CROP HEALTH ================= */}
        <div id="crop-health" className="crop-health-card">
          <h2>Crop Health &amp; Biophysical Parameters</h2>
        {/* =====================================================
            CROP HEALTH
        ===================================================== */}

        <div
          id="crop-health"
          className="crop-health-card"
        >

          <h2>Crop Health</h2>

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

              <span>
                Crop Health
              </span>

              <strong className="status-healthy">
                Healthy
              </strong>

            </div>

            <div className="health-item">

              <span>
                NDVI
              </span>

              <strong>
                0.72
              </strong>

            </div>

            <div className="health-item">

              <span>
                Soil Moisture
              </span>

              <strong>
                68%
              </strong>

            </div>

            <div className="health-item">

              <span>
                Temperature
              </span>

              <strong>
                28°C
              </strong>

            </div>

            <div className="health-item">

              <span>
                Rainfall
              </span>

              <strong>
                42 mm
              </strong>


            </div>
          </div>

        </div>

        {/* ================= CROP HEALTH TREND ================= */}
        <div id="health-trend" className="trend-card">
        {/* =====================================================
            HEALTH TREND
        ===================================================== */}

        <div
          id="health-trend"
          className="trend-card"
        >

          <div className="trend-header">
            <div>
              <h2>Crop Health Trend</h2>
              <p>Mean NDVI variation across the 1,000-acre farm over the last 7 days</p>
            </div>

            <div className="trend-current">
              <span>Current Mean NDVI</span>
              <strong>0.72</strong>

              <h2>
                Crop Health Trend
              </h2>

              <p>
                NDVI variation over the last 7 days
              </p>

            </div>

            <div className="trend-current">

              <span>
                Current NDVI
              </span>

              <strong>
                0.72
              </strong>

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
        {/* =====================================================
            FARM MONITORING
        ===================================================== */}

        <div
          id="farm-monitoring"
          className="status-card"
        >

          <h2>
            Farm Monitoring
          </h2>

          <div className="status-grid">
            <div className="monitor-item">
              <div className="monitor-icon warning-dot"></div>
              <div>
                <span>Hyperspectral Disease Risk</span>
                <strong className="status-warning">High in Parcel C (5.2 Ac)</strong>

                <span>
                  Crop Condition
                </span>

                <strong>
                  Healthy
                </strong>

              </div>
            </div>

            <div className="monitor-item">
              <div className="monitor-icon healthy-dot"></div>
              <div>
                <span>Soil Moisture Status</span>
                <strong>Optimal (68% Capacity)</strong>

                <span>
                  Soil Condition
                </span>

                <strong>
                  Good
                </strong>

              </div>
            </div>

            <div className="monitor-item">
              <div className="monitor-icon healthy-dot"></div>
              <div>
                <span>Weather &amp; Humidity Risk</span>
                <strong>Moderate (Warm / Humid)</strong>

                <span>
                  Weather Risk
                </span>

                <strong>
                  Low
                </strong>

              </div>
            </div>

            <div className="monitor-item">
              <div className="monitor-icon healthy-dot"></div>
              <div>
                <span>GIS Pipeline Status</span>
                <strong>Operational (Port 8080)</strong>

                <span>
                  Farm Status
                </span>

                <strong>
                  Operational
                </strong>

              </div>
            </div>
          </div>
        </div>

        {/* ================= ALERTS & RECOMMENDATIONS ================= */}
        <div id="alerts" className="alerts-card">
        {/* =====================================================
            ALERTS
        ===================================================== */}

        <div
          id="alerts"
          className="alerts-card"
        >

          <div className="alerts-header">
            <div>
              <h2>Early Warning Alerts &amp; Precision Prescriptions</h2>
              <p>Actionable AI insights generated by 3D-CNN &amp; Vision Transformer</p>

              <h2>
                Alerts & Recommendations
              </h2>

              <p>
                Important observations for your farm
              </p>

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


            <div className="alert-item alert-warning">

              <div className="alert-icon">
                ⚠️
              </div>

              <div className="alert-content">

                <h3>
                  Monitor Soil Moisture
                </h3>

                <p>
                  Soil moisture is currently at
                  68%. Continue monitoring
                  irrigation levels.

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

            <div className="alert-item alert-info">

              <div className="alert-icon">
                💡

              </div>
            </div>

            {/* Alert 3 */}
            <div className="alert-item alert-info">
              <div className="alert-icon">✅</div>
              <div className="alert-content">
                <h3>Parcels A, B, D, E, F Operating at Peak Vegetative Health</h3>
                <p>
                  976.4 acres of wheat crops display robust chlorophyll synthesis and optimal photochemical reflectance (PRI &gt; +0.035).


                <h3>
                  Crop Health Looks Good
                </h3>

                <p>
                  Current NDVI is 0.72,
                  indicating healthy crop
                  vegetation.

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