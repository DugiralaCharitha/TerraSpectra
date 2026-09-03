import { useState, useRef } from 'react'
import '../../styles/CropImagePrediction.css'

const API_BASE_URL = 'http://localhost:8000'

function CropImagePrediction({ farmId = null, farmName = null }) {
  const [selectedFile, setSelectedFile] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [uploadPhase, setUploadPhase] = useState('')
  const [prediction, setPrediction] = useState(null)
  const [isDragging, setIsDragging] = useState(false)
  
  const fileInputRef = useRef(null)

  // ---------------------------------------------------------
  // FILE SELECTION & VALIDATION
  // ---------------------------------------------------------
  const processFile = (file) => {
    setError(null)
    setPrediction(null)
    setUploadPhase('')

    if (!file) {
      setSelectedFile(null)
      return
    }

    const isNpy = file.name.toLowerCase().endsWith('.npy')
    if (!isNpy) {
      setSelectedFile(null)
      setError({
        title: 'Unsupported File Format',
        message: 'Please select a valid .npy hyperspectral cube file.'
      })
      return
    }

    const maxSize = 50 * 1024 * 1024 // 50 MB
    if (file.size > maxSize) {
      setSelectedFile(null)
      setError({
        title: 'File Size Exceeded',
        message: 'Selected hyperspectral cube exceeds the 50 MB limit.'
      })
      return
    }

    setSelectedFile(file)
  }

  const handleFileChange = (e) => {
    const file = e.target.files?.[0]
    processFile(file)
  }

  const handleDragOver = (e) => {
    e.preventDefault()
    e.stopPropagation()
    setIsDragging(true)
  }

  const handleDragLeave = (e) => {
    e.preventDefault()
    e.stopPropagation()
    setIsDragging(false)
  }

  const handleDrop = (e) => {
    e.preventDefault()
    e.stopPropagation()
    setIsDragging(false)
    const file = e.dataTransfer.files?.[0]
    processFile(file)
  }

  const removeSelectedFile = () => {
    setSelectedFile(null)
    setPrediction(null)
    setError(null)
    setUploadPhase('')
    if (fileInputRef.current) {
      fileInputRef.current.value = ''
    }
  }

  const triggerBrowse = () => {
    fileInputRef.current?.click()
  }

  // ---------------------------------------------------------
  // FORMATTERS
  // ---------------------------------------------------------
  const formatFileSize = (bytes) => {
    if (!bytes || bytes === 0) return '0 B'
    const k = 1024
    const sizes = ['B', 'KB', 'MB', 'GB']
    const i = Math.floor(Math.log(bytes) / Math.log(k))
    return `${(bytes / Math.pow(k, i)).toFixed(1)} ${sizes[i]}`
  }

  const formatConfidence = (val) => {
    if (val === undefined || val === null || val === '') return 'N/A'
    const num = Number(val)
    if (Number.isNaN(num)) return String(val)
    const pct = num <= 1 ? num * 100 : num
    return `${pct.toFixed(2)}%`
  }

  // ---------------------------------------------------------
  // API PREDICTION PIPELINE
  // ---------------------------------------------------------
  const handlePrediction = async () => {
    if (!selectedFile) {
      setError({
        title: 'No File Selected',
        message: 'Please select a hyperspectral image scan to begin analysis.'
      })
      return
    }

    setLoading(true)
    setError(null)
    setPrediction(null)
    setUploadPhase('Uploading hyperspectral image...')

    try {
      // Step 1: Upload file to /upload
      const formData = new FormData()
      formData.append('file', selectedFile)

      const uploadRes = await fetch(`${API_BASE_URL}/upload`, {
        method: 'POST',
        body: formData
      })

      const uploadText = await uploadRes.text()
      let uploadData = {}
      try {
        uploadData = uploadText ? JSON.parse(uploadText) : {}
      } catch {
        uploadData = { raw: uploadText }
      }

      if (!uploadRes.ok) {
        throw new Error(
          uploadData.detail ||
          uploadData.message ||
          `Upload failed with server status ${uploadRes.status}`
        )
      }

      const imagePath =
        uploadData.file_path ||
        uploadData.image_path ||
        uploadData.path ||
        uploadData.filename

      if (!imagePath) {
        throw new Error('Upload succeeded, but backend did not return a valid image path.')
      }

      // Step 2: Call /predict with image_path and optional farm_id
      setUploadPhase('Running 3D-CNN hyperspectral prediction...')

      let predictUrl = `${API_BASE_URL}/predict?image_path=${encodeURIComponent(imagePath)}`
      if (farmId) {
        predictUrl += `&farm_id=${encodeURIComponent(farmId)}`
      }

      const predictRes = await fetch(predictUrl, {
        method: 'POST'
      })

      const predictText = await predictRes.text()
      let predictData = {}
      try {
        predictData = predictText ? JSON.parse(predictText) : {}
      } catch {
        throw new Error('Prediction service returned an unreadable response format.')
      }

      if (!predictRes.ok) {
        throw new Error(
          predictData.detail ||
          predictData.message ||
          `Prediction model returned error code ${predictRes.status}`
        )
      }

      setPrediction(predictData)
      setUploadPhase('')
    } catch (err) {
      console.error('Prediction workflow error:', err)
      setError({
        title: 'Analysis Could Not Be Completed',
        message: err?.message || 'Unable to connect to the TerraSpectra prediction service. Please ensure the backend is running and try again.'
      })
      setUploadPhase('')
    } finally {
      setLoading(false)
    }
  }

  // ---------------------------------------------------------
  // DERIVED STATUS HELPERS
  // ---------------------------------------------------------
  const isStressed = () => {
    if (!prediction) return false
    const predStr = String(prediction.prediction || prediction.class_name || '').toLowerCase()
    const prob = Number(prediction.stress_probability || 0)
    return predStr.includes('stress') || prob >= 0.5
  }

  const getPredictionTitle = () => {
    if (!prediction) return ''
    return prediction.class_name || prediction.prediction || 'Analysis Complete'
  }

  const getStatusLabel = () => {
    if (!prediction) return ''
    if (prediction.status && prediction.status !== 'success') {
      return prediction.status
    }
    return isStressed() ? 'Chemical Stress Detected' : 'Optimal Vegetative Health'
  }

  return (
    <div id="prediction" className="cip-card">
      {/* Header */}
      <div className="cip-header">
        <div className="cip-header-title-group">
          <h2>
            Crop Image Prediction
          </h2>
          <p>
            Upload a hyperspectral crop scan to assess crop health and detect chemical stress using TerraSpectra intelligence.
          </p>
        </div>

        <div className="cip-header-badge-group">
          <span className="cip-tech-badge accent">
            <span className="cip-badge-dot"></span>
            Hybrid 3D-CNN Model
          </span>
          <span className="cip-tech-badge">
            125 Spectral Bands
          </span>
        </div>
      </div>

      {/* Hidden file input */}
      <input
        ref={fileInputRef}
        type="file"
        accept=".npy"
        onChange={handleFileChange}
        style={{ display: 'none' }}
        id="crop-image"
      />

      {/* -----------------------------------------------------
          UPLOAD STATE (No file selected)
      ----------------------------------------------------- */}
      {!selectedFile && !loading && (
        <div
          className={`cip-upload-zone ${isDragging ? 'is-dragging' : ''}`}
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onDrop={handleDrop}
          onClick={triggerBrowse}
          role="button"
          tabIndex={0}
          onKeyDown={(e) => {
            if (e.key === 'Enter' || e.key === ' ') {
              e.preventDefault()
              triggerBrowse()
            }
          }}
        >
          <div className="cip-upload-icon-wrap">
            <svg
              width="24"
              height="24"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="2"
              strokeLinecap="round"
              strokeLinejoin="round"
              aria-hidden="true"
            >
              <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
              <polyline points="17 8 12 3 7 8" />
              <line x1="12" y1="3" x2="12" y2="15" />
            </svg>
          </div>

          <h3 className="cip-upload-title">
            Upload Hyperspectral Crop Image
          </h3>

          <p className="cip-upload-desc">
            Upload an NPY hyperspectral scan for AI-powered crop health analysis
          </p>

          <span className="cip-upload-meta">
            NPY • 125 spectral bands • Max 50 MB
          </span>

          <button
            type="button"
            className="cip-browse-btn"
            onClick={(e) => {
              e.stopPropagation()
              triggerBrowse()
            }}
          >
            Browse Files
          </button>
        </div>
      )}

      {/* -----------------------------------------------------
          SELECTED FILE STATE
      ----------------------------------------------------- */}
      {selectedFile && !loading && !prediction && (
        <div className="cip-file-card">
          <div className="cip-file-left">
            <div className="cip-file-avatar">
              NPY
              <span>CUBE</span>
            </div>

            <div className="cip-file-details">
              <span className="cip-file-name" title={selectedFile.name}>
                {selectedFile.name}
              </span>
              <div className="cip-file-meta-row">
                <span>{formatFileSize(selectedFile.size)}</span>
                <span>•</span>
                <span className="cip-file-pill">125 Spectral Bands</span>
                <span>•</span>
                <span>Ready for Inference</span>
              </div>
            </div>
          </div>

          <div className="cip-file-actions">
            <button
              type="button"
              className="cip-btn-secondary"
              onClick={triggerBrowse}
            >
              Replace File
            </button>
            <button
              type="button"
              className="cip-btn-danger"
              onClick={removeSelectedFile}
            >
              Remove
            </button>
            <button
              type="button"
              className="cip-btn-analyze"
              onClick={handlePrediction}
            >
              Analyze Crop Image
            </button>
          </div>
        </div>
      )}

      {/* -----------------------------------------------------
          LOADING STATE
      ----------------------------------------------------- */}
      {loading && (
        <div className="cip-loading-panel">
          <div className="cip-spinner-wrap">
            <div className="cip-spinner" />
          </div>

          <h3 className="cip-loading-title">
            Analyzing Hyperspectral Image
          </h3>

          <p className="cip-loading-desc">
            {uploadPhase || 'Processing spectral signatures and crop health indicators...'}
          </p>

          <div className="cip-loading-progress-bar">
            <div className="cip-loading-progress-fill" />
          </div>
        </div>
      )}

      {/* -----------------------------------------------------
          ERROR STATE
      ----------------------------------------------------- */}
      {error && !loading && (
        <div className="cip-error-box">
          <div className="cip-error-content">
            <h4 className="cip-error-title">
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <circle cx="12" cy="12" r="10" />
                <line x1="12" y1="8" x2="12" y2="12" />
                <line x1="12" y1="16" x2="12.01" y2="16" />
              </svg>
              {error.title}
            </h4>
            <p className="cip-error-msg">{error.message}</p>
          </div>

          <div className="cip-error-actions">
            <button
              type="button"
              className="cip-retry-btn"
              onClick={handlePrediction}
            >
              Retry Analysis
            </button>
            <button
              type="button"
              className="cip-btn-secondary"
              onClick={triggerBrowse}
            >
              Upload Different File
            </button>
          </div>
        </div>
      )}

      {/* -----------------------------------------------------
          PREDICTION RESULT & ANALYSIS SUMMARY
      ----------------------------------------------------- */}
      {prediction && !loading && (
        <div className="cip-result-card">
          {/* Result Card Header */}
          <div className="cip-result-header">
            <div className="cip-result-title-group">
              <h3>
                AI Crop Health Analysis
              </h3>
              <span>
                Hyperspectral 3D-CNN Diagnostic Completed
              </span>
            </div>

            <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
              <span className="cip-result-status-tag complete">
                ✓ Inference Complete
              </span>
              <button
                type="button"
                className="cip-btn-secondary"
                style={{ padding: '5px 12px', fontSize: '12px' }}
                onClick={removeSelectedFile}
              >
                Upload New Scan
              </button>
            </div>
          </div>

          {/* Result Hero Banner */}
          <div className={`cip-hero-banner ${isStressed() ? 'stressed' : 'healthy'}`}>
            <div className="cip-hero-left">
              <span className="cip-hero-eyebrow">Primary Classification</span>
              <h2 className="cip-hero-prediction">
                {getPredictionTitle()}
              </h2>
            </div>

            <div className="cip-hero-right">
              <span className={`cip-hero-badge ${isStressed() ? 'badge-stressed' : 'badge-healthy'}`}>
                {isStressed() ? '⚠️' : '✓'} {getStatusLabel()}
              </span>
            </div>
          </div>

          {/* 4-Metric Grid */}
          <div className="cip-metric-grid">
            <div className="cip-metric-item">
              <span className="cip-metric-label">Prediction</span>
              <strong className="cip-metric-value" style={{ fontSize: '16px' }}>
                {prediction.prediction || prediction.class_name || 'N/A'}
              </strong>
              <span className="cip-metric-sub">Classification output</span>
            </div>

            <div className="cip-metric-item">
              <span className="cip-metric-label">Confidence</span>
              <strong className="cip-metric-value">
                {formatConfidence(prediction.confidence)}
              </strong>
              <span className="cip-metric-sub">Model certainty</span>
            </div>

            <div className="cip-metric-item">
              <span className="cip-metric-label">Stress Probability</span>
              <strong
                className="cip-metric-value"
                style={{ color: isStressed() ? '#d97706' : '#166534' }}
              >
                {formatConfidence(prediction.stress_probability)}
              </strong>
              <span className="cip-metric-sub">Pathogen / chemical indicator</span>
            </div>

            <div className="cip-metric-item">
              <span className="cip-metric-label">Tiles Processed</span>
              <strong className="cip-metric-value">
                {prediction.tiles_processed !== undefined ? prediction.tiles_processed : 'N/A'}
              </strong>
              <span className="cip-metric-sub">32x32 spatial cubes evaluated</span>
            </div>
          </div>

          {/* Analysis Summary Section */}
          <div className="cip-summary-section">
            <h4>
              Analysis Summary
            </h4>

            <table className="cip-summary-table">
              <tbody>
                <tr>
                  <th>Crop Health Status</th>
                  <td>
                    <strong>
                      {prediction.prediction || prediction.class_name || (isStressed() ? 'Chemically Stressed' : 'Healthy')}
                    </strong>
                  </td>
                </tr>
                <tr>
                  <th>Confidence</th>
                  <td>
                    <strong>{formatConfidence(prediction.confidence)}</strong>
                  </td>
                </tr>
                <tr>
                  <th>Stress Probability</th>
                  <td>
                    <strong>{formatConfidence(prediction.stress_probability)}</strong>
                  </td>
                </tr>
                <tr>
                  <th>Tiles Analyzed</th>
                  <td>
                    <strong>{prediction.tiles_processed !== undefined ? prediction.tiles_processed : 4}</strong>
                  </td>
                </tr>
                <tr>
                  <th>Analysis Type</th>
                  <td>Hyperspectral AI Analysis</td>
                </tr>
                {selectedFile?.name && (
                  <tr>
                    <th>Source Scan</th>
                    <td>{selectedFile.name} ({formatFileSize(selectedFile.size)})</td>
                  </tr>
                )}
                {farmId && (
                  <tr>
                    <th>Monitored Farm</th>
                    <td>
                      {farmName ? `${farmName} (${farmId})` : farmId}
                    </td>
                  </tr>
                )}
                {prediction.prediction_id && (
                  <tr>
                    <th>Record ID</th>
                    <td>#{prediction.prediction_id}</td>
                  </tr>
                )}
              </tbody>
            </table>

            {/* Scientific Explanation Box */}
            <div className={`cip-explanation-box ${isStressed() ? 'alert-tone' : ''}`}>
              <div className="cip-explanation-header">
                <strong>Model Diagnostic Assessment</strong>
              </div>
              <p className="cip-explanation-text">
                {isStressed()
                  ? 'The hyperspectral analysis indicates potential chemical stress in the crop. The model identified spectral patterns associated with chemical stress across the analyzed crop tiles.'
                  : 'The hyperspectral analysis indicates healthy vegetative vigor across the scanned crop area. Spectral reflectance curves align with optimal chlorophyll absorption across all processed tiles.'}
              </p>
            </div>

            {/* Backend Recommendation Message */}
            {prediction.message && (
              <div className="cip-recommendation-box">
                <span className="cip-recommendation-icon">ℹ️</span>
                <p className="cip-recommendation-text">
                  <strong>Recommendation:</strong> {prediction.message}
                </p>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  )
}

export default CropImagePrediction
