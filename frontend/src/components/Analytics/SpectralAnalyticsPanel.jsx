import { useState } from 'react'
import { getAcreageAnalytics } from '../../services/gisService'

function SpectralAnalyticsPanel({ currentWeek = 0 }) {
  const [selectedBand, setSelectedBand] = useState('705nm')
  const analytics = getAcreageAnalytics(currentWeek)

  // 200+ Band Hyperspectral Reflection Profile points (Wavelength in nm vs Reflectance %)
  // Compares Healthy Crop vs Fungal Blight Infected Crop
  const spectralData = [
    { wl: 450, healthy: 5, infected: 6, band: 'Band 15 (Blue)' },
    { wl: 500, healthy: 9, infected: 11, band: 'Band 28 (Blue-Green)' },
    { wl: 550, healthy: 18, infected: 15, band: 'Band 42 (Green Peak)' },
    { wl: 600, healthy: 12, infected: 14, band: 'Band 58 (Yellow)' },
    { wl: 670, healthy: 6, infected: 14, band: 'Band 78 (Chlorophyll Max Absorption)' },
    { wl: 705, healthy: 32, infected: 16, band: 'Band 88 (Red Edge Inflection - CRITICAL)' },
    { wl: 740, healthy: 54, infected: 35, band: 'Band 98 (Red Edge Shoulder)' },
    { wl: 800, healthy: 68, infected: 48, band: 'Band 112 (NIR Plateau)' },
    { wl: 860, healthy: 70, infected: 49, band: 'Band 128 (NIR Cellular Structure)' },
    { wl: 920, healthy: 66, infected: 44, band: 'Band 140 (Moisture Transition)' },
    { wl: 970, healthy: 58, infected: 32, band: 'Band 148 (Water Absorption Dip)' },
    { wl: 1020, healthy: 64, infected: 42, band: 'Band 160 (Shortwave NIR)' }
  ]

  return (
    <div id="analytics-panel" className="analytics-card">
      <div className="analytics-header">
        <div>
          <span className="analytics-tag">Week 3 GIS Milestone</span>
          <h2>Acreage at Risk & Spectral Chemical Anomalies</h2>
          <p>3D-CNN & ViT 200+ Band Hyperspectral Chlorophyll Diagnostic</p>
        </div>

        <div className="lead-time-badge">
          <span className="badge-icon">⏱</span>
          <div>
            <strong>21 Days Lead Time</strong>
            <span>3 Weeks Before Leaf Yellowing</span>
          </div>
        </div>
      </div>

      {/* ================= ACREAGE RISK BREAKDOWN ================= */}
      <div className="acreage-summary-grid">
        <div className="acreage-stat total">
          <span className="stat-label">Total Monitored Farm</span>
          <strong className="stat-value">{analytics.totalFarmAcres}</strong>
          <span className="stat-unit">Acres (Wadgaon Site)</span>
        </div>

        <div className="acreage-stat critical">
          <span className="stat-label">High-Risk Outbreak Zone</span>
          <strong className="stat-value">{analytics.highRiskAcres}</strong>
          <span className="stat-unit">Acres (Parcel C - Zone Alpha)</span>
          <span className="stat-alert">Fungal Blight Infiltration</span>
        </div>

        <div className="acreage-stat moderate">
          <span className="stat-label">Moderate Stress Buffer</span>
          <strong className="stat-value">{analytics.moderateStressAcres}</strong>
          <span className="stat-unit">Acres Under Watch</span>
        </div>

        <div className="acreage-stat healthy">
          <span className="stat-label">Healthy Vegetative Canopy</span>
          <strong className="stat-value">{analytics.healthyAcres}</strong>
          <span className="stat-unit">Acres Unaffected</span>
        </div>
      </div>

      {/* Acreage Proportional Bar */}
      <div className="acreage-bar-wrapper">
        <div className="acreage-bar">
          <div
            className="bar-segment bar-critical"
            style={{ width: `${(analytics.highRiskAcres / analytics.totalFarmAcres) * 100}%` }}
            title={`High Risk: ${analytics.highRiskAcres} Acres`}
          />
          <div
            className="bar-segment bar-moderate"
            style={{ width: `${(analytics.moderateStressAcres / analytics.totalFarmAcres) * 100}%` }}
            title={`Moderate Stress: ${analytics.moderateStressAcres} Acres`}
          />
          <div
            className="bar-segment bar-healthy"
            style={{ width: `${(analytics.healthyAcres / analytics.totalFarmAcres) * 100}%` }}
            title={`Healthy Canopy: ${analytics.healthyAcres} Acres`}
          />
        </div>

        <div className="acreage-savings-callout">
          <span className="savings-icon">💡</span>
          <span>
            <strong>Hyper-Targeted Treatment:</strong> Spraying only the 5.2-acre zone instead of blanket-treating the entire 1,000-acre farm achieves <strong>94.8% pesticide reduction</strong>, saving an estimated <strong>$38,500</strong> in operational costs while preventing blight dissemination.
          </span>
        </div>
      </div>

      {/* ================= CHEMICAL ANOMALIES BREAKDOWN ================= */}
      <div className="chemical-anomalies-section">
        <h3>Specific Chemical Anomalies Detected</h3>

        <div className="chemical-grid">
          {/* Anomaly 1 */}
          <div className="chemical-card alert">
            <div className="chemical-card-header">
              <span className="chem-icon">🌱</span>
              <h4>Chlorophyll a/b Reflection Dip</h4>
            </div>
            <div className="chem-stat-row">
              <span className="chem-delta">-28.4%</span>
              <span className="chem-band">Band 88 (705.4 nm)</span>
            </div>
            <p>Subtle steepness decrease in the Red Edge inflection band indicates early chlorophyll breakdown before naked-eye chlorosis.</p>
          </div>

          {/* Anomaly 2 */}
          <div className="chemical-card warning">
            <div className="chemical-card-header">
              <span className="chem-icon">⚡</span>
              <h4>Photochemical Reflectance Index</h4>
            </div>
            <div className="chem-stat-row">
              <span className="chem-delta">-0.142 PRI</span>
              <span className="chem-band">Band 32 vs Band 44</span>
            </div>
            <p>Drastic reduction in photosynthetic xanthophyll cycle efficiency, signaling cellular defense activation.</p>
          </div>

          {/* Anomaly 3 */}
          <div className="chemical-card info">
            <div className="chemical-card-header">
              <span className="chem-icon">💧</span>
              <h4>Canopy Water Index (CWI)</h4>
            </div>
            <div className="chem-stat-row">
              <span className="chem-delta">-18.8%</span>
              <span className="chem-band">Band 148 (970.0 nm)</span>
            </div>
            <p>Internal mesophyll cellular water retention drops as fungal hyphae disrupt xylem transport.</p>
          </div>

          {/* Anomaly 4 */}
          <div className="chemical-card secondary">
            <div className="chemical-card-header">
              <span className="chem-icon">🍂</span>
              <h4>Carotenoid-to-Chlorophyll Ratio</h4>
            </div>
            <div className="chem-stat-row">
              <span className="chem-delta">+34.1%</span>
              <span className="chem-band">Band 22 / Band 82</span>
            </div>
            <p>Antioxidant carotenoid accumulation triggered by oxidative stress responses to fungal pathogen colonization.</p>
          </div>
        </div>
      </div>

      {/* ================= SPECTRAL REFLECTANCE SIGNATURE GRAPH ================= */}
      <div className="spectral-curve-section">
        <div className="spectral-curve-header">
          <div>
            <h3>200-Band Hyperspectral Signature Comparison</h3>
            <p>Wavelength (400nm - 1050nm) vs Canopy Reflectance (%)</p>
          </div>

          <div className="spectral-legend">
            <span className="legend-healthy"><span className="dot"></span> Healthy Canopy</span>
            <span className="legend-infected"><span className="dot"></span> Fungal Blight Hotspot</span>
            <span className="legend-critical-band"><span className="dot"></span> 705nm Red Edge Dip</span>
          </div>
        </div>

        <div className="svg-chart-container">
          <svg viewBox="0 0 700 240" className="spectral-svg">
            {/* Grid Lines */}
            <line x1="60" y1="30" x2="680" y2="30" stroke="#e5e7eb" strokeDasharray="3 3" />
            <line x1="60" y1="80" x2="680" y2="80" stroke="#e5e7eb" strokeDasharray="3 3" />
            <line x1="60" y1="130" x2="680" y2="130" stroke="#e5e7eb" strokeDasharray="3 3" />
            <line x1="60" y1="180" x2="680" y2="180" stroke="#e5e7eb" strokeDasharray="3 3" />
            <line x1="60" y1="210" x2="680" y2="210" stroke="#9ca3af" />

            {/* Y Axis Labels */}
            <text x="45" y="35" fontSize="11" fill="#6b7280" textAnchor="end">70%</text>
            <text x="45" y="85" fontSize="11" fill="#6b7280" textAnchor="end">50%</text>
            <text x="45" y="135" fontSize="11" fill="#6b7280" textAnchor="end">30%</text>
            <text x="45" y="185" fontSize="11" fill="#6b7280" textAnchor="end">10%</text>

            {/* Red Edge 705nm Target Zone Highlight */}
            <rect x="330" y="20" width="65" height="190" fill="#fee2e2" opacity="0.65" rx="4" />
            <text x="362" y="18" fontSize="10" fill="#dc2626" textAnchor="middle" fontWeight="bold">
              705nm Red Edge Anomaly
            </text>

            {/* Path: Healthy Canopy Curve (Green) */}
            <path
              d="M 70 198 Q 120 190 170 165 T 270 195 T 350 120 T 450 45 T 560 40 T 670 55"
              fill="none"
              stroke="#22c55e"
              strokeWidth="3.5"
            />

            {/* Path: Infected Canopy Curve (Red/Orange) */}
            <path
              d="M 70 195 Q 120 185 170 170 T 270 180 T 350 165 T 450 95 T 560 92 T 670 115"
              fill="none"
              stroke="#ef4444"
              strokeWidth="3.5"
              strokeDasharray="5 3"
            />

            {/* Data Markers */}
            {spectralData.map((d, i) => {
              const x = 70 + (i * 54)
              const yHealthy = 210 - (d.healthy * 2.5)
              const yInfected = 210 - (d.infected * 2.5)
              const is705 = d.wl === 705

              return (
                <g key={d.wl} onClick={() => setSelectedBand(`${d.wl}nm`)} style={{ cursor: 'pointer' }}>
                  <circle cx={x} cy={yHealthy} r={is705 ? 6 : 4} fill="#22c55e" stroke="#fff" strokeWidth="1.5" />
                  <circle cx={x} cy={yInfected} r={is705 ? 6 : 4} fill="#ef4444" stroke="#fff" strokeWidth="1.5" />
                  <text x={x} y="228" fontSize="10" fill={is705 ? '#dc2626' : '#6b7280'} textAnchor="middle" fontWeight={is705 ? 'bold' : 'normal'}>
                    {d.wl}
                  </text>
                </g>
              )
            })}
          </svg>
        </div>

        <div className="spectral-note">
          <p>
            <strong>Analytical Insight:</strong> The sharp diverge at <strong>705.4nm (Band 88)</strong> reveals a <strong>-28.4% reflection drop</strong> caused by microscopic chlorophyll cell degradation. This subtle shift is captured exclusively by hyperspectral sensors and decoded by the 3D-CNN & ViT model <strong>21 days ahead</strong> of conventional satellite observation.
          </p>
        </div>
      </div>
    </div>
  )
}

export default SpectralAnalyticsPanel
