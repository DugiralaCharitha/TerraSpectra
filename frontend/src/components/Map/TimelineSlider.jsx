import { useEffect, useRef } from 'react'
import { TIMELINE_STEPS } from '../../services/gisService'

function TimelineSlider({ currentWeek, onWeekChange, isPlaying, setIsPlaying }) {
  const timerRef = useRef(null)

  const currentStep = TIMELINE_STEPS.find(s => s.week === currentWeek) || TIMELINE_STEPS[2]

  // Automated time-lapse player
  useEffect(() => {
    if (isPlaying) {
      timerRef.current = setInterval(() => {
        onWeekChange(prev => {
          const currentIndex = TIMELINE_STEPS.findIndex(s => s.week === prev)
          const nextIndex = (currentIndex + 1) % TIMELINE_STEPS.length
          return TIMELINE_STEPS[nextIndex].week
        })
      }, 1800)
    } else {
      if (timerRef.current) {
        clearInterval(timerRef.current)
      }
    }

    return () => {
      if (timerRef.current) clearInterval(timerRef.current)
    }
  }, [isPlaying, onWeekChange])

  return (
    <div className="timeline-container">
      <div className="timeline-header">
        <div className="timeline-title-group">
          <h3>Multi-Temporal Crop Health Progression</h3>
        </div>

        <div className="timeline-controls">
          <button
            className={`timeline-play-btn ${isPlaying ? 'playing' : ''}`}
            onClick={() => setIsPlaying(!isPlaying)}
            title={isPlaying ? 'Pause time-lapse' : 'Play time-lapse animation'}
          >
            {isPlaying ? '⏸ Pause' : '▶ Play Time-Lapse'}
          </button>
        </div>
      </div>

      {/* Progress Track & Markers */}
      <div className="timeline-track-wrapper">
        <div className="timeline-track">
          <div
            className="timeline-track-fill"
            style={{
              width: `${((TIMELINE_STEPS.findIndex(s => s.week === currentWeek)) / (TIMELINE_STEPS.length - 1)) * 100}%`
            }}
          />

          {TIMELINE_STEPS.map((step) => {
            const isActive = step.week === currentWeek
            const isPast = step.week < 0
            const isToday = step.week === 0

            return (
              <button
                key={step.week}
                className={`timeline-node ${isActive ? 'active' : ''} ${isToday ? 'today' : ''} ${isPast ? 'past' : 'forecast'}`}
                onClick={() => {
                  setIsPlaying(false)
                  onWeekChange(step.week)
                }}
              >
                <span className="node-dot" style={{ backgroundColor: step.color }} />
                <span className="node-label">{step.label}</span>
                <span className="node-date">{step.dateOffset}</span>
              </button>
            )
          })}
        </div>
      </div>

      {/* Active Step Details Card */}
      <div className="timeline-details-card">
        <div className="detail-status-pill" style={{ backgroundColor: `${currentStep.color}22`, borderColor: currentStep.color, color: currentStep.color }}>
          <span className="pulsing-indicator" style={{ backgroundColor: currentStep.color }} />
          <strong>{currentStep.stage}</strong>
        </div>

        <div className="detail-metrics">
          <div className="detail-item">
            <span>Symptom Visibility</span>
            <strong className={currentStep.week <= 0 ? 'status-green' : 'status-danger'}>
              {currentStep.visibility}
            </strong>
          </div>

          <div className="detail-item">
            <span>Chlorophyll Refl. Dip</span>
            <strong className="status-danger">{currentStep.chlorophyllDip}</strong>
          </div>

          <div className="detail-item">
            <span>Photochemical Index (PRI)</span>
            <strong>{currentStep.pri}</strong>
          </div>

          <div className="detail-item">
            <span>At-Risk Acreage</span>
            <strong className="status-danger">{currentStep.affectedAcres} Acres</strong>
          </div>
        </div>

        <p className="detail-desc">{currentStep.description}</p>
      </div>
    </div>
  )
}

export default TimelineSlider
