// Vertical, append-only timeline -- each pipeline step becomes a row as it
// actually happens (GitHub Actions / Vercel deploy-log style), not a fixed
// pre-drawn map of every possible stage. The last row spins until either
// another step arrives or the run reaches a final state.

export default function Timeline({ steps, finalState }) {
  if (steps.length === 0) return null

  return (
    <div className="timeline">
      {steps.map((s, i) => {
        const isLast = i === steps.length - 1
        let status = 'done'
        if (isLast && !finalState) status = 'active'
        else if (isLast && finalState === 'error') status = 'error'

        return (
          <div className="timeline-row" key={i}>
            <div className="timeline-rail">
              <div className={`timeline-dot timeline-dot-${status}`}>
                {status === 'done' && '✓'}
                {status === 'error' && '✗'}
                {status === 'active' && <span className="timeline-spinner" />}
              </div>
              {!isLast && <div className="timeline-line" />}
            </div>
            <div className="timeline-content">
              <span className={`timeline-tag tag-${s.tag.toLowerCase()}`}>{s.tag}</span>
              <span className="timeline-msg">{s.msg}</span>
            </div>
          </div>
        )
      })}
    </div>
  )
}
