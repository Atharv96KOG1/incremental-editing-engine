const RISK_CLASS = { HIGH: 'risk-high', MEDIUM: 'risk-medium', LOW: 'risk-low' }

export default function RetrievalResults({ metadata }) {
  const candidates = metadata.candidates || []

  return (
    <div className="retrieval-results">
      <div className="retrieval-meta">
        symbols indexed: <b>{metadata.symbols_indexed}</b> &nbsp;·&nbsp; confidence:{' '}
        <b>{metadata.confidence}</b>
      </div>

      {candidates.length === 0 ? (
        <div className="retrieval-empty">no candidates found</div>
      ) : (
        <table className="retrieval-table">
          <thead>
            <tr>
              <th>file</th>
              <th>symbol</th>
              <th>lines</th>
              <th>score</th>
              <th>risk</th>
              <th>callers</th>
            </tr>
          </thead>
          <tbody>
            {candidates.map((c, i) => (
              <tr key={i}>
                <td>{c.file}</td>
                <td>{c.symbol}</td>
                <td>
                  {c.start_line}-{c.end_line}
                </td>
                <td>{c.fused_score}</td>
                <td>
                  <span className={`risk-badge ${RISK_CLASS[c.risk] || ''}`}>{c.risk}</span>
                </td>
                <td>{c.semgrep_call_sites.length || c.called_by_count}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  )
}
