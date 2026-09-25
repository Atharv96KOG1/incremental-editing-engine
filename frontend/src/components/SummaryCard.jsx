function Row({ label, children }) {
  return (
    <div className="summary-row">
      <div className="summary-key">{label}</div>
      <div className="summary-val">{children}</div>
    </div>
  )
}

export default function SummaryCard({ metadata }) {
  const ok = metadata.result.status === 'success'
  const rejected = metadata.result.status === 'rejected'
  const gen = metadata.generation

  return (
    <div className={`summary-card ${ok ? 'summary-success' : rejected ? 'summary-neutral' : 'summary-failed'}`}>
      <Row label="status">
        <span className={ok ? 'status-success' : rejected ? 'status-pending' : 'status-failed'}>
          {metadata.result.status}
        </span>
      </Row>
      <Row label="strategy">{metadata.strategy}</Row>
      {metadata.note && (
        <Row label="note">
          <span className="status-pending">{metadata.note}</span>
        </Row>
      )}
      {metadata.new_version && (
        <Row label="version">
          {metadata.base_version} → {metadata.new_version}
        </Row>
      )}
      {metadata.change_ratio !== undefined && (
        <Row label="change ratio">{(metadata.change_ratio * 100).toFixed(1)}%</Row>
      )}
      {metadata.context && (
        <Row label="context">
          {metadata.context.context_lines}/{metadata.context.total_lines} lines (symbols=
          {JSON.stringify(metadata.context.affected_symbols)})
        </Row>
      )}
      <Row label="model">{gen.model}</Row>
      <Row label="tokens">
        in={gen.input_tokens} out={gen.output_tokens} total={gen.total_tokens}
      </Row>
      <Row label="cost">${gen.estimated_cost_usd.toFixed(6)}</Row>
      <Row label="latency">{gen.latency_ms} ms</Row>
      {metadata.result.retry_count > 0 && (
        <Row label="repaired">
          <span className="op-badge op-replace">{metadata.result.retry_count} attempt(s)</span>
        </Row>
      )}
      {!ok && !rejected && metadata.result.failure_class && <Row label="failure class">{metadata.result.failure_class}</Row>}
      {!ok && !rejected && <Row label="error">{(metadata.error || 'unknown error').slice(0, 600)}</Row>}
    </div>
  )
}
