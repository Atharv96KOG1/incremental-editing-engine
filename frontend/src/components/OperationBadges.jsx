const OP_CLASS = {
  REPLACE: 'op-replace',
  INSERT: 'op-insert',
  DELETE: 'op-delete',
}

function lineLabel(lineRange) {
  if (!lineRange) return ''
  if (lineRange.after_line !== undefined) return ` · after line ${lineRange.after_line}`
  if (lineRange.start === lineRange.end) return ` · line ${lineRange.start}`
  return ` · lines ${lineRange.start}–${lineRange.end}`
}

export default function OperationBadges({ operations }) {
  if (!operations || operations.length === 0) return null

  return (
    <div className="op-badges">
      {operations.map((op, i) => (
        <span className={`op-badge ${OP_CLASS[op.operation] || ''}`} key={i}>
          {op.operation} <b>{op.target.symbol_name}</b>
          {op.target.anchor ? ` → after ${op.target.anchor}` : ''}
          <span className="op-badge-lines">{lineLabel(op.line_range)}</span>
        </span>
      ))}
    </div>
  )
}
