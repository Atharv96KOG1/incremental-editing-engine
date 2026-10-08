import { useState } from 'react'

// Shown when a "remove/delete X" request's target word matched more than
// one real symbol -- a deletion is destructive and permanent, so the
// backend refuses to guess and sends back every plausible match instead.
// Chips toggle selection (multiple at once); confirming resubmits with
// every selected symbol confirmed, which skips the LLM entirely (each
// DELETE's shape is already fully known once confirmed).
export default function CandidatePicker({ candidates, onConfirm, disabled }) {
  const [selected, setSelected] = useState(() => new Set())

  const key = (c) => `${c.symbol_type}:${c.name}:${c.start_line}`

  const toggle = (c) => {
    if (disabled) return
    setSelected((prev) => {
      const next = new Set(prev)
      const k = key(c)
      if (next.has(k)) next.delete(k)
      else next.add(k)
      return next
    })
  }

  const selectedCandidates = candidates.filter((c) => selected.has(key(c)))

  return (
    <div className="candidate-picker">
      <div className="candidate-picker-title">multiple matches — select one or more to delete</div>
      <div className="candidate-picker-list">
        {candidates.map((c) => (
          <button
            key={key(c)}
            type="button"
            className={`candidate-chip${selected.has(key(c)) ? ' candidate-chip-selected' : ''}`}
            disabled={disabled}
            aria-pressed={selected.has(key(c))}
            onClick={() => toggle(c)}
          >
            <span className="candidate-chip-name">{c.name}</span>
            <span className="candidate-chip-meta">
              {c.symbol_type} · lines {c.start_line}-{c.end_line}
            </span>
          </button>
        ))}
      </div>
      <button
        type="button"
        className="candidate-picker-confirm"
        disabled={disabled || selectedCandidates.length === 0}
        onClick={() => onConfirm(selectedCandidates)}
      >
        Delete {selectedCandidates.length || ''} selected
      </button>
    </div>
  )
}
