// Shown when a "remove/delete X" request's target word matched more than
// one real symbol -- a deletion is destructive and permanent, so the
// backend refuses to guess and sends back every plausible match instead.
// Picking one resubmits with that exact symbol confirmed, which skips the
// LLM entirely (the DELETE's shape is already fully known once confirmed).
export default function CandidatePicker({ candidates, onSelect, disabled }) {
  return (
    <div className="candidate-picker">
      <div className="candidate-picker-title">multiple matches — which one do you mean?</div>
      <div className="candidate-picker-list">
        {candidates.map((c) => (
          <button
            key={`${c.symbol_type}:${c.name}:${c.start_line}`}
            type="button"
            className="candidate-chip"
            disabled={disabled}
            onClick={() => onSelect(c)}
          >
            <span className="candidate-chip-name">{c.name}</span>
            <span className="candidate-chip-meta">
              {c.symbol_type} · lines {c.start_line}-{c.end_line}
            </span>
          </button>
        ))}
      </div>
    </div>
  )
}
