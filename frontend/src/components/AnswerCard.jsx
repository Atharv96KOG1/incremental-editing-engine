// Rendered when the model recognized the request was a question about the
// code, not an edit instruction (result.status === "answered") -- no diff,
// no operations, nothing was written. Just the answer, plus the same real
// cost accounting every other result shows.
export default function AnswerCard({ metadata }) {
  const gen = metadata.generation
  return (
    <div className="answer-card">
      <div className="answer-card-text">{metadata.answer}</div>
      <div className="answer-card-footer">
        {gen.model} · {gen.total_tokens} tokens · ${gen.estimated_cost_usd.toFixed(6)} · {gen.latency_ms} ms
      </div>
    </div>
  )
}
