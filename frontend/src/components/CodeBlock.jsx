export default function CodeBlock({ code, label }) {
  if (!code) return null
  return (
    <div className="code-block">
      {label && <div className="code-block-label">{label}</div>}
      <pre className="code-block-pre">{code}</pre>
    </div>
  )
}
