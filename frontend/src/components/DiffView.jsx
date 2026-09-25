function lineClass(line) {
  if (line.startsWith('+++') || line.startsWith('---')) return 'diff-file'
  if (line.startsWith('@@')) return 'diff-hunk'
  if (line.startsWith('+')) return 'diff-add'
  if (line.startsWith('-')) return 'diff-del'
  return 'diff-context'
}

export default function DiffView({ diff }) {
  if (!diff || !diff.trim()) return null
  const lines = diff.replace(/\n$/, '').split('\n')

  return (
    <div className="diff-view">
      {lines.map((line, i) => (
        <div className={`diff-line ${lineClass(line)}`} key={i}>
          {line === '' ? ' ' : line}
        </div>
      ))}
    </div>
  )
}
