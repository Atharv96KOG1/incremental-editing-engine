// Turns a unified diff (as produced by Python's difflib.unified_diff, what
// the pipeline sends as metadata.diff) into two things a line-numbered code
// view needs to animate a "live edit": which line numbers in the *new*
// file are additions, and where deleted lines used to sit so they can be
// shown briefly (fading out) at the right spot before settling.

const HUNK_RE = /^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@/

export function parseUnifiedDiff(diffText) {
  if (!diffText) return []
  const hunks = []
  let current = null

  for (const line of diffText.split('\n')) {
    const m = HUNK_RE.exec(line)
    if (m) {
      current = { newStart: parseInt(m[3], 10), entries: [] }
      hunks.push(current)
      continue
    }
    if (!current) continue // skip the "--- a/x" / "+++ b/x" file header lines
    if (line.startsWith('+')) current.entries.push({ type: 'add', text: line.slice(1) })
    else if (line.startsWith('-')) current.entries.push({ type: 'del', text: line.slice(1) })
    else if (line.startsWith(' ')) current.entries.push({ type: 'ctx', text: line.slice(1) })
    // a bare '\' (no-newline marker) or blank trailing line: ignored
  }
  return hunks
}

// addedLines: Set of 1-based line numbers in the new file that are additions
// ghostBlocks: [{ beforeNewLine, lines: string[] }] -- deleted text that
// used to sit immediately before that new-file line number
export function buildLiveEditView(diffText) {
  const hunks = parseUnifiedDiff(diffText)
  const addedLines = new Set()
  const ghostBlocks = []

  for (const hunk of hunks) {
    let cursor = hunk.newStart
    let pendingDel = []
    const flushDel = () => {
      if (pendingDel.length) {
        ghostBlocks.push({ beforeNewLine: cursor, lines: pendingDel })
        pendingDel = []
      }
    }
    for (const entry of hunk.entries) {
      if (entry.type === 'add') {
        addedLines.add(cursor)
        flushDel()
        cursor++
      } else if (entry.type === 'del') {
        pendingDel.push(entry.text)
      } else {
        flushDel()
        cursor++
      }
    }
    flushDel()
  }

  return { addedLines, ghostBlocks }
}
