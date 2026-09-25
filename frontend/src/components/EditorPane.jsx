import { useEffect, useMemo, useState } from 'react'
import { buildLiveEditView } from '../lib/parseDiff.js'

const GHOST_LIFETIME_MS = 2200
const HIGHLIGHT_LIFETIME_MS = 2600

export default function EditorPane({ tab, onAccept, onReject }) {
  // A live edit re-arms this pane's animation every time `tab.revision`
  // changes (a fresh preview/diff landed for this exact file), even if the
  // diff text happens to repeat -- so the "just edited" flash is always
  // tied to an actual new event, not just to content that differs. While a
  // run is awaiting human accept/reject, stay "live" indefinitely instead
  // of auto-settling after a couple seconds -- the review isn't done yet.
  const [phase, setPhase] = useState('settled') // 'live' | 'settled'

  useEffect(() => {
    if (!tab?.diff) return
    setPhase('live')
    if (tab.pendingRunId) return
    const t = setTimeout(() => setPhase('settled'), HIGHLIGHT_LIFETIME_MS)
    return () => clearTimeout(t)
    // eslint-disable-next-line react-hooks/exhaustive-deps -- keyed on revision
    // on purpose: re-arm even if a repeat edit produces an identical diff.
  }, [tab?.revision, tab?.pendingRunId])

  const [showGhosts, setShowGhosts] = useState(true)
  useEffect(() => {
    if (!tab?.diff) return
    setShowGhosts(true)
    if (tab.pendingRunId) return
    const t = setTimeout(() => setShowGhosts(false), GHOST_LIFETIME_MS)
    return () => clearTimeout(t)
    // eslint-disable-next-line react-hooks/exhaustive-deps -- same as above
  }, [tab?.revision, tab?.pendingRunId])

  const { addedLines, ghostBlocks } = useMemo(() => {
    if (phase !== 'live' || !tab?.diff) return { addedLines: new Set(), ghostBlocks: [] }
    return buildLiveEditView(tab.diff)
  }, [tab?.diff, phase])

  if (!tab) {
    return (
      <div className="editor-pane editor-pane-empty">
        <div className="editor-empty-hint">Select a file from the explorer, or send a request to see a live edit.</div>
      </div>
    )
  }

  const lines = (tab.content ?? '').split('\n')
  const ghostsByLine = new Map()
  if (showGhosts) {
    for (const block of ghostBlocks) {
      const list = ghostsByLine.get(block.beforeNewLine) || []
      list.push(block)
      ghostsByLine.set(block.beforeNewLine, list)
    }
  }

  return (
    <div className="editor-pane">
      <div className="editor-pane-path">
        {tab.path}
        {phase === 'live' && !tab.pendingRunId && <span className="editor-live-badge">● editing…</span>}
      </div>
      {tab.pendingRunId && (
        <div className={`confirm-bar${tab.pendingDelete ? ' confirm-bar-delete' : ''}`}>
          <span className="confirm-bar-label">
            {tab.pendingDelete
              ? '🗑 this file will be permanently deleted — review before confirming'
              : tab.testsPassed === false
                ? '⚠ tests failed — review before accepting'
                : tab.testsPassed === true
                  ? '✓ tests passed — review the change above'
                  : 'review the change above'}
          </span>
          <div className="confirm-bar-actions">
            {tab.confirmError && <span className="confirm-bar-error">{tab.confirmError}</span>}
            <button className="confirm-btn confirm-btn-reject" disabled={tab.confirming} onClick={onReject}>
              {tab.pendingDelete ? 'Cancel' : 'Reject'}
            </button>
            <button className="confirm-btn confirm-btn-accept" disabled={tab.confirming} onClick={onAccept}>
              {tab.confirming ? (tab.pendingDelete ? 'Deleting…' : 'Applying…') : tab.pendingDelete ? 'Delete' : 'Accept'}
            </button>
          </div>
        </div>
      )}
      <div className="editor-code">
        {lines.map((text, idx) => {
          const lineNo = idx + 1
          const ghosts = ghostsByLine.get(lineNo)
          return (
            <div key={lineNo}>
              {ghosts?.map((block, gi) =>
                block.lines.map((delText, li) => (
                  <div className="editor-line editor-line-ghost" key={`g-${gi}-${li}`}>
                    <span className="editor-line-no"> </span>
                    <span className="editor-line-text">{delText || ' '}</span>
                  </div>
                )),
              )}
              <div className={`editor-line ${addedLines.has(lineNo) ? 'editor-line-added' : ''}`}>
                <span className="editor-line-no">{lineNo}</span>
                <span className="editor-line-text">{text || ' '}</span>
              </div>
            </div>
          )
        })}
      </div>
    </div>
  )
}
