export default function Sidebar({ config, onConfigChange, onNewChat }) {
  const set = (key) => (e) => onConfigChange({ ...config, [key]: e.target.value })

  return (
    <aside className="sidebar">
      <button className="new-chat-btn" onClick={onNewChat}>
        <span className="new-chat-icon">+</span>
        New chat
      </button>

      <div className="sidebar-section">
        <div className="sidebar-label">Mode</div>
        <select className="sidebar-input" value={config.mode} onChange={set('mode')}>
          <option value="edit">Edit existing file</option>
          <option value="create">Create new file</option>
          <option value="find">Find (locate only, no editing)</option>
        </select>
      </div>

      <div className="sidebar-section">
        <div className="sidebar-label">Project directory</div>
        <input
          className="sidebar-input"
          value={config.projectDir}
          onChange={set('projectDir')}
          placeholder="e.g. my_project"
        />
      </div>

      {config.mode !== 'find' && (
        <div className="sidebar-section">
          <div className="sidebar-label">File{config.mode === 'edit' ? ' (optional)' : ''}</div>
          <input
            className="sidebar-input"
            value={config.file}
            onChange={set('file')}
            placeholder={config.mode === 'edit' ? 'leave blank to auto-locate' : ''}
          />
        </div>
      )}

      {config.mode === 'edit' && (
        <div className="sidebar-section">
          <div className="sidebar-label">Test target</div>
          <input className="sidebar-input" value={config.testTarget} onChange={set('testTarget')} />
        </div>
      )}

      {config.mode === 'find' && (
        <>
          <div className="sidebar-section">
            <div className="sidebar-label">Joern call graph</div>
            <select className="sidebar-input" value={config.useJoern} onChange={set('useJoern')}>
              <option value="off">Off — native call graph only</option>
              <option value="auto">Auto — only if confidence is low</option>
              <option value="on">Always</option>
            </select>
            <div className="sidebar-hint">
              A real CPG instead of the name-only native graph — more accurate across files. Real cost when it
              runs: ~12-45s+ depending on languages present (JVM startup, cached after the first run). "Auto" only
              pays that cost when the top match barely beat the runner-up. Requires Joern installed on the server
              — falls back to the native graph automatically if it isn't.
            </div>
          </div>
          <div className="sidebar-hint">
            Hybrid retrieval only (symbol + BM25 + vector + Semgrep) — shows which file/symbol matches, doesn't edit
            anything.
          </div>
        </>
      )}

      <div className="sidebar-footer">Adaptive Incremental Editing Engine</div>
    </aside>
  )
}
