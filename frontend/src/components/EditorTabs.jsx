export default function EditorTabs({ tabs, activePath, onSelect, onClose }) {
  if (tabs.length === 0) return null
  return (
    <div className="editor-tabs">
      {tabs.map((tab) => (
        <div
          key={tab.path}
          className={`editor-tab ${tab.path === activePath ? 'editor-tab-active' : ''}`}
          onClick={() => onSelect(tab.path)}
        >
          {tab.pendingRunId ? (
            <span className="editor-tab-pending-dot" title="awaiting your review" />
          ) : (
            tab.isLive && <span className="editor-tab-live-dot" title="just edited" />
          )}
          <span className="editor-tab-name">{tab.path.split('/').pop()}</span>
          <button
            className="editor-tab-close"
            onClick={(e) => {
              e.stopPropagation()
              onClose(tab.path)
            }}
          >
            ×
          </button>
        </div>
      ))}
    </div>
  )
}
