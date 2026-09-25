import { useEffect, useState } from 'react'

const FILE_ICON = {
  py: '🐍', js: '📄', jsx: '⚛️', ts: '📘', tsx: '⚛️', java: '☕',
  go: '🐹', rb: '💎', rs: '🦀', c: '📄', cpp: '📄', cs: '📄',
  json: '🧾', md: '📝', css: '🎨', html: '🌐',
}

function iconFor(name) {
  const ext = name.includes('.') ? name.split('.').pop().toLowerCase() : ''
  return FILE_ICON[ext] || '📄'
}

function TreeNode({ node, depth, openPaths, onToggle, activePath, onOpenFile, onDeleteFile, changedPaths, pendingPaths }) {
  const isOpen = openPaths.has(node.path)
  const isChanged = changedPaths.has(node.path)
  const isPending = pendingPaths.has(node.path)

  if (node.type === 'dir') {
    return (
      <div>
        <div
          className="explorer-row explorer-row-dir"
          style={{ paddingLeft: 8 + depth * 14 }}
          onClick={() => onToggle(node.path)}
        >
          <span className={`explorer-caret ${isOpen ? 'explorer-caret-open' : ''}`}>▸</span>
          <span className="explorer-name">{node.name}</span>
        </div>
        {isOpen &&
          node.children.map((child) => (
            <TreeNode
              key={child.path}
              node={child}
              depth={depth + 1}
              openPaths={openPaths}
              onToggle={onToggle}
              activePath={activePath}
              onOpenFile={onOpenFile}
              onDeleteFile={onDeleteFile}
              changedPaths={changedPaths}
              pendingPaths={pendingPaths}
            />
          ))}
      </div>
    )
  }

  // Pending review (still awaiting Accept/Reject) always wins over a plain
  // "edited earlier this session" highlight -- it's the more urgent state.
  const rowClass = [
    'explorer-row',
    'explorer-row-file',
    activePath === node.path && 'explorer-row-active',
    isPending ? 'explorer-row-pending' : isChanged && 'explorer-row-changed',
  ]
    .filter(Boolean)
    .join(' ')

  return (
    <div
      className={rowClass}
      style={{ paddingLeft: 8 + depth * 14 + 14 }}
      onClick={() => onOpenFile(node.path)}
      title={isPending ? 'awaiting your review' : isChanged ? 'edited this session' : undefined}
    >
      <span className="explorer-icon">{iconFor(node.name)}</span>
      <span className="explorer-name">{node.name}</span>
      {(isPending || isChanged) && (
        <span className={isPending ? 'explorer-pending-dot' : 'explorer-changed-dot'} />
      )}
      {onDeleteFile && (
        <button
          className="explorer-delete-btn"
          title={`delete ${node.name}`}
          onClick={(e) => {
            e.stopPropagation()
            onDeleteFile(node.path)
          }}
        >
          🗑
        </button>
      )}
    </div>
  )
}

export default function FileExplorer({ tree, loading, error, activePath, onOpenFile, onDeleteFile, changedPaths, pendingPaths }) {
  const [openPaths, setOpenPaths] = useState(new Set())

  // Auto-expand every ancestor of whatever file is currently active/changed/
  // pending so a live edit is always visible without hunting for it.
  useEffect(() => {
    if (!tree) return
    setOpenPaths((prev) => {
      const next = new Set(prev)
      const expandAncestors = (path) => {
        const parts = path.split('/')
        parts.pop()
        let acc = ''
        for (const part of parts) {
          acc = acc ? `${acc}/${part}` : part
          next.add(acc)
        }
      }
      if (activePath) expandAncestors(activePath)
      changedPaths.forEach(expandAncestors)
      pendingPaths.forEach(expandAncestors)
      return next
    })
  }, [tree, activePath, changedPaths, pendingPaths])

  const toggle = (path) => {
    setOpenPaths((prev) => {
      const next = new Set(prev)
      if (next.has(path)) next.delete(path)
      else next.add(path)
      return next
    })
  }

  return (
    <aside className="file-explorer">
      <div className="file-explorer-header">Explorer</div>
      <div className="file-explorer-body">
        {loading && <div className="explorer-hint">loading project…</div>}
        {error && <div className="explorer-hint explorer-hint-error">{error}</div>}
        {!loading && !error && tree && tree.children.length === 0 && (
          <div className="explorer-hint">empty directory</div>
        )}
        {!loading &&
          !error &&
          tree?.children.map((child) => (
            <TreeNode
              key={child.path}
              node={child}
              depth={0}
              openPaths={openPaths}
              onToggle={toggle}
              activePath={activePath}
              onOpenFile={onOpenFile}
              onDeleteFile={onDeleteFile}
              changedPaths={changedPaths}
              pendingPaths={pendingPaths}
            />
          ))}
      </div>
    </aside>
  )
}
