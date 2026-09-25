import { useEffect, useMemo, useRef, useState } from 'react'
import './App.css'
import Sidebar from './components/Sidebar.jsx'
import FileExplorer from './components/FileExplorer.jsx'
import EditorTabs from './components/EditorTabs.jsx'
import EditorPane from './components/EditorPane.jsx'
import Message from './components/Message.jsx'
import Composer from './components/Composer.jsx'
import { streamRun, fetchTree, fetchFile, confirmRun, fetchChatHistory, saveChatHistory } from './api.js'

let nextId = 1
let nextRevision = 1
const TAB_LIVE_DOT_MS = 2600

export default function App() {
  const [config, setConfig] = useState({
    mode: 'edit',
    projectDir: '',
    file: '',
    testTarget: '.',
    useJoern: 'off',
  })
  const [messages, setMessages] = useState([])
  const [input, setInput] = useState('')
  const [sending, setSending] = useState(false)
  const logEndRef = useRef(null)

  // ---- VS Code-style file explorer + live-edit view state ----
  // Hidden by default -- this is a chat app first; the code view is an
  // on-demand look at what's being edited, not the default screen. It
  // auto-reveals itself the moment there's something to look at (see
  // applyLiveEdit) and can be hidden again any time.
  const [showCodePanels, setShowCodePanels] = useState(false)
  const [tree, setTree] = useState(null)
  const [treeLoading, setTreeLoading] = useState(false)
  const [treeError, setTreeError] = useState(null)
  // tab: { path, content, diff, revision, isLive, pendingRunId,
  // pendingAssistantId, testsPassed } -- the last four only set while that
  // file has a run paused awaiting human accept/reject.
  const [tabs, setTabs] = useState([])
  const [activeTabPath, setActiveTabPath] = useState(null)
  const [changedPaths, setChangedPaths] = useState(new Set())

  const loadTree = (projectDir) => {
    setTreeLoading(true)
    setTreeError(null)
    fetchTree(projectDir)
      .then((t) => setTree(t))
      .catch((err) => setTreeError(err.message))
      .finally(() => setTreeLoading(false))
  }

  // Sidebar's "Project directory"/"File" inputs update config on every
  // keystroke (needed so a submit always sees the latest typed value), but
  // reacting to *that* directly -- as an earlier version of this file did --
  // fires a tree/file fetch and opens a junk tab for every partial value
  // typed along the way ("S", "Sh", "Sha", ...). Debounce a settled copy
  // and react to that instead; submission still reads the live config.
  const [debouncedProjectDir, setDebouncedProjectDir] = useState(config.projectDir)
  const [debouncedFile, setDebouncedFile] = useState(config.file)

  useEffect(() => {
    const t = setTimeout(() => setDebouncedProjectDir(config.projectDir), 500)
    return () => clearTimeout(t)
  }, [config.projectDir])

  useEffect(() => {
    const t = setTimeout(() => setDebouncedFile(config.file), 500)
    return () => clearTimeout(t)
  }, [config.file])

  // Tracks the project dir this effect last actually ran for, so a
  // project switch and a same-tick "open the file" don't race: without
  // this, both branches would run in the same commit, and the tab-open
  // one would decide "already open?" from the `tabs` closure captured
  // *before* the reset below had applied -- silently leaving
  // activeTabPath pointing at a path with no matching tab. Reachable in
  // the ordinary case too, since changing "Project directory" in the
  // Sidebar does not clear the "File" field.
  // null, not debouncedProjectDir -- seeding it with the current value
  // would make the very first effect run see "unchanged" and skip
  // loadTree/tab setup entirely, leaving the explorer blank until the
  // user manually changed the project dir. null can never equal a real
  // project dir, so the first run always counts as a change.
  const prevProjectDirRef = useRef(null)

  // Simple resume (not a multi-session browser): one continuous chat log
  // per project. Tracks which project's history has actually finished
  // loading so the auto-save effect below can't race the fetch below --
  // without this, the setMessages([]) reset a few lines down would itself
  // trigger a debounced save of an *empty* history for the new project,
  // potentially overwriting real saved history if that save's timer fires
  // before the fetch resolves.
  const historyLoadedForRef = useRef(null)

  useEffect(() => {
    const projectChanged = prevProjectDirRef.current !== debouncedProjectDir
    prevProjectDirRef.current = debouncedProjectDir

    if (projectChanged) {
      setChangedPaths(new Set())
      setMessages([])
      historyLoadedForRef.current = null
      // No project directory chosen yet (the app's own starting state,
      // not a leftover default) -- nothing to fetch, and fetchTree('')
      // against the server's own cwd would be a real, misleading call.
      if (!debouncedProjectDir) {
        setTree(null)
        setTabs([])
        setActiveTabPath(null)
        return
      }
      loadTree(debouncedProjectDir)
      fetchChatHistory(debouncedProjectDir)
        .then(({ messages: loaded }) => {
          if (loaded?.length) {
            setMessages(loaded)
            nextId = Math.max(0, ...loaded.map((m) => m.id || 0)) + 1
          }
        })
        .catch(() => {})
        .finally(() => {
          historyLoadedForRef.current = debouncedProjectDir
        })
      if (config.mode !== 'find' && debouncedFile) {
        const path = debouncedFile
        setTabs([{ path, content: '', diff: null, revision: 0, isLive: false, pendingRunId: null }])
        setActiveTabPath(path)
        fetchFile(debouncedProjectDir, path)
          .then(({ content }) => setTabs((prev) => prev.map((t) => (t.path === path ? { ...t, content } : t))))
          .catch((err) =>
            setTabs((prev) => prev.map((t) => (t.path === path ? { ...t, content: `// failed to load: ${err.message}` } : t))),
          )
      } else {
        setTabs([])
        setActiveTabPath(null)
      }
      return
    }

    // Project unchanged -- just the (settled) file or mode changed, so
    // there's no competing reset in this tick and the normal openFile path
    // (which reads the current `tabs` closure to skip a redundant fetch)
    // is safe.
    if (config.mode !== 'find' && debouncedFile) openFile(debouncedFile)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [debouncedProjectDir, debouncedFile, config.mode])

  // Auto-saves the chat log ~800ms after the last change (coalesces a
  // run's many rapid step updates into one write instead of one per
  // event) -- gated on historyLoadedForRef so this can't fire with an
  // empty array before this project's real history has even finished
  // loading (see the load effect above).
  useEffect(() => {
    if (historyLoadedForRef.current !== config.projectDir) return
    const t = setTimeout(() => {
      saveChatHistory(config.projectDir, messages)
    }, 800)
    return () => clearTimeout(t)
  }, [messages, config.projectDir])

  // Explorer clicks go through this instead of bare openFile -- syncs
  // config.file (the actual edit target every request is sent with) to
  // whatever file you just clicked, so opening a file and typing a
  // request edits *that* file without a separate manual step. Edit mode
  // only: in create mode config.file names the brand-new file being
  // written, not something to point at an existing one (would make the
  // next create request collide with it); find mode never reads
  // config.file at all.
  const handleExplorerFileClick = (path) => {
    openFile(path)
    if (config.mode === 'edit') setConfig((prev) => ({ ...prev, file: path }))
  }

  const openFile = (path, opts = {}) => {
    setActiveTabPath(path)
    const alreadyOpen = tabs.some((t) => t.path === path)
    if (!alreadyOpen) {
      setTabs((prev) => [
        ...prev,
        { path, content: opts.content ?? '', diff: null, revision: 0, isLive: false, pendingRunId: null },
      ])
    }
    if (!alreadyOpen && !opts.content) {
      fetchFile(config.projectDir, path)
        .then(({ content }) => {
          setTabs((prev) => prev.map((t) => (t.path === path ? { ...t, content } : t)))
        })
        .catch((err) => {
          setTabs((prev) => prev.map((t) => (t.path === path ? { ...t, content: `// failed to load: ${err.message}` } : t)))
        })
    }
  }

  // Re-fetches an ALREADY-open tab's content from disk, unconditionally
  // (unlike openFile, which only fetches once on first open). Real bug
  // this closes: a tab opened before its file existed (e.g. typing a
  // not-yet-created "data.db" into the File field before submitting a
  // create) shows a real, correctly-reported "failed to load: ... 404"
  // placeholder at that moment -- but nothing ever refreshed it after
  // the run went on to create that exact file, so the tab kept showing
  // the stale not-found error even though the real file now existed on
  // disk. Also the only way a binary artifact's tab (data.db, .xlsx --
  // no live preview/diff event fires for those, see run_pipeline.py's
  // on_preview skip) ever reflects its real post-run state at all.
  const refreshOpenTab = (path) => {
    if (!tabs.some((t) => t.path === path)) return
    fetchFile(config.projectDir, path)
      .then(({ content }) => {
        setTabs((prev) => prev.map((t) => (t.path === path ? { ...t, content } : t)))
      })
      .catch((err) => {
        setTabs((prev) => prev.map((t) => (t.path === path ? { ...t, content: `// failed to load: ${err.message}` } : t)))
      })
  }

  const closeTab = (path) => {
    setTabs((prev) => {
      const next = prev.filter((t) => t.path !== path)
      if (activeTabPath === path) {
        setActiveTabPath(next.length ? next[next.length - 1].path : null)
      }
      return next
    })
  }

  // Fires whenever a preview/done event carries an actual code change --
  // opens (or updates) that file's tab with the new content + diff and
  // bumps `revision` so EditorPane re-arms its live-edit animation even if
  // the same file gets edited twice in a row with an identical diff.
  const applyLiveEdit = ({ file, diff, new_file_content }) => {
    if (!file || new_file_content === undefined) return
    const revision = nextRevision++
    setShowCodePanels(true)
    setTabs((prev) => {
      const existing = prev.find((t) => t.path === file)
      const entry = { path: file, content: new_file_content, diff: diff || null, revision, isLive: true, pendingRunId: null }
      return existing ? prev.map((t) => (t.path === file ? entry : t)) : [...prev, entry]
    })
    setActiveTabPath(file)
    setTimeout(() => {
      setTabs((prev) => prev.map((t) => (t.path === file && t.revision === revision ? { ...t, isLive: false } : t)))
    }, TAB_LIVE_DOT_MS)
  }

  // Fires when a run comes back with result.status === "awaiting_confirmation"
  // -- the backend already applied + tested the change in a temp copy but
  // deliberately did not touch the real file(s) yet. Attaches everything
  // the editor pane's Accept/Reject bar needs onto every touched file's
  // tab -- a multi-file create (metadata.files, e.g. "frontend/index.html"
  // + "backend/server.py" from one request) resolves as one run_id no
  // matter which of its tabs you click Accept/Reject from, so every one
  // of them needs the same pendingRunId, not just the first.
  const markPendingConfirmation = (files, { runId, assistantId, testsPassed, isDelete }) => {
    const paths = (Array.isArray(files) ? files : [files]).filter(Boolean)
    if (!paths.length || !runId) return
    setTabs((prev) =>
      prev.map((t) =>
        paths.includes(t.path) ? { ...t, pendingRunId: runId, pendingAssistantId: assistantId, testsPassed, pendingDelete: !!isDelete } : t,
      ),
    )
  }

  // Guards against a double-click firing this twice before React re-renders
  // with confirming:true (the disabled-button state) -- a synchronous ref
  // check, not state, since state wouldn't be visible until the next render.
  // Keyed by run_id, not path: a multi-file create (e.g. "make a .env file
  // that links to this chatbot" -> .env + chatbot.py, one run_id) shares
  // one pendingRunId across every tab it touched, so clicking Accept on
  // *any* of them must resolve -- and lock -- all of them together.
  const confirmingRunsRef = useRef(new Set())

  // Shared core for both Accept/Reject entry points: the editor tab's
  // confirm bar (handleConfirmTab, below) and the chat message's own
  // confirm bar (Message.jsx's onConfirmRun) -- the latter exists
  // because a pure folder create or a binary artifact (.xlsx/.db) has
  // no tab at all to attach a confirm bar to (nothing to fetch/preview
  // as text), so relying only on the tab-based path left those runs
  // stuck at "awaiting_confirmation" with no visible way to accept or
  // reject them. relatedPaths naturally comes back empty when called
  // with no matching tabs -- every tab-touching step below is then a
  // harmless no-op, and the actual confirm/reject call still goes through.
  const handleConfirmRun = async (runId, assistantId, accept) => {
    if (!runId || confirmingRunsRef.current.has(runId)) return
    confirmingRunsRef.current.add(runId)
    const relatedPaths = tabs.filter((t) => t.pendingRunId === runId).map((t) => t.path)
    const pendingDelete = tabs.some((t) => t.pendingRunId === runId && t.pendingDelete)
    setTabs((prev) => prev.map((t) => (relatedPaths.includes(t.path) ? { ...t, confirming: true } : t)))
    patchMessage(assistantId, { confirming: true, confirmError: null })
    try {
      const data = await confirmRun(runId, accept)
      setMessages((prev) =>
        prev.map((m) =>
          m.id === assistantId
            ? { ...m, metadata: { ...m.metadata, result: data.result, new_version: data.new_version }, confirming: false }
            : m,
        ),
      )
      if (accept) {
        loadTree(config.projectDir)
        if (data.result.status === 'success') {
          if (pendingDelete) {
            // The file no longer exists -- nothing left for this tab to
            // show, unlike an edit/create where the new content stays.
            relatedPaths.forEach(closeTab)
          } else {
            setChangedPaths((prev) => {
              const next = new Set(prev)
              relatedPaths.forEach((p) => next.add(p))
              return next
            })
            // Same stale-tab gap as the direct onDone success path above:
            // a binary artifact's tab never gets a live-preview event, so
            // this accept round-trip is the only place its real content
            // (or, for a real .db/.xlsx, the "binary file, cannot
            // display" it should show) ever gets loaded.
            relatedPaths.forEach(refreshOpenTab)
            // The web UI's require_confirmation is always on for create
            // (unlike the CLI), so a create's own file is only actually
            // written here, on accept -- not in the direct onDone success
            // branch above, which a create never reaches. Real bug this
            // closes: after creating a file, the app stayed in create
            // mode, so a natural follow-up request ("add opencv in it")
            // was refused by create mode's own "already exists" guard
            // instead of editing what was just written. Switch to edit
            // mode targeting the new file -- only for a single-file
            // create, since a multi-file create has no one obvious next
            // edit target to guess.
            if (config.mode === 'create' && relatedPaths.length === 1) {
              setConfig((prev) => ({ ...prev, mode: 'edit', file: relatedPaths[0] }))
            }
          }
        }
        if (!pendingDelete) {
          setTabs((prev) => prev.map((t) => (relatedPaths.includes(t.path) ? { ...t, pendingRunId: null, confirming: false } : t)))
        }
      } else {
        // Nothing was ever written to disk -- reload each touched tab's
        // real (unchanged) content so none of them keep showing the
        // rejected, never-committed edit.
        for (const relatedPath of relatedPaths) {
          try {
            const { content } = await fetchFile(config.projectDir, relatedPath)
            setTabs((prev) =>
              prev.map((t) => (t.path === relatedPath ? { ...t, content, diff: null, pendingRunId: null, confirming: false } : t)),
            )
          } catch {
            // create-mode reject: this file never existed on disk --
            // nothing to reload, so there's nothing left this tab should
            // keep showing.
            closeTab(relatedPath)
          }
        }
      }
    } catch (err) {
      setTabs((prev) => prev.map((t) => (relatedPaths.includes(t.path) ? { ...t, confirming: false, confirmError: err.message } : t)))
      patchMessage(assistantId, { confirming: false, confirmError: err.message })
    } finally {
      confirmingRunsRef.current.delete(runId)
    }
  }

  const handleConfirmTab = (path, accept) => {
    const tab = tabs.find((t) => t.path === path)
    if (tab?.pendingRunId) handleConfirmRun(tab.pendingRunId, tab.pendingAssistantId, accept)
  }

  const scrollToEnd = () => {
    requestAnimationFrame(() => logEndRef.current?.scrollIntoView({ behavior: 'smooth' }))
  }

  const patchMessage = (id, patch) => {
    setMessages((prev) => prev.map((m) => (m.id === id ? { ...m, ...patch } : m)))
  }

  const appendStep = (id, tag, msg) => {
    setMessages((prev) =>
      prev.map((m) => (m.id === id ? { ...m, steps: [...m.steps, { tag, msg }] } : m)),
    )
    scrollToEnd()
  }

  // VS Code-style Explorer delete: click a row's trash icon instead of
  // typing "delete this file" in the composer -- goes through the exact
  // same backend pipeline (zero LLM cost, blocked if it'd break the test
  // suite, reviewed via the same red confirm-bar before anything is
  // actually removed), just triggered by a click. Logged as its own
  // chat turn so that review has somewhere to render.
  const handleExplorerDelete = async (path) => {
    if (sending) return
    if (!window.confirm(`Delete "${path}"?`)) return

    const userId = nextId++
    const assistantId = nextId++
    setMessages((prev) => [
      ...prev,
      { id: userId, role: 'user', text: `Delete ${path}` },
      { id: assistantId, role: 'assistant', steps: [], preview: null, metadata: null, error: null },
    ])
    setSending(true)
    scrollToEnd()

    const payload = {
      mode: 'edit',
      project_dir: config.projectDir,
      file: path,
      request: 'delete this file',
      test_target: config.testTarget,
      require_confirmation: true,
    }

    try {
      await streamRun(payload, {
        onStep: (tag, msg) => appendStep(assistantId, tag, msg),
        onPreview: () => {},
        onDone: (metadata) => {
          patchMessage(assistantId, { metadata })
          if (metadata?.result?.status === 'awaiting_confirmation') {
            if (metadata.file) openFile(metadata.file)
            markPendingConfirmation(metadata.file, {
              runId: metadata.run_id,
              assistantId,
              testsPassed: metadata.validation?.tests_passed,
              isDelete: true,
            })
          } else if (metadata?.result?.status === 'success') {
            loadTree(config.projectDir)
          }
          scrollToEnd()
        },
        onError: (message) => {
          patchMessage(assistantId, { error: message })
          scrollToEnd()
        },
      })
    } catch (err) {
      patchMessage(assistantId, { error: `request failed: ${err.message}` })
    } finally {
      setSending(false)
    }
  }

  const handleSend = async () => {
    const text = input.trim()
    if (!text || sending) return

    // No hardcoded fallback project dir exists anymore -- a request sent
    // before one is chosen would otherwise resolve server-side against
    // whatever the server's own cwd happens to be, silently editing the
    // wrong thing. Refuse it here instead, same as any other invalid
    // request, rather than letting it reach the backend at all.
    if (!config.projectDir.trim()) {
      setMessages((prev) => [
        ...prev,
        { id: nextId++, role: 'user', text },
        {
          id: nextId++,
          role: 'assistant',
          steps: [],
          preview: null,
          metadata: null,
          error: 'Set a project directory (left sidebar) before sending a request.',
        },
      ])
      setInput('')
      scrollToEnd()
      return
    }

    const userId = nextId++
    const assistantId = nextId++
    setMessages((prev) => [
      ...prev,
      { id: userId, role: 'user', text },
      { id: assistantId, role: 'assistant', steps: [], preview: null, metadata: null, error: null },
    ])
    setInput('')
    setSending(true)
    scrollToEnd()

    const payload = {
      mode: config.mode,
      project_dir: config.projectDir,
      file: config.mode === 'find' ? null : config.file || null,
      request: text,
      test_target: config.mode === 'edit' ? config.testTarget : null,
      use_joern: config.mode === 'find' ? config.useJoern : 'off',
      require_confirmation: config.mode !== 'find',
    }

    try {
      await streamRun(payload, {
        onStep: (tag, msg) => appendStep(assistantId, tag, msg),
        onPreview: (preview) => {
          patchMessage(assistantId, { preview })
          applyLiveEdit(preview)
          scrollToEnd()
        },
        onDone: (metadata) => {
          patchMessage(assistantId, { metadata })
          if (metadata?.result?.status === 'awaiting_confirmation') {
            const isDelete = metadata.strategy === 'FILE_DELETE'
            if (isDelete && metadata.file) openFile(metadata.file)
            markPendingConfirmation(metadata.files || metadata.file, {
              runId: metadata.run_id,
              assistantId,
              testsPassed: metadata.validation?.tests_passed,
              isDelete,
            })
          } else if (metadata?.result?.status === 'success' && config.mode !== 'find') {
            loadTree(config.projectDir)
            if (!metadata.result.no_op) {
              const touched = metadata.files || (metadata.file ? [metadata.file] : [])
              setChangedPaths((prev) => {
                const next = new Set(prev)
                touched.forEach((p) => next.add(p))
                return next
              })
              // Any of those files already open in a tab needs its real,
              // current content -- a binary artifact (data.db, .xlsx)
              // never fires the live-preview/diff event above at all
              // (run_pipeline.py deliberately skips it -- no meaningful
              // diff for real binary bytes), so this is the only refresh
              // its tab would otherwise ever get.
              touched.forEach(refreshOpenTab)
              // A successful create's file now exists on disk -- switch to
              // edit mode targeting it so a natural follow-up ("add opencv
              // in it") edits what was just created instead of being
              // refused by create mode's own "already exists" guard. Only
              // when exactly one file was created: a multi-file create
              // ("make a .env that links to this chatbot") has no single
              // obvious next edit target, so mode is left alone rather
              // than guessing which of several files to point at.
              if (config.mode === 'create' && touched.length === 1) {
                setConfig((prev) => ({ ...prev, mode: 'edit', file: touched[0] }))
              }
            }
          }
          scrollToEnd()
        },
        onNeedsSelection: (metadata) => {
          patchMessage(assistantId, { metadata })
          scrollToEnd()
        },
        onError: (message) => {
          patchMessage(assistantId, { error: message })
          scrollToEnd()
        },
      })
    } catch (err) {
      patchMessage(assistantId, { error: `request failed: ${err.message}` })
    } finally {
      setSending(false)
    }
  }

  // Fires when the user picks one candidate off a needs_selection picker
  // (a destructive "remove/delete X" request that matched more than one
  // real symbol). Resubmits the same request with that exact symbol
  // confirmed -- skips localization and the LLM call entirely, since the
  // DELETE's shape is fully determined once the target is known.
  const handleSelectCandidate = async (assistantId, candidate, priorMetadata) => {
    patchMessage(assistantId, { selectionPending: true })
    appendStep(assistantId, 'CONFIRM', `confirmed target: ${candidate.name} (${candidate.symbol_type})`)
    setSending(true)

    const payload = {
      mode: 'edit',
      project_dir: config.projectDir,
      file: priorMetadata.file,
      request: priorMetadata.user_request,
      test_target: config.testTarget,
      confirm_symbol: candidate.name,
      confirm_symbol_type: candidate.symbol_type,
      confirm_symbol_line: candidate.start_line,
      require_confirmation: true,
    }

    try {
      await streamRun(payload, {
        onStep: (tag, msg) => appendStep(assistantId, tag, msg),
        onPreview: (preview) => {
          patchMessage(assistantId, { preview })
          applyLiveEdit(preview)
          scrollToEnd()
        },
        onDone: (metadata) => {
          patchMessage(assistantId, { metadata, selectionPending: false })
          if (metadata?.result?.status === 'awaiting_confirmation') {
            markPendingConfirmation(metadata.file, {
              runId: metadata.run_id,
              assistantId,
              testsPassed: metadata.validation?.tests_passed,
            })
          } else if (metadata?.result?.status === 'success') {
            loadTree(config.projectDir)
          }
          scrollToEnd()
        },
        onError: (message) => {
          patchMessage(assistantId, { error: message, selectionPending: false })
          scrollToEnd()
        },
      })
    } catch (err) {
      patchMessage(assistantId, { error: `request failed: ${err.message}`, selectionPending: false })
    } finally {
      setSending(false)
    }
  }

  const handleNewChat = () => {
    if (sending) return
    setMessages([])
  }

  const activeTab = tabs.find((t) => t.path === activeTabPath) || null
  // Keyed on the joined path list (not `tabs` itself, which changes identity
  // on every unrelated tab update like `confirming`) so FileExplorer's
  // ancestor-expand effect only re-runs when membership actually changes.
  const pendingPathsKey = tabs
    .filter((t) => t.pendingRunId)
    .map((t) => t.path)
    .sort()
    .join('\n')
  const pendingPaths = useMemo(() => new Set(pendingPathsKey ? pendingPathsKey.split('\n') : []), [pendingPathsKey])

  return (
    <div className="layout">
      <Sidebar config={config} onConfigChange={setConfig} onNewChat={handleNewChat} />

      {showCodePanels && (
        <>
          <FileExplorer
            tree={tree}
            loading={treeLoading}
            error={treeError}
            activePath={activeTabPath}
            onOpenFile={handleExplorerFileClick}
            onDeleteFile={handleExplorerDelete}
            changedPaths={changedPaths}
            pendingPaths={pendingPaths}
          />

          <div className="workspace">
            <EditorTabs tabs={tabs} activePath={activeTabPath} onSelect={setActiveTabPath} onClose={closeTab} />
            <EditorPane
              tab={activeTab}
              onAccept={() => handleConfirmTab(activeTab.path, true)}
              onReject={() => handleConfirmTab(activeTab.path, false)}
            />
          </div>
        </>
      )}

      <main className="main">
        <div className="main-topbar">
          <button className="code-toggle-btn" onClick={() => setShowCodePanels((v) => !v)}>
            {showCodePanels ? '‹‹ hide code' : '›› show code'}
          </button>
        </div>
        <div className="log">
          {messages.length === 0 ? (
            <div className="empty-state">
              <div className="empty-title">Incremental Editing Engine</div>
              <div className="empty-sub">
                Describe a change to “{config.file}” in “{config.projectDir}”, or switch to create mode
                for a brand-new file.
              </div>
            </div>
          ) : (
            messages.map((m) => (
              <Message key={m.id} message={m} onSelectCandidate={handleSelectCandidate} onConfirmRun={handleConfirmRun} />
            ))
          )}
          <div ref={logEndRef} />
        </div>

        {config.mode === 'edit' && config.file && (
          <div className="target-file-chip-row">
            <span className="target-file-chip">
              <span className="target-file-chip-icon">📄</span>
              {config.file}
              <button
                className="target-file-chip-close"
                title="stop targeting this file -- next request auto-locates instead"
                onClick={() => setConfig((prev) => ({ ...prev, file: '' }))}
              >
                ×
              </button>
            </span>
          </div>
        )}

        <Composer
          value={input}
          onChange={setInput}
          onSend={handleSend}
          disabled={sending}
          placeholder={
            config.mode === 'edit'
              ? 'e.g. add validation for the email field'
              : config.mode === 'find'
                ? 'e.g. compares two ML models and shows strengths/weaknesses'
                : 'e.g. implement is_palindrome(s) and reverse_words(s)'
          }
        />
      </main>
    </div>
  )
}
