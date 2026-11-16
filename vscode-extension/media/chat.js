// Sidebar chat webview. Plain DOM, no framework: the extension host does all
// network I/O and relays engine events here via postMessage.
;(function () {
  const vscode = acquireVsCodeApi()

  // ---- small DOM helpers ---------------------------------------------------

  const SVG_NS = 'http://www.w3.org/2000/svg'

  function h(tag, props, ...children) {
    const el = document.createElement(tag)
    for (const [k, v] of Object.entries(props || {})) {
      if (v === undefined || v === null || v === false) continue
      if (k === 'class') el.className = v
      else if (k.startsWith('on')) el.addEventListener(k.slice(2).toLowerCase(), v)
      else el.setAttribute(k, v === true ? '' : v)
    }
    append(el, children)
    return el
  }

  function append(el, children) {
    for (const c of children.flat()) {
      if (c === undefined || c === null || c === false) continue
      el.appendChild(typeof c === 'string' || typeof c === 'number' ? document.createTextNode(String(c)) : c)
    }
  }

  // Icons are static path data authored here, never user input.
  function icon(paths, opts = {}) {
    const svg = document.createElementNS(SVG_NS, 'svg')
    svg.setAttribute('viewBox', opts.viewBox || '0 0 24 24')
    svg.setAttribute('fill', opts.fill ? 'currentColor' : 'none')
    svg.setAttribute('stroke', opts.fill ? 'none' : 'currentColor')
    svg.setAttribute('stroke-width', '1.7')
    svg.setAttribute('stroke-linecap', 'round')
    svg.setAttribute('stroke-linejoin', 'round')
    svg.setAttribute('aria-hidden', 'true')
    for (const d of [].concat(paths)) {
      const p = document.createElementNS(SVG_NS, 'path')
      p.setAttribute('d', d)
      svg.appendChild(p)
    }
    return svg
  }

  const ICONS = {
    history: () => icon(['M3 12a9 9 0 1 0 3-6.7', 'M3 4v4h4', 'M12 7.5V12l3 2']),
    newChat: () => icon(['M21 12a8.5 8.5 0 0 1-12.4 7.6L4 21l1.4-4.6A8.5 8.5 0 1 1 21 12Z', 'M12 8.5v7', 'M8.5 12h7']),
    plus: () => icon(['M12 5v14', 'M5 12h14']),
    slash: () => icon(['M4 4h16v16H4z', 'M14.5 7.5l-5 9']),
    zap: () => icon(['M13 2 4 14h7l-1 8 9-12h-7l1-8Z']),
    hand: () => icon(['M8 13V5.5a1.5 1.5 0 0 1 3 0V12', 'M11 11V4.5a1.5 1.5 0 0 1 3 0V12', 'M14 11.5V6a1.5 1.5 0 0 1 3 0v8a7 7 0 0 1-7 7h-.5a6 6 0 0 1-4.6-2.2L3 16.3a1.6 1.6 0 0 1 2.4-2L8 16']),
    send: () => icon(['M12 19V5', 'M6 11l6-6 6 6']),
    stop: () => icon('M7 7h10v10H7z', { fill: true }),
    bolt: () => icon('M13.5 1.5 3.8 13.6h6.9L9.6 22.5l10.6-13h-7.1l.4-8Z', { fill: true }),
    close: () => icon(['M6 6l12 12', 'M18 6 6 18']),
    arrow: () => icon(['M5 12h14', 'M13 6l6 6-6 6']),
    play: () => icon('M8 5.5v13l10.5-6.5z', { fill: true }),
    check: () => icon('M5 12.5l4.5 4.5L19 7.5'),
  }

  // Original pixel mascot: a little screen-faced bot. '#' body, 's' screen,
  // 'e' eye, 'a' antenna tip.
  const MASCOT = [
    '......a......',
    '......#......',
    '.###########.',
    '.#sssssssss#.',
    '##seesssees##',
    '##seesssees##',
    '.#sssssssss#.',
    '.###########.',
    '...##...##...',
    '..###...###..',
  ]

  function mascot() {
    const svg = document.createElementNS(SVG_NS, 'svg')
    svg.setAttribute('viewBox', `0 0 ${MASCOT[0].length} ${MASCOT.length}`)
    svg.setAttribute('class', 'mascot')
    svg.setAttribute('shape-rendering', 'crispEdges')
    svg.setAttribute('aria-hidden', 'true')
    // One path per color, so scaled rendering shows no seams between pixels.
    const fills = { '#': 'var(--accent)', a: 'var(--accent)', s: 'var(--bg)', e: 'var(--fg)' }
    const paths = {}
    MASCOT.forEach((row, y) => {
      ;[...row].forEach((ch, x) => {
        if (fills[ch]) paths[ch] = (paths[ch] || '') + `M${x} ${y}h1v1h-1z`
      })
    })
    for (const [ch, d] of Object.entries(paths)) {
      const p = document.createElementNS(SVG_NS, 'path')
      p.setAttribute('d', d)
      p.setAttribute('fill', fills[ch])
      if (ch === 'e') p.setAttribute('class', 'blink')
      svg.appendChild(p)
    }
    return svg
  }

  // ---- state ---------------------------------------------------------------

  const MODES = {
    edit: { label: 'Edit', desc: 'Change existing code, one symbol at a time' },
    create: { label: 'Create', desc: 'Write a new file' },
    find: { label: 'Find', desc: 'Locate code across the repo, no edits' },
  }

  const TIPS = [
    ['Point at a function. Only ', 'that', ' gets sent to the model.'],
    ['Type ', '/', ' for commands: edit, create, find.'],
    ['Edits land as ', 'REPLACE / INSERT / DELETE', ' on named symbols. Never a whole-file rewrite.'],
    ['Nothing touches disk until you accept the diff.'],
  ]

  const SUGGESTIONS = {
    edit: ['Add input validation to the current function', 'Handle division by zero in divide', 'Rename get_user to fetch_user'],
    create: ['Write utils/slug.py with slugify(text)', 'Add a pytest file for the calculator'],
    find: ['Where is the auth token checked?', 'What calls save_order?'],
  }

  const uid = () => Math.random().toString(36).slice(2, 10)

  function freshSession() {
    return { id: uid(), title: '', updatedAt: Date.now(), turns: [] }
  }

  const saved = vscode.getState() || {}
  const state = {
    session: saved.session || freshSession(),
    mode: saved.mode || 'edit',
    autoAccept: !!saved.autoAccept,
    useActiveFile: saved.useActiveFile !== false,
    fileOverride: saved.fileOverride || null,
    activeFile: null,
    model: saved.model || null,
    serverOk: null,
    tip: TIPS[Math.floor(Math.random() * TIPS.length)],
  }

  // A turn left "running" by a reload never got its terminal event.
  for (const t of state.session.turns) {
    if (t.status === 'running') {
      t.status = 'error'
      t.error = t.error || 'interrupted (panel reloaded)'
    }
  }

  function persist() {
    vscode.setState({
      session: state.session,
      mode: state.mode,
      autoAccept: state.autoAccept,
      useActiveFile: state.useActiveFile,
      fileOverride: state.fileOverride,
      model: state.model,
    })
  }

  function saveSession() {
    state.session.updatedAt = Date.now()
    persist()
    vscode.postMessage({ type: 'saveSession', session: state.session })
  }

  const findTurn = (id) => state.session.turns.find((t) => t.id === id)
  const runningTurn = () => state.session.turns.find((t) => t.status === 'running')

  // ---- layout (built once) ---------------------------------------------------

  const app = document.getElementById('app')
  const titleEl = h('div', { class: 'header-title' })
  const header = h(
    'div',
    { class: 'header' },
    h('span', { class: 'brand-mark', 'aria-hidden': 'true' }, ICONS.bolt()),
    titleEl,
    h('button', { class: 'icon-btn', title: 'Past conversations', 'aria-label': 'Past conversations', onClick: () => vscode.postMessage({ type: 'history' }) }, ICONS.history()),
    h('button', { class: 'icon-btn', title: 'New conversation', 'aria-label': 'New conversation', onClick: newChat }, ICONS.newChat())
  )
  const scroll = h('div', { class: 'scroll' })

  const contextRow = h('div', { class: 'context-row' })
  const textarea = h('textarea', {
    rows: 1,
    placeholder: navigator.platform.includes('Mac') ? '⌘ Esc to focus or unfocus' : 'Ctrl+Esc to focus or unfocus',
    'aria-label': 'Message',
  })
  const modeTabs = h('div', { class: 'segmented', role: 'tablist', 'aria-label': 'Mode' })
  const modelLabel = h('span', { class: 'model', title: 'Model used by the engine' })
  const autoBtn = h('button', { class: 'auto-toggle', onClick: toggleAuto })
  const sendBtn = h('button', { class: 'send', 'aria-label': 'Send', onClick: onSendClick })
  const toolbar = h(
    'div',
    { class: 'toolbar' },
    h('button', { class: 'icon-btn', title: 'Choose target file', 'aria-label': 'Choose target file', onClick: () => vscode.postMessage({ type: 'pickFile' }) }, ICONS.plus()),
    h('button', { class: 'icon-btn', title: 'Commands', 'aria-label': 'Commands', onClick: openSlashFromButton }, ICONS.slash()),
    modelLabel,
    h('span', { class: 'grow' }),
    autoBtn,
    sendBtn
  )
  const composer = h('div', { class: 'composer' }, h('div', { class: 'composer-top' }, modeTabs, contextRow), textarea, toolbar)
  const hintBar = h('div', { class: 'hint-bar' })
  append(app, [header, scroll, h('div', { class: 'composer-wrap' }, composer, hintBar)])

  // ---- rendering -----------------------------------------------------------

  function renderHeader() {
    titleEl.textContent = state.session.title || 'Untitled'
    titleEl.title = titleEl.textContent
  }

  function targetFile() {
    if (state.mode === 'find') return null
    if (state.fileOverride) return state.fileOverride
    if (state.mode === 'edit' && state.useActiveFile) return state.activeFile
    return null
  }

  function renderComposer() {
    contextRow.replaceChildren()
    if (state.mode !== 'find') {
      if (state.fileOverride) {
        contextRow.appendChild(
          h('span', { class: 'chip', title: state.fileOverride },
            h('span', { class: 'name' }, state.fileOverride),
            h('button', { title: 'Clear', 'aria-label': 'Clear target file', onClick: () => { state.fileOverride = null; persist(); renderComposer() } }, '×'))
        )
      } else if (state.mode === 'edit' && state.activeFile) {
        contextRow.appendChild(
          h('span', { class: 'chip' + (state.useActiveFile ? '' : ' off'), title: state.useActiveFile ? 'Target: current file (click to auto-locate instead)' : 'Auto-locate across the repo (click to target current file)' },
            h('button', { class: 'name', onClick: () => { state.useActiveFile = !state.useActiveFile; persist(); renderComposer() } }, state.activeFile))
        )
      }
      if (!targetFile()) {
        contextRow.appendChild(h('span', { class: 'chip off', title: 'The engine picks the file' }, state.mode === 'create' ? 'new file: engine picks a path' : 'auto-locate'))
      }
    }
    contextRow.hidden = contextRow.childElementCount === 0

    modeTabs.replaceChildren(
      ...Object.entries(MODES).map(([k, v]) =>
        h('button', { role: 'tab', 'aria-selected': String(k === state.mode), class: k === state.mode ? 'on' : '', title: v.desc, onClick: () => setMode(k) }, v.label))
    )
    modelLabel.textContent = state.model || ''
    modelLabel.hidden = !state.model

    autoBtn.replaceChildren(h('span', { class: 'switch', 'aria-hidden': 'true' }), state.autoAccept ? 'Auto-apply' : 'Review')
    autoBtn.setAttribute('role', 'switch')
    autoBtn.setAttribute('aria-checked', String(state.autoAccept))
    autoBtn.className = 'auto-toggle' + (state.autoAccept ? ' on' : '')
    autoBtn.title = state.autoAccept
      ? 'Auto: validated changes are written immediately'
      : 'Ask: review each diff and accept before it is written'
    autoBtn.hidden = state.mode === 'find'

    const busy = !!runningTurn()
    sendBtn.replaceChildren(busy ? ICONS.stop() : ICONS.send())
    sendBtn.className = 'send' + (busy ? ' stop' : '')
    sendBtn.disabled = !busy && !textarea.value.trim()
    sendBtn.setAttribute('aria-label', busy ? 'Stop' : 'Send')
    sendBtn.title = busy ? 'Stop' : 'Send (Enter)'

    hintBar.textContent = state.serverOk === false ? 'Engine server is not running. Check the output channel.' : ''
    hintBar.hidden = !hintBar.textContent
  }

  function renderEmpty() {
    const tip = h('div', { class: 'tip' })
    state.tip.forEach((part, i) => tip.appendChild(i % 2 ? h('code', {}, part) : document.createTextNode(part)))
    const suggestions = (SUGGESTIONS[state.mode] || []).map((sg) =>
      h('button', { class: 'suggestion', onClick: () => { textarea.value = sg; autosize(); renderComposer(); textarea.focus() } },
        h('span', { class: 'suggestion-text' }, sg), ICONS.arrow()))
    return h(
      'div',
      { class: 'empty' },
      h('div', { class: 'hero' },
        h('div', { class: 'mascot-wrap' }, mascot()),
        h('h1', { class: 'hero-title' }, 'What should change?'),
        tip),
      h('div', { class: 'suggestions' }, suggestions),
      h('button', { class: 'link-btn how', onClick: () => vscode.postMessage({ type: 'walkthrough' }) }, ICONS.play(), 'See how it works')
    )
  }

  function renderDiff(diff) {
    const pre = h('pre', { class: 'diff' })
    for (const line of diff.replace(/\n$/, '').split('\n')) {
      if (line.startsWith('+++') || line.startsWith('---')) continue
      const cls = line.startsWith('@@') ? 'hunk' : line.startsWith('+') ? 'add' : line.startsWith('-') ? 'del' : ''
      pre.appendChild(h('span', { class: cls }, line || ' '))
    }
    return pre
  }

  function stats(meta) {
    const g = meta?.generation
    if (!g) return null
    const parts = []
    if (meta.strategy) parts.push(meta.strategy)
    if (g.total_tokens) parts.push(`${g.total_tokens.toLocaleString()} tok`)
    if (typeof g.estimated_cost_usd === 'number') parts.push(`$${g.estimated_cost_usd.toFixed(4)}`)
    if (g.latency_ms) parts.push(`${(g.latency_ms / 1000).toFixed(1)}s`)
    const c = meta.context
    if (c?.context_lines && c?.total_lines) parts.push(`ctx ${c.context_lines}/${c.total_lines} lines`)
    return parts.length ? h('div', { class: 'stats' }, parts.map((p) => h('span', {}, p))) : null
  }

  function previewCard(t) {
    const p = t.preview
    if (!p || (!p.diff && !p.new_file_content)) return null
    const status = t.metadata?.result?.status
    const awaiting = status === 'awaiting_confirmation' && !t.confirmed
    const card = h(
      'div',
      { class: 'card' },
      h('div', { class: 'card-head' },
        h('span', { class: 'file', title: p.file }, p.file || 'change'),
        p.new_file_content != null ? h('button', { class: 'link-btn', onClick: () => vscode.postMessage({ type: 'openDiff', file: p.file, content: p.new_file_content }) }, 'Open diff') : null),
      p.diff ? renderDiff(p.diff) : renderDiff(p.new_file_content.split('\n').map((l) => '+' + l).join('\n'))
    )
    if (awaiting) {
      card.appendChild(
        h('div', { class: 'actions' },
          h('button', { class: 'btn primary', disabled: t.confirming, onClick: () => confirm(t, true) }, 'Accept'),
          h('button', { class: 'btn', disabled: t.confirming, onClick: () => confirm(t, false) }, 'Reject'),
          h('span', { class: 'note' }, t.metadata?.validation?.tests_passed === false ? 'tests failed' : 'validated, not written yet'))
      )
    }
    return card
  }

  function selectionCard(t) {
    const cands = t.metadata?.candidates || []
    t.selected = t.selected || []
    const list = h('ul', { class: 'candidates' })
    cands.forEach((c, i) => {
      const box = h('input', { type: 'checkbox', checked: t.selected.includes(i), onChange: (e) => {
        t.selected = e.target.checked ? [...t.selected, i] : t.selected.filter((x) => x !== i)
        render()
      } })
      list.appendChild(h('li', {}, h('label', {}, box, c.name, h('span', { class: 'muted' }, `${c.symbol_type} · L${c.start_line}-${c.end_line}`))))
    })
    return h('div', { class: 'card' },
      h('div', { class: 'card-head' }, h('span', { class: 'file' }, `Which one? (${t.metadata?.file || ''})`)),
      list,
      t.resolved ? null : h('div', { class: 'actions' },
        h('button', { class: 'btn primary', disabled: !t.selected.length || !!runningTurn(), onClick: () => confirmSelection(t) }, 'Delete selected')))
  }

  function findResults(meta) {
    const cands = meta.candidates || []
    if (!cands.length) return h('div', { class: 'status-line' }, 'No matching code found.')
    return h('div', { class: 'card' },
      h('div', { class: 'card-head' }, h('span', { class: 'file' }, `${cands.length} match${cands.length === 1 ? '' : 'es'} · confidence ${meta.confidence ?? '?'}`)),
      h('ul', { class: 'results' }, cands.slice(0, 15).map((c) =>
        h('li', {}, h('button', { onClick: () => vscode.postMessage({ type: 'openFile', file: c.file }) },
          h('span', { class: 'sym' }, c.symbol || '(module)'),
          h('span', { class: 'where' }, `${c.file}:${c.start_line}`))))))
  }

  function resultLine(t) {
    const meta = t.metadata
    const status = meta?.result?.status
    if (t.confirmed === 'accepted') return h('div', { class: 'status-line ok' }, '✓ Written to disk')
    if (t.confirmed === 'rejected') return h('div', { class: 'status-line' }, 'Rejected. Nothing was written.')
    if (status === 'success') {
      if (meta.result?.no_op) return h('div', { class: 'status-line' }, 'No change needed.')
      return h('div', { class: 'status-line ok' }, meta.new_version ? `✓ Applied (${meta.new_version})` : '✓ Applied')
    }
    if (status === 'failed') {
      return h('div', { class: 'error' }, [meta.result?.failure_class, meta.error].filter(Boolean).join(': ') || 'Run failed.')
    }
    return null
  }

  function renderTurn(t) {
    const steps = h('ul', { class: 'steps' })
    t.steps.forEach((s, i) => {
      const active = t.status === 'running' && i === t.steps.length - 1
      steps.appendChild(h('li', { class: 'step' + (active ? ' active' : '') },
        h('span', { class: 'step-dot' }, active ? null : ICONS.check()), h('span', { class: 'step-tag' }, s.tag), h('span', { class: 'step-msg' }, s.msg)))
    })
    if (t.status === 'running' && t.steps.length === 0) {
      steps.appendChild(h('li', { class: 'step active' }, h('span', { class: 'step-dot' }), h('span', { class: 'step-tag' }, 'START'), h('span', { class: 'step-msg' }, 'connecting to engine…')))
    }

    const meta = t.metadata
    const assistant = h('div', { class: 'assistant' }, steps)
    if (t.mode === 'find' && meta) assistant.appendChild(findResults(meta))
    if (meta?.answer) assistant.appendChild(h('div', { class: 'answer' }, meta.answer))
    if (t.status === 'needs_selection') assistant.appendChild(selectionCard(t))
    assistant.appendChild(previewCard(t) || document.createTextNode(''))
    const rl = resultLine(t)
    if (rl) assistant.appendChild(rl)
    if (t.error) assistant.appendChild(h('div', { class: 'error' }, t.error))
    if (t.confirmError) assistant.appendChild(h('div', { class: 'error' }, t.confirmError))
    if (t.cancelled) assistant.appendChild(h('div', { class: 'status-line' }, 'Stopped.'))
    const st = stats(meta)
    if (st) assistant.appendChild(st)

    return h('div', { class: 'turn' },
      h('div', { class: 'user-msg' }, t.text),
      h('div', { class: 'user-meta' }, h('span', {}, MODES[t.mode]?.label || t.mode), t.file ? h('span', {}, `· ${t.file}`) : null),
      assistant)
  }

  function render() {
    renderHeader()
    const atBottom = scroll.scrollHeight - scroll.scrollTop - scroll.clientHeight < 40
    scroll.replaceChildren(...(state.session.turns.length ? state.session.turns.map(renderTurn) : [renderEmpty()]))
    if (atBottom) scroll.scrollTop = scroll.scrollHeight
    renderComposer()
  }

  // ---- actions ---------------------------------------------------------------

  function newChat() {
    const r = runningTurn()
    if (r) vscode.postMessage({ type: 'stop', turnId: r.id })
    state.session = freshSession()
    state.tip = TIPS[Math.floor(Math.random() * TIPS.length)]
    persist()
    render()
    textarea.focus()
  }

  function startTurn(text, mode, file, extra = {}) {
    const turn = { id: uid(), text, mode, file, steps: [], status: 'running' }
    state.session.turns.push(turn)
    if (!state.session.title) state.session.title = text.length > 60 ? text.slice(0, 57) + '…' : text
    // "Create" without a chosen path goes through edit's create-intent
    // classifier, which picks the new file's path itself.
    const sendMode = mode === 'create' && !file ? 'edit' : mode
    vscode.postMessage({ type: 'send', turnId: turn.id, text, mode: sendMode, file, autoAccept: state.autoAccept, ...extra })
    persist()
    render()
    scroll.scrollTop = scroll.scrollHeight
  }

  function submit() {
    const text = textarea.value.trim()
    if (!text || runningTurn()) return
    if (text.startsWith('/') && runSlash(text)) return
    textarea.value = ''
    autosize()
    startTurn(text, state.mode, targetFile())
  }

  function onSendClick() {
    const r = runningTurn()
    if (r) vscode.postMessage({ type: 'stop', turnId: r.id })
    else submit()
  }

  function toggleAuto() {
    state.autoAccept = !state.autoAccept
    persist()
    renderComposer()
  }

  function setMode(mode) {
    state.mode = mode
    persist()
    render()
  }

  function confirm(t, accept) {
    t.confirming = true
    t.confirmError = null
    render()
    vscode.postMessage({ type: 'confirm', turnId: t.id, runId: t.metadata.run_id, accept })
  }

  function confirmSelection(t) {
    const picks = t.selected.map((i) => t.metadata.candidates[i])
    t.resolved = true
    startTurn(`${t.text}  [${picks.map((c) => c.name).join(', ')}]`, 'edit', t.metadata.file || t.file, {
      confirmSymbols: picks.map((c) => ({ name: c.name, symbol_type: c.symbol_type, start_line: c.start_line })),
    })
  }

  // ---- menus ---------------------------------------------------------------

  let menu = null

  function closeMenu() {
    if (menu) menu.el.remove()
    menu = null
  }

  function openMenu(anchor, items, { fromTyping = false } = {}) {
    closeMenu()
    if (!items.length) return
    const el = h('div', { class: 'menu', role: 'menu' })
    const buttons = items.map((it, i) =>
      h('button', { role: 'menuitem', class: i === 0 && fromTyping ? 'sel' : '', onMousedown: (e) => { e.preventDefault(); closeMenu(); it.run() } },
        h('span', { class: 'cmd' }, it.label), h('span', { class: 'desc' }, it.desc)))
    append(el, buttons)
    document.body.appendChild(el)
    const r = anchor.getBoundingClientRect()
    const width = Math.min(280, window.innerWidth - 16)
    el.style.width = width + 'px'
    el.style.left = Math.max(8, Math.min(r.left, window.innerWidth - width - 8)) + 'px'
    el.style.top = Math.max(8, r.top - el.offsetHeight - 6) + 'px'
    menu = { el, items, buttons, index: fromTyping ? 0 : -1, fromTyping }
  }

  function moveMenu(delta) {
    if (!menu) return
    menu.index = (menu.index + delta + menu.items.length) % menu.items.length
    menu.buttons.forEach((b, i) => b.classList.toggle('sel', i === menu.index))
  }

  const SLASH = [
    { name: 'edit', desc: MODES.edit.desc, run: () => setMode('edit') },
    { name: 'create', desc: MODES.create.desc, run: () => setMode('create') },
    { name: 'find', desc: MODES.find.desc, run: () => setMode('find') },
    { name: 'file', desc: 'Choose the target file', run: () => vscode.postMessage({ type: 'pickFile' }) },
    { name: 'auto', desc: 'Toggle Auto / Ask before writing', run: toggleAuto },
    { name: 'new', desc: 'Start a new conversation', run: newChat },
    { name: 'history', desc: 'Resume a past conversation', run: () => vscode.postMessage({ type: 'history' }) },
    { name: 'settings', desc: 'Engine extension settings', run: () => vscode.postMessage({ type: 'openSettings' }) },
  ]

  function slashItems(filter) {
    return SLASH.filter((c) => c.name.startsWith(filter)).map((c) => ({
      label: '/' + c.name,
      desc: c.desc,
      run: () => {
        textarea.value = ''
        autosize()
        renderComposer()
        c.run()
        textarea.focus()
      },
    }))
  }

  // "/edit fix the bug" switches mode and sends the rest; a bare "/new" runs.
  function runSlash(text) {
    const m = /^\/(\w+)\s*([\s\S]*)$/.exec(text)
    const cmd = m && SLASH.find((c) => c.name === m[1])
    if (!cmd) return false
    textarea.value = ''
    autosize()
    cmd.run()
    const rest = m[2].trim()
    if (rest && MODES[cmd.name]) startTurn(rest, state.mode, targetFile())
    else renderComposer()
    return true
  }

  function openSlashFromButton(e) {
    openMenu(e.currentTarget, slashItems(''))
  }


  function syncSlashMenu() {
    const v = textarea.value
    const m = /^\/(\w*)$/.exec(v)
    if (m) openMenu(composer, slashItems(m[1]), { fromTyping: true })
    else if (menu?.fromTyping) closeMenu()
  }

  document.addEventListener('mousedown', (e) => {
    if (menu && !menu.el.contains(e.target)) closeMenu()
  })

  // ---- composer input -------------------------------------------------------

  function autosize() {
    textarea.style.height = 'auto'
    textarea.style.height = Math.min(textarea.scrollHeight, 220) + 'px'
  }

  textarea.addEventListener('input', () => {
    autosize()
    syncSlashMenu()
    renderComposer()
  })

  textarea.addEventListener('keydown', (e) => {
    if (menu) {
      if (e.key === 'ArrowDown') { e.preventDefault(); moveMenu(1); return }
      if (e.key === 'ArrowUp') { e.preventDefault(); moveMenu(-1); return }
      if (e.key === 'Escape') { e.preventDefault(); closeMenu(); return }
      if ((e.key === 'Enter' || e.key === 'Tab') && menu.index >= 0) {
        e.preventDefault()
        const it = menu.items[menu.index]
        closeMenu()
        it.run()
        return
      }
    }
    if (e.key === 'Enter' && !e.shiftKey && !e.isComposing) {
      e.preventDefault()
      submit()
    }
  })

  // ---- messages from the extension host ------------------------------------

  window.addEventListener('message', (event) => {
    const m = event.data
    const t = m.turnId ? findTurn(m.turnId) : null
    switch (m.type) {
      case 'init':
        state.activeFile = m.activeFile
        break
      case 'activeFile':
        state.activeFile = m.path
        renderComposer()
        return
      case 'info':
        state.model = m.model || null
        persist()
        renderComposer()
        return
      case 'serverStatus':
        state.serverOk = m.ok
        renderComposer()
        return
      case 'fileChosen':
        state.fileOverride = m.path
        if (state.mode === 'find') state.mode = 'edit'
        persist()
        renderComposer()
        textarea.focus()
        return
      case 'reset':
        newChat()
        return
      case 'loadSession':
        state.session = m.session
        persist()
        break
      case 'step':
        if (!t) return
        t.steps.push({ tag: m.tag, msg: m.msg })
        break
      case 'preview':
        if (!t) return
        t.preview = { file: m.file, diff: m.diff, new_file_content: m.new_file_content }
        break
      case 'done':
        if (!t) return
        t.metadata = m.metadata
        t.status = 'done'
        if (!t.preview && m.metadata?.diff) t.preview = { file: m.metadata.file, diff: m.metadata.diff }
        break
      case 'needs_selection':
        if (!t) return
        t.metadata = m.metadata
        t.status = 'needs_selection'
        break
      case 'error':
        if (!t) return
        t.error = m.message
        t.status = 'error'
        break
      case 'end':
        if (!t) return
        if (t.status === 'running') t.status = m.cancelled ? 'cancelled' : 'done'
        if (m.cancelled) t.cancelled = true
        saveSession()
        break
      case 'confirmed':
        if (!t) return
        t.confirming = false
        t.confirmed = m.accept ? 'accepted' : 'rejected'
        if (m.result?.result?.status === 'success') t.metadata = { ...t.metadata, ...m.result }
        saveSession()
        break
      case 'confirmError':
        if (!t) return
        t.confirming = false
        t.confirmError = m.message
        break
      default:
        return
    }
    persist()
    render()
  })

  render()
  vscode.postMessage({ type: 'ready' })
  textarea.focus()
})()
