// Talks to POST /api/run, which streams Server-Sent Events: one "step"
// event per pipeline phase (LOCALIZE/GENERATE/VALIDATE/APPLY/TEST/COMMIT),
// one "preview" event as soon as the actual code change exists (right
// after APPLY for an edit, right after GENERATE for a create -- well
// before TEST/COMMIT finish), then a single terminal "done", "error", or
// "needs_selection" (a destructive "remove/delete X" request whose target
// matched more than one real symbol -- resubmit with confirm_symbol set
// to the one the user picks instead of guessing).

// File explorer support -- read-only project browsing so the UI can show a
// VS Code-style tree/tabs view alongside the chat, independent of whatever
// the pipeline itself is doing.

export async function fetchTree(projectDir) {
  const resp = await fetch(`/api/tree?project_dir=${encodeURIComponent(projectDir)}`)
  if (!resp.ok) throw new Error(`tree request failed: HTTP ${resp.status}`)
  return resp.json()
}

export async function fetchFile(projectDir, path) {
  const resp = await fetch(`/api/file?project_dir=${encodeURIComponent(projectDir)}&path=${encodeURIComponent(path)}`)
  if (!resp.ok) {
    // The backend's own `detail` (e.g. "binary file, cannot display" for
    // a real .db/.xlsx artifact) is far more useful than a bare status
    // code -- fall back to the status code only if the body isn't the
    // JSON shape FastAPI's HTTPException always sends.
    const detail = await resp
      .json()
      .then((body) => body?.detail)
      .catch(() => null)
    throw new Error(detail || `file request failed: HTTP ${resp.status}`)
  }
  return resp.json()
}

// Resolves a run the backend paused right before COMMIT (result.status ===
// "awaiting_confirmation") -- accept writes the file and creates the
// version; reject just discards the pending state (nothing was ever
// written to disk, so there's nothing to undo on disk itself).
export async function confirmRun(runId, accept) {
  const resp = await fetch('/api/confirm', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ run_id: runId, accept }),
  })
  if (!resp.ok) {
    const body = await resp.json().catch(() => ({}))
    throw new Error(body.detail || `confirm failed: HTTP ${resp.status}`)
  }
  return resp.json()
}

// Simple resume, not a multi-session browser: one continuous chat log per
// project, auto-saved so reopening the app picks the conversation back up
// instead of starting blank.
export async function fetchChatHistory(projectDir) {
  const resp = await fetch(`/api/chat_history?project_dir=${encodeURIComponent(projectDir)}`)
  if (!resp.ok) throw new Error(`chat history request failed: HTTP ${resp.status}`)
  return resp.json()
}

export async function saveChatHistory(projectDir, messages) {
  await fetch('/api/chat_history', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ project_dir: projectDir, messages }),
  })
}

export async function streamRun(payload, { onStep, onPreview, onDone, onError, onNeedsSelection }) {
  const resp = await fetch('/api/run', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  })

  if (!resp.ok || !resp.body) {
    onError(`request failed: HTTP ${resp.status}`)
    return
  }

  const reader = resp.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''

  while (true) {
    const { done, value } = await reader.read()
    if (done) break
    buffer += decoder.decode(value, { stream: true })

    let idx
    while ((idx = buffer.indexOf('\n\n')) !== -1) {
      const chunk = buffer.slice(0, idx)
      buffer = buffer.slice(idx + 2)
      if (!chunk.startsWith('data: ')) continue

      const event = JSON.parse(chunk.slice(6))
      if (event.type === 'step') onStep(event.tag, event.msg)
      else if (event.type === 'preview') onPreview(event)
      else if (event.type === 'done') onDone(event.metadata)
      else if (event.type === 'needs_selection') onNeedsSelection?.(event.metadata)
      else if (event.type === 'error') onError(event.message)
    }
  }
}
