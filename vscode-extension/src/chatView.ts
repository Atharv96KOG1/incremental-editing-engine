import * as crypto from 'crypto'
import * as fs from 'fs'
import * as path from 'path'
import * as vscode from 'vscode'
import { confirmRun, fetchInfo, streamRun } from './api'
import { BackendServer, projectDir } from './server'

const SESSIONS_KEY = 'iee.sessions'
const MAX_SESSIONS = 40
const PROPOSED_SCHEME = 'iee-proposed'

interface StoredSession {
  id: string
  title: string
  updatedAt: number
  turns: unknown[]
}

/** In-memory documents backing the "proposed change" side of a diff view. */
class ProposedContentProvider implements vscode.TextDocumentContentProvider {
  private readonly docs = new Map<string, string>()
  private readonly changed = new vscode.EventEmitter<vscode.Uri>()
  readonly onDidChange = this.changed.event

  set(uri: vscode.Uri, content: string) {
    this.docs.set(uri.toString(), content)
    this.changed.fire(uri)
  }

  provideTextDocumentContent(uri: vscode.Uri): string {
    return this.docs.get(uri.toString()) ?? ''
  }
}

/** The sidebar chat: a native webview that talks to the local engine through
 * the extension host (so the webview needs no network access of its own). */
export class ChatViewProvider implements vscode.WebviewViewProvider, vscode.Disposable {
  static readonly viewType = 'incrementalEditingEngine.chat'

  private view: vscode.WebviewView | undefined
  private readonly running = new Map<string, () => void>()
  private readonly proposed = new ProposedContentProvider()
  private readonly disposables: vscode.Disposable[] = []

  constructor(
    private readonly context: vscode.ExtensionContext,
    private readonly server: BackendServer
  ) {
    this.disposables.push(
      vscode.workspace.registerTextDocumentContentProvider(PROPOSED_SCHEME, this.proposed),
      vscode.window.onDidChangeActiveTextEditor(() => this.postActiveFile())
    )
  }

  dispose() {
    for (const cancel of this.running.values()) cancel()
    this.running.clear()
    this.disposables.forEach((d) => d.dispose())
  }

  resolveWebviewView(view: vscode.WebviewView): void {
    this.view = view
    const mediaRoot = vscode.Uri.joinPath(this.context.extensionUri, 'media')
    view.webview.options = { enableScripts: true, localResourceRoots: [mediaRoot] }
    view.webview.html = this.html(view.webview, mediaRoot)
    view.webview.onDidReceiveMessage((m) => this.onMessage(m), undefined, this.disposables)
    view.onDidDispose(() => (this.view = undefined), undefined, this.disposables)
  }

  // ---- commands exposed to the extension -------------------------------

  async focus() {
    await vscode.commands.executeCommand(`${ChatViewProvider.viewType}.focus`)
  }

  newChat() {
    this.post({ type: 'reset' })
  }

  async showHistory() {
    const sessions = this.sessions()
    if (sessions.length === 0) {
      vscode.window.showInformationMessage('No past conversations in this workspace yet.')
      return
    }
    const pick = await vscode.window.showQuickPick(
      sessions.map((s) => ({
        label: s.title || 'Untitled',
        description: new Date(s.updatedAt).toLocaleString(),
        id: s.id,
      })),
      { placeHolder: 'Resume a conversation' }
    )
    if (!pick) return
    const session = sessions.find((s) => s.id === pick.id)
    if (session) {
      await this.focus()
      this.post({ type: 'loadSession', session })
    }
  }

  // ---- webview <-> host ------------------------------------------------

  private post(message: unknown) {
    this.view?.webview.postMessage(message)
  }

  private relativeActiveFile(): string | undefined {
    const doc = vscode.window.activeTextEditor?.document
    const root = projectDir()
    if (!doc || doc.uri.scheme !== 'file' || !root) return undefined
    const rel = path.relative(root, doc.uri.fsPath)
    if (!rel || rel.startsWith('..') || path.isAbsolute(rel)) return undefined
    return rel.split(path.sep).join('/')
  }

  private postActiveFile() {
    this.post({ type: 'activeFile', path: this.relativeActiveFile() ?? null })
  }

  private async onMessage(m: any) {
    switch (m?.type) {
      case 'ready':
        this.post({ type: 'init', projectDir: projectDir(), activeFile: this.relativeActiveFile() ?? null })
        void this.loadInfo()
        return
      case 'send':
        return this.run(m)
      case 'stop':
        this.running.get(m.turnId)?.()
        return
      case 'confirm':
        return this.confirm(m.turnId, m.runId, m.accept)
      case 'openDiff':
        return this.openDiff(m.file, m.content)
      case 'openFile':
        return this.openFile(m.file)
      case 'pickFile':
        return this.pickFile()
      case 'newChat':
        return this.newChat()
      case 'history':
        return this.showHistory()
      case 'saveSession':
        return this.saveSession(m.session)
      case 'walkthrough':
        await vscode.commands.executeCommand('incrementalEditingEngine.walkthrough')
        return
      case 'openSettings':
        await vscode.commands.executeCommand('workbench.action.openSettings', 'incrementalEditingEngine')
        return
    }
  }

  private async loadInfo() {
    const addr = await this.server.ensure()
    if (!addr) {
      this.post({ type: 'serverStatus', ok: false })
      return
    }
    this.post({ type: 'serverStatus', ok: true })
    try {
      const info = await fetchInfo(addr)
      this.post({ type: 'info', model: info.model })
    } catch {
      // Older engine without /api/info -- the chip just shows the mode.
    }
  }

  private async run(m: {
    turnId: string
    text: string
    mode: 'edit' | 'create' | 'find'
    file?: string | null
    autoAccept?: boolean
    confirmSymbols?: unknown[]
  }) {
    const root = projectDir()
    if (!root) {
      this.post({ type: 'error', turnId: m.turnId, message: 'Open a folder first -- the engine edits files inside a project directory.' })
      this.post({ type: 'end', turnId: m.turnId })
      return
    }
    const addr = await this.server.ensure()
    if (!addr) {
      this.post({ type: 'error', turnId: m.turnId, message: 'Engine server is not running. See the "Incremental Editing Engine" output.' })
      this.post({ type: 'end', turnId: m.turnId })
      return
    }

    const payload: Record<string, unknown> = {
      mode: m.mode,
      project_dir: root,
      file: m.file || null,
      request: m.text,
      test_target: '.',
      require_confirmation: !m.autoAccept,
    }
    if (m.confirmSymbols) payload.confirm_symbols = m.confirmSymbols

    const cancel = streamRun(
      addr,
      payload,
      (event) => this.post({ ...event, turnId: m.turnId }),
      (err) => {
        this.running.delete(m.turnId)
        if (err && err.message !== 'cancelled') {
          this.post({ type: 'error', turnId: m.turnId, message: err.message })
        }
        this.post({ type: 'end', turnId: m.turnId, cancelled: err?.message === 'cancelled' })
      }
    )
    this.running.set(m.turnId, cancel)
  }

  private async confirm(turnId: string, runId: string, accept: boolean) {
    const addr = await this.server.ensure()
    if (!addr) return
    try {
      const result = await confirmRun(addr, runId, accept)
      this.post({ type: 'confirmed', turnId, accept, result })
    } catch (err) {
      this.post({ type: 'confirmError', turnId, message: (err as Error).message })
    }
  }

  private async openDiff(file: string, content: string) {
    const root = projectDir()
    if (!root || typeof content !== 'string') return
    const original = vscode.Uri.file(path.join(root, file))
    const proposed = vscode.Uri.from({ scheme: PROPOSED_SCHEME, path: '/' + file, query: String(Date.now()) })
    this.proposed.set(proposed, content)
    if (fs.existsSync(original.fsPath)) {
      await vscode.commands.executeCommand('vscode.diff', original, proposed, `${path.basename(file)} (proposed)`)
    } else {
      await vscode.window.showTextDocument(proposed, { preview: true })
    }
  }

  private async openFile(file: string) {
    const root = projectDir()
    if (!root || !file) return
    const uri = vscode.Uri.file(path.join(root, file))
    try {
      await vscode.window.showTextDocument(uri, { preview: true })
    } catch {
      vscode.window.showWarningMessage(`Couldn't open ${file}`)
    }
  }

  private async pickFile() {
    const root = projectDir()
    if (!root) return
    const uris = await vscode.workspace.findFiles(
      '**/*',
      '{**/node_modules/**,**/.git/**,**/.venv/**,**/venv/**,**/__pycache__/**,**/dist/**}',
      2000
    )
    const items = uris
      .map((u) => path.relative(root, u.fsPath).split(path.sep).join('/'))
      .filter((p) => p && !p.startsWith('..'))
      .sort()
    const pick = await vscode.window.showQuickPick(items, { placeHolder: 'Target file for this request' })
    if (pick) this.post({ type: 'fileChosen', path: pick })
  }

  private sessions(): StoredSession[] {
    return this.context.workspaceState.get<StoredSession[]>(SESSIONS_KEY, [])
  }

  private async saveSession(session: StoredSession) {
    if (!session?.id) return
    const rest = this.sessions().filter((s) => s.id !== session.id)
    const next = session.turns?.length ? [session, ...rest] : rest
    await this.context.workspaceState.update(SESSIONS_KEY, next.slice(0, MAX_SESSIONS))
  }

  private html(webview: vscode.Webview, mediaRoot: vscode.Uri): string {
    const nonce = crypto.randomBytes(16).toString('base64')
    const css = webview.asWebviewUri(vscode.Uri.joinPath(mediaRoot, 'chat.css'))
    const js = webview.asWebviewUri(vscode.Uri.joinPath(mediaRoot, 'chat.js'))
    const csp = [
      `default-src 'none'`,
      `style-src ${webview.cspSource}`,
      `img-src ${webview.cspSource} data:`,
      `script-src 'nonce-${nonce}'`,
    ].join('; ')
    return `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta http-equiv="Content-Security-Policy" content="${csp}">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <link rel="stylesheet" href="${css}">
  <title>Incremental Editing Engine</title>
</head>
<body>
  <div id="app"></div>
  <script nonce="${nonce}" src="${js}"></script>
</body>
</html>`
  }
}
