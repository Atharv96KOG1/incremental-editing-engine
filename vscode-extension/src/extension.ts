import * as vscode from 'vscode'
import { ChatViewProvider } from './chatView'
import { BackendServer, cfg, projectDir } from './server'

let panel: vscode.WebviewPanel | undefined

/** The full web UI (file explorer, tabs, etc.) in an editor tab -- kept for
 * when the sidebar chat isn't enough room. */
async function openFullUi(server: BackendServer) {
  const addr = await server.ensure()
  if (!addr) return
  if (panel) {
    panel.reveal(vscode.ViewColumn.Active)
    return
  }
  const src = `http://${addr.host}:${addr.port}/?projectDir=${encodeURIComponent(projectDir())}`
  const csp = [`default-src 'none'`, `frame-src http://${addr.host}:${addr.port}`, `style-src 'unsafe-inline'`].join('; ')
  panel = vscode.window.createWebviewPanel('incrementalEditingEngine.full', 'Incremental Editing Engine', vscode.ViewColumn.Active, {
    enableScripts: true,
    retainContextWhenHidden: true,
  })
  panel.webview.html = `<!DOCTYPE html>
<html><head><meta charset="UTF-8"><meta http-equiv="Content-Security-Policy" content="${csp}">
<style>html, body, iframe { height: 100%; width: 100%; margin: 0; padding: 0; border: 0; }</style></head>
<body><iframe src="${src}" allow="clipboard-read; clipboard-write"></iframe></body></html>`
  panel.onDidDispose(() => (panel = undefined))
}

export function activate(context: vscode.ExtensionContext) {
  const output = vscode.window.createOutputChannel('Incremental Editing Engine')
  const statusBar = vscode.window.createStatusBarItem(vscode.StatusBarAlignment.Right, 100)
  statusBar.command = 'incrementalEditingEngine.focus'
  statusBar.text = '$(zap) IEE'
  statusBar.tooltip = 'Open Incremental Editing Engine chat'
  statusBar.show()

  const server = new BackendServer(output, statusBar)
  const chat = new ChatViewProvider(context, server)

  context.subscriptions.push(
    output,
    statusBar,
    server,
    chat,
    vscode.window.registerWebviewViewProvider(ChatViewProvider.viewType, chat, {
      webviewOptions: { retainContextWhenHidden: true },
    }),
    vscode.commands.registerCommand('incrementalEditingEngine.focus', () => chat.focus()),
    vscode.commands.registerCommand('incrementalEditingEngine.newChat', async () => {
      await chat.focus()
      chat.newChat()
    }),
    vscode.commands.registerCommand('incrementalEditingEngine.walkthrough', () =>
      vscode.commands.executeCommand(
        'workbench.action.openWalkthrough',
        `${context.extension.id}#incrementalEditingEngine.gettingStarted`,
        false
      )
    ),
    vscode.commands.registerCommand('incrementalEditingEngine.history', () => chat.showHistory()),
    vscode.commands.registerCommand('incrementalEditingEngine.openPanel', () => openFullUi(server)),
    vscode.commands.registerCommand('incrementalEditingEngine.start', () => server.ensure()),
    vscode.commands.registerCommand('incrementalEditingEngine.stop', () => server.stop()),
    vscode.commands.registerCommand('incrementalEditingEngine.restart', async () => {
      server.stop()
      await server.ensure()
    }),
    vscode.workspace.onDidChangeConfiguration((e) => {
      if (e.affectsConfiguration('incrementalEditingEngine.port') || e.affectsConfiguration('incrementalEditingEngine.host')) {
        output.appendLine(`Server address changed to ${cfg().host}:${cfg().port}; restart the server to apply.`)
      }
    })
  )
}

export function deactivate() {
  // BackendServer is disposed via context.subscriptions.
}
