import * as http from 'http'
import * as vscode from 'vscode'
import { ChildProcess, spawn } from 'child_process'

export interface ServerAddress {
  host: string
  port: number
}

export function cfg() {
  const c = vscode.workspace.getConfiguration('incrementalEditingEngine')
  return {
    command: c.get<string>('command', 'iee'),
    pythonFallback: c.get<string>('pythonFallback', 'python3'),
    host: c.get<string>('host', '127.0.0.1'),
    port: c.get<number>('port', 8787),
    projectDir: c.get<string>('projectDir', ''),
  }
}

export function projectDir(): string {
  return cfg().projectDir || vscode.workspace.workspaceFolders?.[0]?.uri.fsPath || ''
}

function isAlive(proc: ChildProcess | undefined): proc is ChildProcess {
  return !!proc && proc.exitCode === null && !proc.killed
}

function pingServer(host: string, port: number): Promise<boolean> {
  return new Promise((resolve) => {
    const req = http.get({ host, port, path: '/', timeout: 1000 }, (res) => {
      res.resume()
      resolve(res.statusCode !== undefined)
    })
    req.on('error', () => resolve(false))
    req.on('timeout', () => {
      req.destroy()
      resolve(false)
    })
  })
}

async function waitForServer(host: string, port: number, timeoutMs: number): Promise<boolean> {
  const deadline = Date.now() + timeoutMs
  while (Date.now() < deadline) {
    if (await pingServer(host, port)) return true
    await new Promise((r) => setTimeout(r, 300))
  }
  return false
}

/** Owns the local `iee serve` process: starts it on demand, reuses one that
 * is already listening, and reports state through the status bar. */
export class BackendServer implements vscode.Disposable {
  private proc: ChildProcess | undefined
  private port: number | undefined
  private starting: Promise<ServerAddress | undefined> | undefined

  constructor(
    private readonly output: vscode.OutputChannel,
    private readonly statusBar: vscode.StatusBarItem
  ) {}

  ensure(): Promise<ServerAddress | undefined> {
    // Concurrent callers (sidebar + command) share one startup attempt.
    if (!this.starting) {
      this.starting = this.start().finally(() => {
        this.starting = undefined
      })
    }
    return this.starting
  }

  private spawn(command: string, args: string[]): ChildProcess {
    const proc = spawn(command, args, {
      cwd: vscode.workspace.workspaceFolders?.[0]?.uri.fsPath,
      shell: process.platform === 'win32',
    })
    proc.stdout?.on('data', (d) => this.output.append(d.toString()))
    proc.stderr?.on('data', (d) => this.output.append(d.toString()))
    return proc
  }

  private async start(): Promise<ServerAddress | undefined> {
    const { command, pythonFallback, host, port } = cfg()

    if (isAlive(this.proc) && this.port === port) return { host, port }

    if (await pingServer(host, port)) {
      // Something is already serving on this host/port -- reuse it.
      this.port = port
      this.statusBar.text = '$(zap) IEE'
      return { host, port }
    }

    this.output.appendLine(`Starting Incremental Editing Engine on ${host}:${port}...`)
    this.statusBar.text = '$(sync~spin) IEE: starting'

    const serveArgs = ['serve', '--host', host, '--port', String(port)]
    let failedToSpawn = false
    const proc = this.spawn(command, serveArgs)
    proc.on('error', (err: Error) => {
      this.output.appendLine(
        `'${command}' didn't launch (${err.message}); falling back to '${pythonFallback} -m incremental_editing.cli'...`
      )
      const fallback = this.spawn(pythonFallback, ['-m', 'incremental_editing.cli', ...serveArgs])
      fallback.on('error', (fallbackErr: Error) => {
        failedToSpawn = true
        this.output.appendLine(`Failed to launch '${pythonFallback}' too: ${fallbackErr.message}`)
      })
      this.proc = fallback
    })
    this.proc = proc
    this.port = port

    const ready = await waitForServer(host, port, 20000)
    if (!ready || failedToSpawn) {
      this.statusBar.text = '$(error) IEE: failed'
      this.output.show(true)
      vscode.window.showErrorMessage(
        `Incremental Editing Engine server didn't come up on ${host}:${port}. ` +
          `Check the "Incremental Editing Engine" output channel, and make sure '${command}' ` +
          `is on PATH (pip install -e . in the engine repo) or set incrementalEditingEngine.command.`
      )
      return undefined
    }

    this.statusBar.text = '$(zap) IEE'
    return { host, port }
  }

  stop(): void {
    if (isAlive(this.proc)) {
      this.proc.kill()
      this.output.appendLine('Backend stopped.')
    }
    this.proc = undefined
    this.port = undefined
    this.statusBar.text = '$(circle-slash) IEE: stopped'
  }

  dispose(): void {
    this.stop()
  }
}
