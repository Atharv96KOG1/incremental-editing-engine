import * as http from 'http'
import { ServerAddress } from './server'

export type RunEvent =
  | { type: 'step'; tag: string; msg: string }
  | { type: 'preview'; file?: string; diff?: string; new_file_content?: string; operations?: unknown[] }
  | { type: 'done'; metadata: any }
  | { type: 'needs_selection'; metadata: any }
  | { type: 'error'; message: string }

function requestJson(
  addr: ServerAddress,
  method: 'GET' | 'POST',
  path: string,
  body?: unknown
): Promise<any> {
  return new Promise((resolve, reject) => {
    const payload = body === undefined ? undefined : JSON.stringify(body)
    const req = http.request(
      {
        host: addr.host,
        port: addr.port,
        path,
        method,
        headers: payload ? { 'Content-Type': 'application/json', 'Content-Length': Buffer.byteLength(payload) } : {},
      },
      (res) => {
        let data = ''
        res.setEncoding('utf8')
        res.on('data', (c) => (data += c))
        res.on('end', () => {
          let parsed: any
          try {
            parsed = data ? JSON.parse(data) : {}
          } catch {
            parsed = { detail: data }
          }
          if ((res.statusCode ?? 500) >= 400) {
            reject(new Error(parsed?.detail || `HTTP ${res.statusCode}`))
          } else {
            resolve(parsed)
          }
        })
      }
    )
    req.on('error', reject)
    if (payload) req.write(payload)
    req.end()
  })
}

export function fetchInfo(addr: ServerAddress): Promise<{ model?: string }> {
  return requestJson(addr, 'GET', '/api/info')
}

export function confirmRun(addr: ServerAddress, runId: string, accept: boolean): Promise<any> {
  return requestJson(addr, 'POST', '/api/confirm', { run_id: runId, accept })
}

/** POST /api/run and stream its Server-Sent Events. Returns a cancel
 * function; `onEnd` fires exactly once, after the last event or on cancel. */
export function streamRun(
  addr: ServerAddress,
  payload: Record<string, unknown>,
  onEvent: (e: RunEvent) => void,
  onEnd: (err?: Error) => void
): () => void {
  let ended = false
  const end = (err?: Error) => {
    if (ended) return
    ended = true
    onEnd(err)
  }

  const body = JSON.stringify(payload)
  const req = http.request(
    {
      host: addr.host,
      port: addr.port,
      path: '/api/run',
      method: 'POST',
      headers: { 'Content-Type': 'application/json', 'Content-Length': Buffer.byteLength(body) },
    },
    (res) => {
      if ((res.statusCode ?? 500) >= 400) {
        res.resume()
        end(new Error(`request failed: HTTP ${res.statusCode}`))
        return
      }
      res.setEncoding('utf8')
      let buffer = ''
      res.on('data', (chunk: string) => {
        buffer += chunk
        let idx
        while ((idx = buffer.indexOf('\n\n')) !== -1) {
          const frame = buffer.slice(0, idx)
          buffer = buffer.slice(idx + 2)
          if (!frame.startsWith('data: ')) continue
          try {
            onEvent(JSON.parse(frame.slice(6)))
          } catch {
            // A malformed frame shouldn't kill the whole stream.
          }
        }
      })
      res.on('end', () => end())
      res.on('error', (err) => end(err))
    }
  )
  req.on('error', (err) => end(err))
  req.write(body)
  req.end()

  return () => {
    req.destroy()
    end(new Error('cancelled'))
  }
}
