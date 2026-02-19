import type { AnswerMode, ChatResponse } from './types'

const DEFAULT_BACKEND = 'http://localhost:8000'

export function backendUrl(): string {
  return (import.meta as any).env?.VITE_BACKEND_URL || DEFAULT_BACKEND
}

async function httpJson<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${backendUrl()}${path}`, {
    ...init,
    headers: {
      'Content-Type': 'application/json',
      ...(init?.headers || {})
    }
  })

  if (!res.ok) {
    const text = await res.text()
    throw new Error(text || `HTTP ${res.status}`)
  }
  return (await res.json()) as T
}

export async function listSubjects(): Promise<string[]> {
  const res = await httpJson<{ subjects: string[] }>('/subjects')
  return res.subjects
}

export async function ingestText(params: {
  subject: string
  unit?: string
  topic?: string
  source_name?: string
  text: string
}): Promise<void> {
  const fd = new FormData()
  fd.set('subject', params.subject)
  if (params.unit) fd.set('unit', params.unit)
  if (params.topic) fd.set('topic', params.topic)
  fd.set('source_name', params.source_name || 'typed-notes')
  fd.set('text', params.text)

  const res = await fetch(`${backendUrl()}/ingest/text`, { method: 'POST', body: fd })
  if (!res.ok) throw new Error((await res.text()) || `HTTP ${res.status}`)
}

export async function ingestPdf(params: {
  subject: string
  unit?: string
  topic?: string
  file: File
}): Promise<void> {
  const fd = new FormData()
  fd.set('subject', params.subject)
  if (params.unit) fd.set('unit', params.unit)
  if (params.topic) fd.set('topic', params.topic)
  fd.set('file', params.file)

  const res = await fetch(`${backendUrl()}/ingest/pdf`, { method: 'POST', body: fd })
  if (!res.ok) throw new Error((await res.text()) || `HTTP ${res.status}`)
}

export async function ingestImage(params: {
  subject: string
  unit?: string
  topic?: string
  kind: 'handwritten' | 'diagram'
  file: File
}): Promise<void> {
  const fd = new FormData()
  fd.set('subject', params.subject)
  if (params.unit) fd.set('unit', params.unit)
  if (params.topic) fd.set('topic', params.topic)
  fd.set('kind', params.kind)
  fd.set('file', params.file)

  const res = await fetch(`${backendUrl()}/ingest/image`, { method: 'POST', body: fd })
  if (!res.ok) throw new Error((await res.text()) || `HTTP ${res.status}`)
}

export async function chatOnce(params: {
  subject: string
  question: string
  mode: AnswerMode
  top_k?: number
}): Promise<ChatResponse> {
  return await httpJson<ChatResponse>('/chat', {
    method: 'POST',
    body: JSON.stringify({
      subject: params.subject,
      question: params.question,
      mode: params.mode,
      top_k: params.top_k ?? 6
    })
  })
}

export function chatStream(params: {
  subject: string
  question: string
  mode: AnswerMode
  top_k?: number
  onToken: (t: string) => void
  onDone: (chunkRefsJson: string) => void
  onError: (e: any) => void
}): () => void {
  const url = new URL(`${backendUrl()}/chat/stream`)
  url.searchParams.set('subject', params.subject)
  url.searchParams.set('question', params.question)
  url.searchParams.set('mode', params.mode)
  url.searchParams.set('top_k', String(params.top_k ?? 6))

  const es = new EventSource(url.toString())
  es.addEventListener('token', (ev) => {
    params.onToken((ev as MessageEvent).data)
  })
  es.addEventListener('done', (ev) => {
    params.onDone((ev as MessageEvent).data)
    es.close()
  })
  es.onerror = (e) => {
    params.onError(e)
    es.close()
  }

  return () => es.close()
}
