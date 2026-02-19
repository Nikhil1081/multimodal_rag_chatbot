import React, { useEffect, useMemo, useRef, useState } from 'react'
import { backendUrl, chatOnce, chatStream, ingestImage, ingestPdf, ingestText, listSubjects } from './api'
import type { AnswerMode, ChunkRef } from './types'

type ChatMsg = {
  role: 'user' | 'assistant'
  text: string
  chunks?: ChunkRef[]
}

function useSpeechToText() {
  const [supported] = useState(() => {
    const w = window as any
    return Boolean(w.SpeechRecognition || w.webkitSpeechRecognition)
  })
  const [listening, setListening] = useState(false)

  const start = (onText: (t: string) => void, onError: (e: any) => void) => {
    const w = window as any
    const SR = w.SpeechRecognition || w.webkitSpeechRecognition
    if (!SR) return

    const rec = new SR()
    rec.lang = 'en-US'
    rec.interimResults = false
    rec.maxAlternatives = 1

    rec.onstart = () => setListening(true)
    rec.onend = () => setListening(false)
    rec.onerror = (e: any) => {
      setListening(false)
      onError(e)
    }
    rec.onresult = (event: any) => {
      const text = event.results?.[0]?.[0]?.transcript
      if (text) onText(text)
    }

    rec.start()
  }

  return { supported, listening, start }
}

export default function App() {
  const [subjects, setSubjects] = useState<string[]>([])
  const [subject, setSubject] = useState('DBMS')
  const [unit, setUnit] = useState('')
  const [topic, setTopic] = useState('')

  const [mode, setMode] = useState<AnswerMode>('5_mark')
  const [question, setQuestion] = useState('')

  const [uploadPdf, setUploadPdf] = useState<File | null>(null)
  const [uploadImg, setUploadImg] = useState<File | null>(null)
  const [imgKind, setImgKind] = useState<'handwritten' | 'diagram'>('handwritten')
  const [notes, setNotes] = useState('')

  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const [messages, setMessages] = useState<ChatMsg[]>([])
  const bottomRef = useRef<HTMLDivElement | null>(null)

  const speech = useSpeechToText()

  const canAsk = useMemo(() => subject.trim().length > 0 && question.trim().length > 0 && !busy, [subject, question, busy])

  useEffect(() => {
    listSubjects()
      .then((s) => {
        setSubjects(s)
        if (s.length > 0 && !s.includes(subject)) setSubject(s[0])
      })
      .catch(() => {
        // ignore; backend might be down
      })
  }, [])

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  async function refreshSubjects() {
    const s = await listSubjects()
    setSubjects(s)
  }

  async function onIngestPdf() {
    if (!uploadPdf) return
    setError(null)
    setBusy(true)
    try {
      await ingestPdf({ subject, unit: unit || undefined, topic: topic || undefined, file: uploadPdf })
      setUploadPdf(null)
      await refreshSubjects()
    } catch (e: any) {
      setError(e?.message || String(e))
    } finally {
      setBusy(false)
    }
  }

  async function onIngestImage() {
    if (!uploadImg) return
    setError(null)
    setBusy(true)
    try {
      await ingestImage({ subject, unit: unit || undefined, topic: topic || undefined, kind: imgKind, file: uploadImg })
      setUploadImg(null)
      await refreshSubjects()
    } catch (e: any) {
      setError(e?.message || String(e))
    } finally {
      setBusy(false)
    }
  }

  async function onIngestNotes() {
    if (!notes.trim()) return
    setError(null)
    setBusy(true)
    try {
      await ingestText({ subject, unit: unit || undefined, topic: topic || undefined, text: notes, source_name: 'typed-notes' })
      setNotes('')
      await refreshSubjects()
    } catch (e: any) {
      setError(e?.message || String(e))
    } finally {
      setBusy(false)
    }
  }

  async function onAskStream() {
    if (!canAsk) return
    setError(null)

    const q = question.trim()
    setQuestion('')

    setMessages((m: ChatMsg[]) => [...m, { role: 'user', text: q }, { role: 'assistant', text: '' }])

    setBusy(true)
    let stop: null | (() => void) = null

    try {
      stop = chatStream({
        subject,
        question: q,
        mode,
        top_k: 6,
        onToken: (t) => {
          setMessages((prev: ChatMsg[]) => {
            const next = [...prev]
            const last = next[next.length - 1]
            if (!last || last.role !== 'assistant') return prev
            last.text += t
            return next
          })
        },
        onDone: (json) => {
          try {
            const chunks = JSON.parse(json) as ChunkRef[]
            setMessages((prev: ChatMsg[]) => {
              const next = [...prev]
              const last = next[next.length - 1]
              if (last && last.role === 'assistant') last.chunks = chunks
              return next
            })
          } catch {
            // ignore parse errors
          }
          setBusy(false)
        },
        onError: (e) => {
          setBusy(false)
          setError('Streaming error. Check backend and CORS.')
          console.error(e)
        }
      })
    } catch (e: any) {
      setBusy(false)
      setError(e?.message || String(e))
      if (stop) stop()
    }
  }

  async function onAskOnce() {
    if (!canAsk) return
    setError(null)
    const q = question.trim()
    setQuestion('')
    setBusy(true)
    setMessages((m: ChatMsg[]) => [...m, { role: 'user', text: q }])

    try {
      const res = await chatOnce({ subject, question: q, mode, top_k: 6 })
      setMessages((m: ChatMsg[]) => [...m, { role: 'assistant', text: res.answer, chunks: res.chunks_used }])
    } catch (e: any) {
      setError(e?.message || String(e))
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="min-h-screen">
      <header className="border-b bg-white">
        <div className="mx-auto max-w-6xl px-4 py-3 flex items-center justify-between">
          <div>
            <div className="text-lg font-semibold">B.Tech Multimodal Study RAG Assistant</div>
            <div className="text-xs text-gray-500">Backend: {backendUrl()}</div>
          </div>
          <button
            className="text-sm px-3 py-1.5 rounded border bg-white hover:bg-gray-50"
            onClick={() => refreshSubjects().catch(() => {})}
            disabled={busy}
          >
            Refresh subjects
          </button>
        </div>
      </header>

      <main className="mx-auto max-w-6xl px-4 py-4 grid grid-cols-1 lg:grid-cols-3 gap-4">
        <section className="lg:col-span-1 space-y-4">
          <div className="rounded border bg-white p-3">
            <div className="font-medium mb-2">Study context</div>
            <label className="text-sm text-gray-600">Subject</label>
            <div className="flex gap-2 mt-1">
              <input
                className="flex-1 rounded border px-2 py-1.5"
                value={subject}
                onChange={(e: React.ChangeEvent<HTMLInputElement>) => setSubject(e.target.value)}
                placeholder="DBMS / OS / DSA ..."
              />
              <select
                className="rounded border px-2 py-1.5"
                value={subject}
                onChange={(e: React.ChangeEvent<HTMLSelectElement>) => setSubject(e.target.value)}
              >
                {[subject, ...subjects.filter((s: string) => s !== subject)].map((s: string) => (
                  <option key={s} value={s}>
                    {s}
                  </option>
                ))}
              </select>
            </div>

            <div className="grid grid-cols-2 gap-2 mt-3">
              <div>
                <label className="text-sm text-gray-600">Unit (optional)</label>
                <input
                  className="w-full rounded border px-2 py-1.5 mt-1"
                  value={unit}
                  onChange={(e: React.ChangeEvent<HTMLInputElement>) => setUnit(e.target.value)}
                />
              </div>
              <div>
                <label className="text-sm text-gray-600">Topic (optional)</label>
                <input
                  className="w-full rounded border px-2 py-1.5 mt-1"
                  value={topic}
                  onChange={(e: React.ChangeEvent<HTMLInputElement>) => setTopic(e.target.value)}
                />
              </div>
            </div>

            <div className="mt-3">
              <label className="text-sm text-gray-600">Answer mode</label>
              <select className="w-full rounded border px-2 py-1.5 mt-1" value={mode} onChange={(e) => setMode(e.target.value as AnswerMode)}>
                <option value="short">Short (2 marks)</option>
                <option value="5_mark">5-mark university</option>
                <option value="10_mark">10-mark university</option>
                <option value="detailed">Detailed</option>
                <option value="numerical">Numerical</option>
              </select>
            </div>
          </div>

          <div className="rounded border bg-white p-3 space-y-3">
            <div className="font-medium">Ingest study material</div>

            <div className="rounded border p-2">
              <div className="text-sm font-medium">PDF</div>
              <input
                type="file"
                accept="application/pdf"
                onChange={(e: React.ChangeEvent<HTMLInputElement>) => setUploadPdf(e.target.files?.[0] || null)}
                className="mt-2 block w-full text-sm"
              />
              <button
                className="mt-2 w-full rounded bg-gray-900 text-white px-3 py-2 text-sm disabled:opacity-50"
                disabled={!uploadPdf || busy}
                onClick={() => onIngestPdf()}
              >
                Upload PDF
              </button>
            </div>

            <div className="rounded border p-2">
              <div className="text-sm font-medium">Image (handwritten/diagram)</div>
              <div className="flex gap-2 mt-2">
                <select
                  className="rounded border px-2 py-1.5 text-sm"
                  value={imgKind}
                  onChange={(e: React.ChangeEvent<HTMLSelectElement>) => setImgKind(e.target.value as any)}
                >
                  <option value="handwritten">Handwritten</option>
                  <option value="diagram">Diagram</option>
                </select>
                <input
                  type="file"
                  accept="image/*"
                  onChange={(e: React.ChangeEvent<HTMLInputElement>) => setUploadImg(e.target.files?.[0] || null)}
                  className="flex-1 block w-full text-sm"
                />
              </div>
              <button
                className="mt-2 w-full rounded bg-gray-900 text-white px-3 py-2 text-sm disabled:opacity-50"
                disabled={!uploadImg || busy}
                onClick={() => onIngestImage()}
              >
                Upload Image
              </button>
            </div>

            <div className="rounded border p-2">
              <div className="text-sm font-medium">Typed notes</div>
              <textarea
                className="mt-2 w-full rounded border p-2 text-sm min-h-24"
                value={notes}
                onChange={(e: React.ChangeEvent<HTMLTextAreaElement>) => setNotes(e.target.value)}
                placeholder="Paste notes here..."
              />
              <button
                className="mt-2 w-full rounded bg-gray-900 text-white px-3 py-2 text-sm disabled:opacity-50"
                disabled={!notes.trim() || busy}
                onClick={() => onIngestNotes()}
              >
                Ingest Notes
              </button>
            </div>
          </div>

          {error ? (
            <div className="rounded border border-red-200 bg-red-50 p-3 text-sm text-red-700">{error}</div>
          ) : null}

          <div className="rounded border bg-white p-3 text-sm text-gray-600">
            <div className="font-medium text-gray-800 mb-1">Setup note</div>
            Backend must have <code className="font-mono">GEMINI_API_KEY</code> set in <code className="font-mono">backend/.env</code>.
          </div>
        </section>

        <section className="lg:col-span-2 rounded border bg-white flex flex-col min-h-[70vh]">
          <div className="border-b px-3 py-2 font-medium">Chat</div>

          <div className="flex-1 p-3 overflow-auto space-y-3">
            {messages.length === 0 ? (
              <div className="text-sm text-gray-500">
                Upload material, then ask a doubt like: <span className="font-mono">Explain 2-phase locking with example</span>
              </div>
            ) : null}

            {messages.map((m, idx) => (
              <div key={idx} className={m.role === 'user' ? 'flex justify-end' : 'flex justify-start'}>
                <div
                  className={
                    (m.role === 'user'
                      ? 'bg-gray-900 text-white'
                      : 'bg-gray-100 text-gray-900') +
                    ' max-w-[85%] rounded px-3 py-2 text-sm whitespace-pre-wrap'
                  }
                >
                  {m.text || (m.role === 'assistant' && busy && idx === messages.length - 1 ? '...' : '')}
                  {m.role === 'assistant' && m.chunks && m.chunks.length > 0 ? (
                    <div className="mt-2 text-xs text-gray-600">
                      Sources: {m.chunks.slice(0, 3).map((c, i) => (
                        <span key={i} className="mr-2">
                          {c.source_name}{c.page ? ` p.${c.page}` : ''}
                        </span>
                      ))}
                    </div>
                  ) : null}
                </div>
              </div>
            ))}
            <div ref={bottomRef} />
          </div>

          <div className="border-t p-3">
            <div className="flex gap-2">
              <input
                className="flex-1 rounded border px-3 py-2 text-sm"
                value={question}
                onChange={(e: React.ChangeEvent<HTMLInputElement>) => setQuestion(e.target.value)}
                onKeyDown={(e: React.KeyboardEvent<HTMLInputElement>) => {
                  if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault()
                    onAskStream().catch(() => {})
                  }
                }}
                placeholder="Ask your doubt..."
                disabled={busy}
              />
              <button
                className="rounded bg-gray-900 text-white px-3 py-2 text-sm disabled:opacity-50"
                onClick={() => onAskStream()}
                disabled={!canAsk}
              >
                Ask (stream)
              </button>
              <button
                className="rounded border px-3 py-2 text-sm disabled:opacity-50"
                onClick={() => onAskOnce()}
                disabled={!canAsk}
              >
                Ask
              </button>
            </div>

            <div className="mt-2 flex items-center justify-between">
              <div className="text-xs text-gray-500">Tip: press Enter to send</div>
              {speech.supported ? (
                <button
                  className="text-xs rounded border px-2 py-1 disabled:opacity-50"
                  disabled={busy || speech.listening}
                  onClick={() =>
                    speech.start(
                      (t) => setQuestion((q) => (q ? q + ' ' : '') + t),
                      () => setError('Voice input error')
                    )
                  }
                >
                  {speech.listening ? 'Listening…' : 'Voice → Text'}
                </button>
              ) : (
                <div className="text-xs text-gray-400">Voice not supported in this browser</div>
              )}
            </div>
          </div>
        </section>
      </main>
    </div>
  )
}
