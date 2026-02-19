export type AnswerMode = 'short' | '5_mark' | '10_mark' | 'detailed' | 'numerical'

export type ChunkRef = {
  source_name: string
  source_type: string
  page?: number | null
  subject: string
  unit?: string | null
  topic?: string | null
}

export type ChatResponse = {
  answer: string
  chunks_used: ChunkRef[]
}
