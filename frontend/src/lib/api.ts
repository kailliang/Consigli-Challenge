import { z } from "zod";

import type { ChatMessage } from "@/lib/types";

const envSchema = z.object({
  VITE_API_BASE_URL: z.string().url().optional().default("http://localhost:8000/v1")
});

const env = envSchema.parse(import.meta.env);

const citationSchema = z.object({
  id: z.string(),
  source: z.string(),
  page: z.string().optional().nullable(),
  section: z.string().optional().nullable(),
  snippet: z.string()
});

const messageSchema = z.object({
  id: z.string(),
  role: z.enum(["assistant", "user", "system"]),
  content: z.string(),
  created_at: z.string(),
  citations: z.array(citationSchema).optional()
});

const queryResponseSchema = z.object({
  session_id: z.string(),
  message: messageSchema
});

const streamEventSchema = z.object({
  event: z.string(),
  data: z.record(z.any())
});

export type QueryResponse = z.infer<typeof queryResponseSchema>;

const toChatMessage = (backend: z.infer<typeof messageSchema>): ChatMessage => ({
  id: backend.id,
  role: backend.role,
  content: backend.content,
  createdAt: new Date(backend.created_at).toISOString(),
  citations:
    backend.citations?.map((citation) => ({
      id: citation.id,
      source: citation.source,
      page: citation.page ?? undefined,
      snippet: citation.snippet,
      section: citation.section ?? undefined
    })) ?? []
});

export const queryReports = async (
  prompt: string,
  sessionId?: string
): Promise<{ sessionId: string; message: ChatMessage }> => {
  const response = await fetch(`${env.VITE_API_BASE_URL}/query`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json"
    },
    body: JSON.stringify({ prompt, session_id: sessionId })
  });

  if (!response.ok) {
    const detail = await response.text();
    throw new Error(detail || "Query failed");
  }

  const data = queryResponseSchema.parse(await response.json());

  return {
    sessionId: data.session_id,
    message: toChatMessage(data.message)
  };
};

type StreamCallback = (event: { type: "token"; content: string } | { type: "done"; sessionId: string; message: ChatMessage }) => void;

export const queryReportsStream = async (
  prompt: string,
  sessionId: string | undefined,
  onEvent: StreamCallback
): Promise<{ sessionId: string; message: ChatMessage }> => {
  const response = await fetch(`${env.VITE_API_BASE_URL}/query/stream`, {
    method: "POST",
    headers: {
      Accept: "text/event-stream",
      "Content-Type": "application/json"
    },
    body: JSON.stringify({ prompt, session_id: sessionId })
  });

  if (!response.ok) {
    const detail = await response.text();
    throw new Error(detail || "Streaming request failed");
  }

  const reader = response.body?.getReader();
  if (!reader) {
    throw new Error("Streaming not supported in this environment");
  }

  const decoder = new TextDecoder();
  let buffer = "";
  let finalSessionId = sessionId ?? "";
  let finalMessage: ChatMessage | null = null;

  while (true) {
    const { value, done } = await reader.read();
    if (done) {
      break;
    }

    buffer += decoder.decode(value, { stream: true });

    let delimiterIndex: number;
    while ((delimiterIndex = buffer.indexOf("\n\n")) !== -1) {
      const rawEvent = buffer.slice(0, delimiterIndex).trim();
      buffer = buffer.slice(delimiterIndex + 2);

      if (!rawEvent.startsWith("data:")) {
        continue;
      }

      const payload = rawEvent.slice(5).trim();
      if (!payload) {
        continue;
      }

      const parsed = streamEventSchema.parse(JSON.parse(payload));

      if (parsed.event === "token") {
        const content = typeof parsed.data?.content === "string" ? parsed.data.content : "";
        onEvent({ type: "token", content });
      }

      if (parsed.event === "done") {
        const backendMessage = messageSchema.parse(parsed.data.message);
        finalSessionId = typeof parsed.data.session_id === "string" ? parsed.data.session_id : finalSessionId;
        finalMessage = toChatMessage(backendMessage);
        onEvent({ type: "done", sessionId: finalSessionId, message: finalMessage });
      }
    }
  }

  if (!finalMessage) {
    throw new Error("Streaming response ended without completion event");
  }

  return {
    sessionId: finalSessionId,
    message: finalMessage
  };
};
