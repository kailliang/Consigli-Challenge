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

const resolveWsUrl = (): string => {
  const base = new URL(env.VITE_API_BASE_URL);
  const protocol = base.protocol === "https:" ? "wss:" : "ws:";
  const path = base.pathname.endsWith("/") ? base.pathname.slice(0, -1) : base.pathname;
  return `${protocol}//${base.host}${path}/query/ws`;
};

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

export const queryReportsStream = (
  prompt: string,
  sessionId: string | undefined,
  onEvent: StreamCallback
): Promise<{ sessionId: string; message: ChatMessage }> => {
  const wsUrl = resolveWsUrl();

  return new Promise((resolve, reject) => {
    let settled = false;
    let finalSessionId = sessionId ?? "";
    let finalMessage: ChatMessage | null = null;

    const socket = new WebSocket(wsUrl);

    socket.onopen = () => {
      socket.send(JSON.stringify({ prompt, session_id: sessionId }));
    };

    socket.onmessage = (event) => {
      let parsed;
      try {
        parsed = streamEventSchema.parse(JSON.parse(event.data));
      } catch {
        return;
      }

      if (parsed.event === "token") {
        const content = typeof parsed.data?.content === "string" ? parsed.data.content : "";
        onEvent({ type: "token", content });
        return;
      }

      if (parsed.event === "done") {
        try {
          const backendMessage = messageSchema.parse(parsed.data.message);
          finalSessionId = typeof parsed.data.session_id === "string" ? parsed.data.session_id : finalSessionId;
          finalMessage = toChatMessage(backendMessage);
          onEvent({ type: "done", sessionId: finalSessionId, message: finalMessage });
          settled = true;
          socket.close(1000, "completed");
          resolve({ sessionId: finalSessionId, message: finalMessage });
        } catch (error) {
          settled = true;
          socket.close(1002, "invalid message");
          reject(error instanceof Error ? error : new Error("Invalid completion payload"));
        }
        return;
      }

      if (parsed.event === "error") {
        const detail = typeof parsed.data?.message === "string" ? parsed.data.message : "Streaming error";
        settled = true;
        socket.close(1011, "error");
        reject(new Error(detail));
      }
    };

    socket.onerror = () => {
      if (settled) {
        return;
      }
      settled = true;
      socket.close();
      reject(new Error("WebSocket connection error"));
    };

    socket.onclose = (event) => {
      if (settled) {
        return;
      }
      settled = true;
      if (finalMessage) {
        resolve({ sessionId: finalSessionId, message: finalMessage });
        return;
      }

      const reason = event.reason || "WebSocket connection closed unexpectedly";
      reject(new Error(reason));
    };
  });
};
