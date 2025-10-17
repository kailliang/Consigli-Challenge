import { z } from "zod";

import type { ChatMessage } from "@/lib/types";

const envSchema = z.object({
  VITE_API_BASE_URL: z.string().url().optional().default("http://localhost:8000/v1")
});

const env = envSchema.parse(import.meta.env);

const queryResponseSchema = z.object({
  session_id: z.string(),
  message: z.object({
    id: z.string(),
    role: z.enum(["assistant", "user", "system"]),
    content: z.string(),
    created_at: z.string(),
    citations: z
      .array(
        z.object({
          id: z.string(),
          source: z.string(),
          page: z.string().optional().nullable(),
          section: z.string().optional().nullable(),
          snippet: z.string()
        })
      )
      .optional()
  })
});

export type QueryResponse = z.infer<typeof queryResponseSchema>;

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
    message: {
      id: data.message.id,
      role: data.message.role,
      content: data.message.content,
      createdAt: new Date(data.message.created_at).toISOString(),
      citations: data.message.citations?.map((citation) => ({
        id: citation.id,
        source: citation.source,
        page: citation.page ?? undefined,
        snippet: citation.snippet,
        section: citation.section ?? undefined
      })) ?? []
    }
  };
};
