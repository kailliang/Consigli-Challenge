import { useCallback, useMemo, useState } from "react";

import { toast } from "sonner";

import type { ChatMessage, ChatSession } from "@/lib/types";
import { queryReports, queryReportsStream } from "@/lib/api";

const createInitialAssistantMessage = (): ChatMessage => ({
  id: crypto.randomUUID(),
  role: "assistant",
  content: "Hello! Upload annual reports on the Ingest tab to begin asking financial questions.",
  createdAt: new Date().toISOString()
});

const createInitialSession = (): ChatSession => ({
  sessionId: crypto.randomUUID(),
  messages: [createInitialAssistantMessage()]
});

export const useChatSession = () => {
  const [session, setSession] = useState<ChatSession>(() => createInitialSession());
  const [isStreaming, setIsStreaming] = useState(false);

  const sendMessage = useCallback(async (prompt: string) => {
    const trimmedPrompt = prompt.trim();

    if (!trimmedPrompt) {
      return;
    }

    const sessionId = session.sessionId;

    const userMessage: ChatMessage = {
      id: crypto.randomUUID(),
      role: "user",
      content: trimmedPrompt,
      createdAt: new Date().toISOString()
    };

    const assistantMessageId = crypto.randomUUID();

    setSession((previous) => {
      const nextMessages = [
        ...previous.messages,
        userMessage,
        {
          id: assistantMessageId,
          role: "assistant",
          content: "",
          createdAt: new Date().toISOString(),
          citations: []
        }
      ];

      const trimmedMessages = nextMessages.slice(-20);

      return { ...previous, messages: trimmedMessages };
    });

    setIsStreaming(true);

    const appendToken = (token: string) => {
      if (!token) {
        return;
      }
      setSession((previous) => ({
        ...previous,
        messages: previous.messages.map((message) =>
          message.id === assistantMessageId
            ? { ...message, content: `${message.content}${token}` }
            : message
        )
      }));
    };

    const finalizeAssistant = (sessionIdValue: string, message: ChatMessage) => {
      setSession((previous) => ({
        sessionId: sessionIdValue,
        messages: previous.messages.map((existing) =>
          existing.id === assistantMessageId ? { ...message, id: assistantMessageId } : existing
        )
      }));
    };

    let streamCompleted = false;

    try {
      const result = await queryReportsStream(trimmedPrompt, sessionId, (event) => {
        if (event.type === "token") {
          appendToken(event.content);
        }

        if (event.type === "done") {
          streamCompleted = true;
          finalizeAssistant(event.sessionId, event.message);
        }
      });

      if (streamCompleted) {
        setIsStreaming(false);
        return;
      }

      finalizeAssistant(result.sessionId, result.message);
    } catch (streamError) {
      try {
        const { sessionId: serverSessionId, message } = await queryReports(trimmedPrompt, sessionId);
        finalizeAssistant(serverSessionId, message);
      } catch (fallbackError) {
        const assistantMessage: ChatMessage = {
          id: assistantMessageId,
          role: "assistant",
          content: "Something went wrong while querying the analyst. Please try again shortly.",
          createdAt: new Date().toISOString()
        };

        finalizeAssistant(sessionId ?? session.sessionId, assistantMessage);

        const message = fallbackError instanceof Error ? fallbackError.message : "Unknown error";
        toast.error(message);
      } finally {
        setIsStreaming(false);
      }

      const message = streamError instanceof Error ? streamError.message : "Unknown error";
      if (!streamCompleted) {
        toast.message("Streaming not available", { description: message });
      }

      return;
    }

    setIsStreaming(false);
  }, [session.sessionId]);

  const resetSession = useCallback(() => {
    setSession(createInitialSession());
  }, []);

  const state = useMemo(
    () => ({
      session,
      isStreaming,
      sendMessage,
      resetSession
    }),
    [isStreaming, resetSession, sendMessage, session]
  );

  return state;
};
