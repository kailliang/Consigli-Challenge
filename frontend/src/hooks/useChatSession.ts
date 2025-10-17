import { useCallback, useMemo, useState } from "react";

import { toast } from "sonner";

import type { ChatMessage, ChatSession } from "@/lib/types";
import { queryReports } from "@/lib/api";

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

    setSession((previous) => {
      const nextMessages = [
        ...previous.messages,
        {
          id: crypto.randomUUID(),
          role: "user",
          content: trimmedPrompt,
          createdAt: new Date().toISOString()
        }
      ];

      // Keep the most recent conversation turns while the summary memory is pending.
      const trimmedMessages = nextMessages.slice(-20);

      return { ...previous, messages: trimmedMessages };
    });

    setIsStreaming(true);

    try {
      const { sessionId: serverSessionId, message } = await queryReports(trimmedPrompt, sessionId);

      setSession((previous) => {
        const combinedMessages = [...previous.messages, message].slice(-20);

        return {
          sessionId: serverSessionId,
          messages: combinedMessages
        };
      });
    } catch (error) {
      const assistantMessage: ChatMessage = {
        id: crypto.randomUUID(),
        role: "assistant",
        content: "Something went wrong while querying the analyst. Please try again shortly.",
        createdAt: new Date().toISOString()
      };

      setSession((previous) => ({
        ...previous,
        messages: [...previous.messages, assistantMessage].slice(-20)
      }));

      const message = error instanceof Error ? error.message : "Unknown error";
      toast.error(message);
    } finally {
      setIsStreaming(false);
    }
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
