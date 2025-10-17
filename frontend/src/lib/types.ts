export type ChatRole = "user" | "assistant" | "system";

export type Citation = {
  id: string;
  source: string;
  page?: string;
  section?: string;
  snippet: string;
};

export type ChatMessage = {
  id: string;
  role: ChatRole;
  content: string;
  createdAt: string;
  citations?: Citation[];
};

export type ChatInputPayload = {
  prompt: string;
};

export type ChatSession = {
  sessionId: string;
  messages: ChatMessage[];
};
