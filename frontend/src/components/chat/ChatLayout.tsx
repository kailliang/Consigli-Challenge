import { ChatInput } from "./ChatInput";
import { ChatMessageList } from "./ChatMessageList";

import type { ChatMessage } from "@/lib/types";

export type ChatLayoutProps = {
  messages: ChatMessage[];
  onPromptSubmit: (prompt: string) => Promise<void> | void;
  isStreaming: boolean;
  onReset: () => void;
};

export const ChatLayout = ({ messages, onPromptSubmit, isStreaming, onReset }: ChatLayoutProps) => {
  return (
    <section className="flex h-full flex-col gap-3">
      <header className="flex items-center justify-between rounded-xl border border-slate-800 bg-slate-900/60 px-4 py-3 shadow-sm">
        <div>
          <h1 className="text-lg font-semibold text-slate-100">Annual Report Analyst</h1>
          <p className="text-xs text-slate-400">
            Retrieval pipeline coming soon. Ask a question to try the conversation flow.
          </p>
        </div>
        <button
          type="button"
          onClick={onReset}
          className="rounded-lg border border-slate-700 px-3 py-1 text-xs font-semibold text-slate-300 transition hover:border-slate-500 hover:text-slate-100"
        >
          Reset
        </button>
      </header>
      <ChatMessageList messages={messages} isStreaming={isStreaming} />
      <ChatInput onSubmit={onPromptSubmit} disabled={isStreaming} />
    </section>
  );
};
