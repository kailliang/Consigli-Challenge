import { useMemo } from "react";

import { useChatSession } from "@/hooks/useChatSession";

import { ChatLayout } from "@/components/chat/ChatLayout";

const App = () => {
  const { session, isStreaming, sendMessage, resetSession } = useChatSession();

  const latestCitations = useMemo(() => {
    const reversed = [...session.messages].reverse();
    const citingMessage = reversed.find(
      (message) => message.role === "assistant" && message.citations && message.citations.length > 0
    );
    return citingMessage?.citations ?? [];
  }, [session.messages]);

  return (
    <div className="flex min-h-screen flex-col bg-slate-950 text-slate-100">
      <main className="mx-auto flex w-full max-w-6xl flex-1 flex-col gap-6 px-4 py-6 md:flex-row">
        <div className="flex-1">
          <ChatLayout
            messages={session.messages}
            isStreaming={isStreaming}
            onPromptSubmit={sendMessage}
            onReset={resetSession}
          />
        </div>
        <aside className="w-full shrink-0 rounded-xl border border-slate-800 bg-slate-900/40 p-4 shadow-sm md:w-80">
          <h2 className="text-sm font-semibold uppercase tracking-wide text-slate-400">Citations</h2>
          {latestCitations.length === 0 ? (
            <p className="mt-3 text-sm text-slate-400">
              Citations will appear here when the analyst references specific tables or passages.
            </p>
          ) : (
            <ul className="mt-3 space-y-2">
              {latestCitations.map((citation) => (
                <li key={citation.id} className="rounded-lg border border-slate-800 bg-slate-950/70 p-3 text-xs text-slate-300">
                  <div className="font-semibold text-slate-200">{citation.source}</div>
                  <div className="mt-1 space-x-2 text-slate-400">
                    {citation.page ? <span>p. {citation.page}</span> : null}
                    {citation.section ? <span>{citation.section}</span> : null}
                  </div>
                  <p className="mt-2 text-slate-400">“{citation.snippet}”</p>
                </li>
              ))}
            </ul>
          )}
        </aside>
      </main>
      <footer className="border-t border-slate-900/60 bg-slate-950/90 py-4 text-center text-xs text-slate-500">
        Prototype build – ingestion, retrieval, analysis and LangSmith tracing.
      </footer>
    </div>
  );
};

export default App;
