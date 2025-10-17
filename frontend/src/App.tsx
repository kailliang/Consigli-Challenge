import { useChatSession } from "@/hooks/useChatSession";

import { ChatLayout } from "@/components/chat/ChatLayout";

const App = () => {
  const { session, isStreaming, sendMessage, resetSession } = useChatSession();

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
          <p className="mt-3 text-sm text-slate-400">
            Citations with precise page and table references will appear here once retrieval is wired up.
          </p>
        </aside>
      </main>
      <footer className="border-t border-slate-900/60 bg-slate-950/90 py-4 text-center text-xs text-slate-500">
        Prototype build – ingestion, retrieval, and LangSmith tracing coming soon.
      </footer>
    </div>
  );
};

export default App;
