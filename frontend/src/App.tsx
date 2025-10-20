import { useChatSession } from "@/hooks/useChatSession";

import { ChatLayout } from "@/components/chat/ChatLayout";

const App = () => {
  const { session, isStreaming, sendMessage, resetSession } = useChatSession();

  return (
    <div className="flex min-h-screen flex-col bg-slate-950 text-slate-100">
      <main className="mx-auto flex w-full max-w-4xl flex-1 flex-col gap-6 px-4 py-6">
        <ChatLayout
          messages={session.messages}
          isStreaming={isStreaming}
          onPromptSubmit={sendMessage}
          onReset={resetSession}
        />
      </main>
      <footer className="border-t border-slate-900/60 bg-slate-950/90 py-4 text-center text-xs text-slate-500">
        Prototype build – ingestion, retrieval, and LangSmith tracing now available.
      </footer>
    </div>
  );
};

export default App;
