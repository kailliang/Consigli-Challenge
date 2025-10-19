import type { ChatMessage } from "@/lib/types";

const roleToLabel: Record<ChatMessage["role"], string> = {
  assistant: "Analyst",
  user: "You",
  system: "System"
};

type Props = {
  messages: ChatMessage[];
  isStreaming: boolean;
};

export const ChatMessageList = ({ messages, isStreaming }: Props) => {
  return (
    <div className="flex-1 space-y-3 overflow-y-auto py-4">
      {messages.map((message) => {
        const isUser = message.role === "user";
        return (
          <div
            key={message.id}
            className={`flex ${isUser ? "justify-end" : "justify-start"}`}
          >
            <article
              className={`w-full rounded-xl border border-slate-800 bg-slate-900/60 p-4 shadow-sm ${
                isUser ? "max-w-[70%] self-end" : ""
              }`}
            >
              <div className="flex items-center justify-between text-xs font-semibold uppercase tracking-wide text-slate-400">
                <span>{roleToLabel[message.role]}</span>
                <time dateTime={message.createdAt} className="text-slate-500">
                  {new Date(message.createdAt).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}
                </time>
              </div>
              <p className="mt-3 whitespace-pre-wrap text-sm leading-relaxed text-slate-100">
                {message.content}
              </p>
            </article>
          </div>
        );
      })}
      {isStreaming ? (
        <div className="animate-pulse text-sm text-slate-400">Analyst is thinking…</div>
      ) : null}
    </div>
  );
};
