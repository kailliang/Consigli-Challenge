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
    <div className="flex-1 overflow-y-auto space-y-3 py-4">
      {messages.map((message) => (
        <article
          key={message.id}
          className="rounded-xl border border-slate-800 bg-slate-900/60 p-4 shadow-sm"
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
          {message.citations && message.citations.length > 0 ? (
            <ul className="mt-3 space-y-1 text-xs text-slate-400">
              {message.citations.map((citation) => (
                <li key={citation.id} className="rounded bg-slate-950/50 px-3 py-2">
                  <span className="font-semibold text-slate-300">{citation.source}</span>
                  {citation.page ? <span className="ml-2">p. {citation.page}</span> : null}
                  {citation.section ? <span className="ml-2">{citation.section}</span> : null}
                  <div className="mt-1 text-slate-500">“{citation.snippet}”</div>
                </li>
              ))}
            </ul>
          ) : null}
        </article>
      ))}
      {isStreaming ? (
        <div className="animate-pulse text-sm text-slate-400">Analyst is thinking…</div>
      ) : null}
    </div>
  );
};
