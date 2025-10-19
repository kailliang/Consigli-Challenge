import type { ChatMessage } from "@/lib/types";

type RetrievalSelection = {
  chunkId?: string;
  chunkText?: string;
  query?: string;
  score?: number;
};

type RetrievalMetadata = {
  used?: boolean;
  gatingReason?: string;
  queries?: string[];
  selection?: RetrievalSelection[];
  error?: string;
};

const parseRetrievalMetadata = (metadata?: Record<string, unknown>): RetrievalMetadata | null => {
  if (!metadata || typeof metadata !== "object") {
    return null;
  }

  const retrieval = Reflect.get(metadata, "retrieval");
  if (!retrieval || typeof retrieval !== "object") {
    return null;
  }

  const record = retrieval as Record<string, unknown>;

  const used = typeof record.used === "boolean" ? record.used : undefined;
  const gatingReason = typeof record.gating_reason === "string" ? record.gating_reason : undefined;
  const error = typeof record.error === "string" ? record.error : undefined;

  const queries = Array.isArray(record.queries)
    ? record.queries.filter((entry): entry is string => typeof entry === "string")
    : undefined;

  const selection = Array.isArray(record.selection)
    ? (record.selection
        .map((item) => {
          if (!item || typeof item !== "object") {
            return null;
          }
          const selectionRecord = item as Record<string, unknown>;
          const chunkId = typeof selectionRecord.chunk_id === "string" ? selectionRecord.chunk_id : undefined;
          const chunkText = typeof selectionRecord.chunk_text === "string" ? selectionRecord.chunk_text : undefined;
          const query = typeof selectionRecord.query === "string" ? selectionRecord.query : undefined;
          const score = typeof selectionRecord.score === "number" ? selectionRecord.score : undefined;
          return { chunkId, chunkText, query, score };
        })
        .filter((entry): entry is RetrievalSelection => entry !== null))
    : undefined;

  return { used, gatingReason, queries, selection, error };
};

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
        const retrieval = message.role === "assistant" ? parseRetrievalMetadata(message.metadata as Record<string, unknown> | undefined) : null;
        const selectionCount = retrieval?.selection?.length ?? 0;

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
              {retrieval ? (
                <div className="mt-3 space-y-2 rounded-lg border border-slate-800 bg-slate-950/60 p-3 text-xs text-slate-400">
                  <div className="flex items-center justify-between font-semibold uppercase tracking-wide text-slate-500">
                    <span>{retrieval.used === false ? "Retrieval Skipped" : "Retrieval Context"}</span>
                    {retrieval.error ? (
                      <span className="text-amber-400">{retrieval.error.split("_").join(" ")}</span>
                    ) : null}
                  </div>
                  {retrieval.used === false ? (
                    <p className="text-slate-400">
                      {retrieval.gatingReason ?? "Router determined no database lookup was required."}
                    </p>
                  ) : (
                    <>
                      <p className="text-slate-400">
                        {selectionCount > 0
                          ? `Used ${selectionCount} retrieved chunk${selectionCount === 1 ? "" : "s"}.`
                          : "No context chunks were selected."}
                      </p>
                      {retrieval.selection && retrieval.selection.length > 0 ? (
                        <div>
                          <div className="text-[10px] font-semibold uppercase tracking-wide text-slate-500">
                            Selected Chunks
                          </div>
                          <ul className="mt-1 space-y-2 text-slate-400">
                            {retrieval.selection.slice(0, 3).map((entry, index) => (
                              <li key={`${message.id}-selection-${index}`} className="leading-snug">
                                {entry.chunkText ? (
                                  <div className="whitespace-pre-wrap text-slate-300">{entry.chunkText}</div>
                                ) : (
                                  <span className="font-medium text-slate-300">{entry.chunkId ?? "unknown"}</span>
                                )}
                                {typeof entry.score === "number" ? (
                                  <span className="ml-2 text-slate-500">score {entry.score.toFixed(2)}</span>
                                ) : null}
                                {entry.query ? (
                                  <div className="text-[10px] text-slate-500">via: {entry.query}</div>
                                ) : null}
                              </li>
                            ))}
                          </ul>
                        </div>
                      ) : null}
                    </>
                  )}
                </div>
              ) : null}
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
