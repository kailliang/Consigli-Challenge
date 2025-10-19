import { FormEvent, useState } from "react";

export type ChatInputProps = {
  onSubmit: (prompt: string) => Promise<void> | void;
  disabled?: boolean;
};

export const ChatInput = ({ onSubmit, disabled = false }: ChatInputProps) => {
  const [value, setValue] = useState("");

  const handleSubmit = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();

    const prompt = value.trim();

    if (!prompt || disabled) {
      return;
    }

    const previousValue = value;
    setValue("");
    try {
      await onSubmit(prompt);
    } catch (error) {
      setValue(previousValue);
      throw error;
    }
  };

  return (
    <form onSubmit={handleSubmit} className="flex flex-col gap-2 rounded-xl border border-slate-800 bg-slate-900/80 p-3 shadow-inner">
      <label htmlFor="chat-input" className="text-xs font-semibold uppercase tracking-wide text-slate-400">
        Ask about revenues, EBITDA, growth, or macro factors
      </label>
      <textarea
        id="chat-input"
        placeholder="e.g. Compare Tesla and Ford revenue growth in 2022"
        value={value}
        onChange={(event) => setValue(event.target.value)}
        rows={3}
        className="w-full resize-none rounded-lg border border-slate-700 bg-slate-950/80 px-3 py-2 text-sm text-slate-100 outline-none transition focus:border-sky-500 focus:ring-2 focus:ring-sky-500/50"
      />
      <div className="flex items-center justify-between text-xs text-slate-500">
        <span>Shift + Enter for newline</span>
        <button
          type="submit"
          disabled={disabled || value.trim().length === 0}
          className="inline-flex items-center gap-2 rounded-lg bg-sky-500 px-4 py-2 font-semibold text-slate-950 transition hover:bg-sky-400 disabled:cursor-not-allowed disabled:bg-slate-700 disabled:text-slate-400"
        >
          Send
        </button>
      </div>
    </form>
  );
};
