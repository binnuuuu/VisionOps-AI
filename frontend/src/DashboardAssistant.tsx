import { Bot, MessageCircle, Send, Sparkles, X } from "lucide-react";
import { FormEvent, KeyboardEvent, useEffect, useRef, useState } from "react";
import { askDashboardAssistant } from "./api";
import type { Inspection } from "./types";

type DashboardAssistantProps = {
  selectedProductId?: string;
  inspection?: Inspection | null;
};

type ChatMessage = {
  id: string;
  role: "assistant" | "user";
  text: string;
};

const starterQuestions = [
  "What is happening right now?",
  "Why was the latest item rejected?",
  "How many products passed and failed in this batch?",
  "Are the camera and AI model online?"
];

const openingMessage: ChatMessage = {
  id: "welcome",
  role: "assistant",
  text: "Line context is ready. What would you like to know?"
};

export default function DashboardAssistant({ selectedProductId, inspection }: DashboardAssistantProps) {
  const [open, setOpen] = useState(false);
  const [question, setQuestion] = useState("");
  const [messages, setMessages] = useState<ChatMessage[]>([openingMessage]);
  const [suggestions, setSuggestions] = useState(starterQuestions);
  const [asking, setAsking] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);
  const transcriptRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!open) return;
    inputRef.current?.focus();
  }, [open]);

  useEffect(() => {
    transcriptRef.current?.scrollTo({ top: transcriptRef.current.scrollHeight, behavior: "smooth" });
  }, [messages, asking]);

  useEffect(() => {
    function closeOnEscape(event: globalThis.KeyboardEvent) {
      if (event.key === "Escape") setOpen(false);
    }
    window.addEventListener("keydown", closeOnEscape);
    return () => window.removeEventListener("keydown", closeOnEscape);
  }, []);

  async function ask(text: string) {
    const cleanQuestion = text.trim();
    if (!cleanQuestion || asking) return;

    setMessages((current) => [
      ...current,
      { id: `user-${Date.now()}`, role: "user", text: cleanQuestion }
    ]);
    setQuestion("");
    setAsking(true);

    try {
      const response = await askDashboardAssistant(cleanQuestion, {
        productId: selectedProductId,
        inspectionId: inspection?.id,
        batchId: inspection?.batch_id ?? undefined
      });
      setMessages((current) => [
        ...current,
        { id: `assistant-${Date.now()}`, role: "assistant", text: response.answer }
      ]);
      if (response.suggested_questions.length) setSuggestions(response.suggested_questions);
    } catch (error) {
      setMessages((current) => [
        ...current,
        {
          id: `error-${Date.now()}`,
          role: "assistant",
          text: error instanceof Error ? error.message : "I could not read the dashboard context. Please try again."
        }
      ]);
    } finally {
      setAsking(false);
    }
  }

  function submitQuestion(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    void ask(question);
  }

  function handleInputKeyDown(event: KeyboardEvent<HTMLInputElement>) {
    if (event.key === "Escape") setOpen(false);
  }

  return (
    <div className={`assistant-dock ${open ? "open" : ""}`}>
      {open && (
        <section className="assistant-panel" role="dialog" aria-label="VisionOps assistant">
          <header className="assistant-header">
            <div className="assistant-identity">
              <span className="assistant-avatar" aria-hidden="true">
                <Bot size={20} />
              </span>
              <div>
                <strong>VisionOps Assistant</strong>
                <span><i /> Live dashboard context</span>
              </div>
            </div>
            <button className="assistant-close" type="button" onClick={() => setOpen(false)} title="Close assistant">
              <X size={18} />
            </button>
          </header>

          <div className="assistant-transcript" ref={transcriptRef} aria-live="polite">
            {messages.map((message) => (
              <div key={message.id} className={`assistant-message ${message.role}`}>
                {message.role === "assistant" && <Bot size={16} aria-hidden="true" />}
                <p>{message.text}</p>
              </div>
            ))}

            {messages.length === 1 && (
              <div className="assistant-suggestions" aria-label="Sample questions">
                <span><Sparkles size={14} /> Sample questions</span>
                {suggestions.map((suggestion) => (
                  <button key={suggestion} type="button" onClick={() => void ask(suggestion)}>
                    {suggestion}
                  </button>
                ))}
              </div>
            )}

            {asking && (
              <div className="assistant-message assistant thinking" aria-label="Assistant is checking live data">
                <Bot size={16} aria-hidden="true" />
                <span className="assistant-thinking"><i /><i /><i /></span>
              </div>
            )}
          </div>

          <form className="assistant-composer" onSubmit={submitQuestion}>
            <input
              ref={inputRef}
              value={question}
              onChange={(event) => setQuestion(event.target.value)}
              onKeyDown={handleInputKeyDown}
              placeholder="Ask about this inspection..."
              aria-label="Ask VisionOps assistant"
              maxLength={500}
              disabled={asking}
            />
            <button type="submit" disabled={asking || !question.trim()} title="Send question">
              <Send size={17} />
            </button>
          </form>
        </section>
      )}

      <button
        className="assistant-launcher"
        type="button"
        onClick={() => setOpen((current) => !current)}
        aria-expanded={open}
        aria-label={open ? "Close VisionOps assistant" : "Open VisionOps assistant"}
        title={open ? "Close assistant" : "Ask VisionOps assistant"}
      >
        {open ? <X size={23} /> : <MessageCircle size={25} />}
        {!open && <Bot className="assistant-launcher-bot" size={14} aria-hidden="true" />}
      </button>
    </div>
  );
}
