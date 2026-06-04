// frontend/src/hooks/useChat.js
import { useCallback, useRef, useState } from "react";
import { streamChat } from "../api/chat";

export function useChat() {
  const [messages, setMessages] = useState([]);
  const [isStreaming, setIsStreaming] = useState(false);
  const threadIdRef = useRef(null);
  const abortRef = useRef(null);

  const sendMessage = useCallback((text) => {
    if (isStreaming || !text.trim()) return;

    // Add user message
    setMessages((prev) => [...prev, { role: "user", content: text }]);

    // Add empty assistant message we'll stream into
    const assistantId = Date.now();
    setMessages((prev) => [
      ...prev,
      { id: assistantId, role: "assistant", content: "", sources: [], streaming: true },
    ]);

    setIsStreaming(true);

    const abort = streamChat(
      text,
      threadIdRef.current,
      {
        onToken: (token) => {
          setMessages((prev) =>
            prev.map((m) =>
              m.id === assistantId ? { ...m, content: m.content + token } : m
            )
          );
        },
        onSources: (sources) => {
          setMessages((prev) =>
            prev.map((m) =>
              m.id === assistantId ? { ...m, sources } : m
            )
          );
        },
        onDone: (returnedThreadId) => {
          if (returnedThreadId) threadIdRef.current = returnedThreadId;
          setMessages((prev) =>
            prev.map((m) =>
              m.id === assistantId ? { ...m, streaming: false } : m
            )
          );
          setIsStreaming(false);
        },
        onError: (err) => {
          setMessages((prev) =>
            prev.map((m) =>
              m.id === assistantId
                ? { ...m, content: `Error: ${err}`, streaming: false, error: true }
                : m
            )
          );
          setIsStreaming(false);
        },
      }
    );

    abortRef.current = abort;
  }, [isStreaming]);

  return { messages, isStreaming, sendMessage };
}