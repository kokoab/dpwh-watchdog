// frontend/src/components/ChatWindow.jsx
import { useEffect, useRef } from "react";
import { MessageBubble } from "./MessageBubble";

export function ChatWindow({ messages, onSourceClick }) {
  const bottomRef = useRef(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  return (
    <div style={{
      flex: 1,
      overflowY: "auto",
      padding: "24px",
    }}>
      {messages.length === 0 && (
        <div style={{
          textAlign: "center", color: "#555", marginTop: "80px", fontSize: "14px"
        }}>
          Ask anything about DPWH contracts.
        </div>
      )}
      {messages.map((m, i) => (
        <MessageBubble key={i} message={m} onSourceClick={onSourceClick} />
      ))}
      <div ref={bottomRef} />
    </div>
  );
}