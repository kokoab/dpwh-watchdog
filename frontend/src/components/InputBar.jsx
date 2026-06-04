// frontend/src/components/InputBar.jsx
import { useState } from "react";

export function InputBar({ onSend, disabled }) {
  const [text, setText] = useState("");

  const handleSend = () => {
    if (!text.trim() || disabled) return;
    onSend(text.trim());
    setText("");
  };

  return (
    <div style={{
      display: "flex", gap: "10px",
      padding: "16px 24px",
      borderTop: "1px solid #2a2a3e",
      background: "#13131f",
    }}>
      <input
        value={text}
        onChange={(e) => setText(e.target.value)}
        onKeyDown={(e) => e.key === "Enter" && handleSend()}
        placeholder="Ask about contracts, regions, contractors..."
        disabled={disabled}
        style={{
          flex: 1, padding: "10px 14px",
          borderRadius: "8px",
          border: "1px solid #333",
          background: "#1e1e2e",
          color: "#f1f5f9",
          fontSize: "14px",
          outline: "none",
        }}
      />
      <button
        onClick={handleSend}
        disabled={disabled}
        style={{
          padding: "10px 20px",
          borderRadius: "8px",
          border: "none",
          background: disabled ? "#334" : "#3b82f6",
          color: "#fff",
          cursor: disabled ? "not-allowed" : "pointer",
          fontSize: "14px",
        }}
      >
        Send
      </button>
    </div>
  );
}