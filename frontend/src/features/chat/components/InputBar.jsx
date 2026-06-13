import { useState } from "react";

export function InputBar({ onSend, disabled, hasMessages }) {
  const [text, setText] = useState("");

  function handleSend() {
    if (!text.trim() || disabled) {
      return;
    }
    onSend(text.trim());
    setText("");
  }

  function handleKeyDown(event) {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      handleSend();
    }
  }

  return (
    <div className={`composer ${hasMessages ? "composer--docked" : ""}`}>
      <div className="composer__panel">
        <button className="composer__action" type="button" aria-label="New prompt option">
          +
        </button>
        <textarea
          className="composer__input"
          value={text}
          onChange={(event) => setText(event.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Ask about DPWH contracts, regions, contractors..."
          disabled={disabled}
          rows={1}
        />
        <button
          className="composer__send"
          disabled={disabled || !text.trim()}
          onClick={handleSend}
          type="button"
        >
          Send
        </button>
      </div>
    </div>
  );
}
