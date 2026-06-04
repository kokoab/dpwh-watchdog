// frontend/src/components/MessageBubble.jsx
import { SourceChip } from "./SourceChip";

export function MessageBubble({ message, onSourceClick }) {
  const isUser = message.role === "user";

  return (
    <div style={{
      display: "flex",
      justifyContent: isUser ? "flex-end" : "flex-start",
      marginBottom: "16px",
    }}>
      <div style={{
        maxWidth: "70%",
        background: isUser ? "#3b82f6" : "#2a2a3e",
        color: "#f1f5f9",
        borderRadius: isUser ? "18px 18px 4px 18px" : "18px 18px 18px 4px",
        padding: "12px 16px",
        fontSize: "14px",
        lineHeight: "1.6",
        border: message.error ? "1px solid #ef4444" : "none",
      }}>
        {/* Message text */}
        <div style={{ whiteSpace: "pre-wrap" }}>
          {message.content}
          {message.streaming && (
            <span style={{ opacity: 0.5, animation: "pulse 1s infinite" }}>▋</span>
          )}
        </div>

        {/* Source chips — only show when streaming is done */}
        {!message.streaming && message.sources?.length > 0 && (
          <div style={{ marginTop: "10px", borderTop: "1px solid #ffffff22", paddingTop: "10px" }}>
            <div style={{ color: "#94a3b8", fontSize: "11px", marginBottom: "4px" }}>
              SOURCES
            </div>
            {message.sources.map((s) => (
              <SourceChip key={s.contractId} source={s} onClick={onSourceClick} />
            ))}
          </div>
        )}
      </div>
    </div>
  );
}