import { SourceChip } from "./SourceChip";

export function MessageBubble({ message, onSourceClick }) {
  const isUser = message.role === "user";

  return (
    <div className={`message-row ${isUser ? "message-row--user" : ""}`}>
      <div className={`message-bubble ${isUser ? "message-bubble--user" : ""} ${message.error ? "message-bubble--error" : ""}`}>
        <div className="message-bubble__text">
          {message.content}
          {message.streaming ? <span className="message-bubble__cursor">▋</span> : null}
        </div>

        {!message.streaming && message.sources?.length > 0 ? (
          <div className="message-bubble__sources">
            <div className="message-bubble__sources-label">Sources</div>
            <div className="message-bubble__sources-list">
              {message.sources.map((source) => (
                <SourceChip key={source.contractId} source={source} onClick={onSourceClick} />
              ))}
            </div>
          </div>
        ) : null}
      </div>
    </div>
  );
}
