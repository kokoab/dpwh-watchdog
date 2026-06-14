import { useEffect, useRef } from "react";
import { EmptyChatState } from "./EmptyChatState";
import { MessageBubble } from "./MessageBubble";

function ChatWindowSkeleton() {
  return (
    <div className="chat-window__messages chat-window__messages--skeleton" aria-hidden="true">
      <div className="message-row">
        <div className="message-bubble message-bubble--skeleton">
          <div className="chat-window__skeleton-line chat-window__skeleton-line--wide" />
          <div className="chat-window__skeleton-line chat-window__skeleton-line--mid" />
          <div className="chat-window__skeleton-line chat-window__skeleton-line--narrow" />
        </div>
      </div>

      <div className="message-row message-row--user">
        <div className="message-bubble message-bubble--user message-bubble--skeleton message-bubble--skeleton-user">
          <div className="chat-window__skeleton-line chat-window__skeleton-line--mid" />
          <div className="chat-window__skeleton-line chat-window__skeleton-line--narrow" />
        </div>
      </div>

      <div className="message-row">
        <div className="message-bubble message-bubble--skeleton">
          <div className="chat-window__skeleton-line chat-window__skeleton-line--wide" />
          <div className="chat-window__skeleton-line chat-window__skeleton-line--mid" />
          <div className="chat-window__skeleton-line chat-window__skeleton-line--short" />
          <div className="chat-window__skeleton-chips">
            <div className="chat-window__skeleton-chip" />
            <div className="chat-window__skeleton-chip" />
            <div className="chat-window__skeleton-chip" />
          </div>
        </div>
      </div>
    </div>
  );
}

export function ChatWindow({ messages, isLoadingMessages, onSourceClick, onSuggestionClick }) {
  const bottomRef = useRef(null);

  useEffect(() => {
    if (!isLoadingMessages) {
      bottomRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
    }
    
  }, [messages, isLoadingMessages]);

  return (
    <div className="chat-window">
      <div className="chat-window__inner">
        {isLoadingMessages ? (
          <ChatWindowSkeleton />
        ) : messages.length === 0 ? (
          <EmptyChatState onSuggestionClick={onSuggestionClick} />
        ) : (
          <div className="chat-window__messages">
            {messages.map((message) => (
              <MessageBubble
                key={message.id}
                message={message}
                onSourceClick={onSourceClick}
              />
            ))}
          </div>
        )}
        <div ref={bottomRef} />
      </div>
    </div>
  );
}
