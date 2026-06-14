import { useEffect, useRef } from "react";
import { EmptyChatState } from "./EmptyChatState";
import { MessageBubble } from "./MessageBubble";

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
          <div className="chat-window__loading">
            <span className="chat-window__loading-text">Loading chat...</span>
          </div>
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
