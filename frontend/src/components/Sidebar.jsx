import { MoreVertical } from "lucide-react";
import { useEffect, useRef, useState } from "react";

function formatRelativeDate(value) {
  if (!value) {
    return "";
  }

  const date = new Date(value);
  if (Number.isNaN(date.getTime())) {
    return "";
  }

  return new Intl.DateTimeFormat("en-PH", {
    month: "short",
    day: "numeric",
  }).format(date);
}

function getThreadTitle(thread) {
  const rawTitle =
    thread.title ||
    thread.last_message_content ||
    thread.last_message_role ||
    "New DPWH chat";

  return rawTitle.replace(/\s+/g, " ").trim() || "New DPWH chat";
}


function KebabMenu({ thread, onDelete, disabled }) {
  const [isOpen, setIsOpen] = useState(false);
  const menuRef = useRef(null);

  useEffect(() => {
    function handleClickOutside(event) {
      if (menuRef.current && !menuRef.current.contains(event.target)) {
        setIsOpen(false);
      }
    }
    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  return (
    <div className="kebab-menu" ref={menuRef}>
      <button
        className="kebab-menu__button"
        onClick={(e) => {
          e.stopPropagation();
          setIsOpen(!isOpen);
        }}
        disabled={disabled}
        aria-label="Thread menu"
      >
        <MoreVertical size={18} />
      </button>

      {isOpen && (
        <div className="kebab-menu__dropdown">
          <button
            className="kebab-menu__item kebab-menu__item--danger"
            onClick={(e) => {
              e.stopPropagation();
              if (disabled) return;
              onDelete(thread);
              setIsOpen(false);
            }}
            disabled={disabled}
          >
            Delete
          </button>
        </div>
      )}
    </div>
  );
}

function getThreadCompactLabel(thread) {
  const title = getThreadTitle(thread);
  const words = title.split(/\s+/).filter(Boolean);
  if (words.length >= 2) {
    return `${words[0][0] || ""}${words[1][0] || ""}`.toUpperCase();
  }
  return title.slice(0, 2).toUpperCase();
}

export function Sidebar({
  threads = [],
  activeThreadId,
  isLoading,
  disabled,
  isOpen,
  isCollapsed,
  onClose,
  onToggleCollapse,
  onNewChat,
  onSelectThread,
  onDeleteThread,
}) {
  return (
    <aside className={`sidebar ${isOpen ? "sidebar--open" : ""} ${isCollapsed ? "sidebar--collapsed" : ""}`}>
      <div className="sidebar__header">
        <div className="sidebar__brand">
          <span className="sidebar__brand-mark">DP</span>
          <div className="sidebar__brand-copy">
            <div className="sidebar__brand-title">DPWH Watchdog</div>
          </div>
        </div>
        <button
          className="sidebar__collapse"
          onClick={onToggleCollapse}
          type="button"
          aria-label={isCollapsed ? "Expand sidebar" : "Collapse sidebar"}
          title={isCollapsed ? "Expand sidebar" : "Collapse sidebar"}
        >
          {isCollapsed ? "»" : "«"}
        </button>
        <button className="sidebar__close" onClick={onClose} type="button">
          Close
        </button>
      </div>

      <button
        className="sidebar__new-chat"
        disabled={disabled}
        onClick={onNewChat}
        type="button"
        title="New chat"
      >
        <span className="sidebar__new-chat-mark">+</span>
        <span className="sidebar__new-chat-label">New chat</span>
      </button>

      <div className="sidebar__section">
        <div className="sidebar__section-label">{isCollapsed ? "Chats" : "Recents"}</div>

        {isLoading ? (
          <div className="sidebar__empty">Loading recent chats...</div>
        ) : threads.length === 0 ? (
          <div className="sidebar__empty">No chats yet. Start a new one.</div>
        ) : (
          <div className="sidebar__thread-list">
            {threads.map((thread) => {
              const title = getThreadTitle(thread);
              const isActive = thread.thread_id === activeThreadId;

              return (
                <div
                  key={thread.thread_id}
                  className={`sidebar__thread ${isActive ? "sidebar__thread--active" : ""}`}
                  // If you need disabled styling, apply a CSS class manually instead of the HTML attribute
                  onClick={() => !disabled && onSelectThread(thread.thread_id)} 
                  title={title}
                  style={{ cursor: disabled ? 'not-allowed' : 'pointer' }}
                >
                  <div className="sidebar__thread-compact">{getThreadCompactLabel(thread)}</div>
                  <div className="sidebar__thread-content">
                    <div className="sidebar__thread-title">{title}</div>
                    <div className="sidebar__thread-meta">
                      <span>{formatRelativeDate(thread.updated_at)}</span>
                      <span>
                        {/* Pass down the thread object and your parent deletion handler function */}
                        <KebabMenu thread={thread} onDelete={onDeleteThread} disabled={disabled} />
                      </span>
                    </div>
                  </div>
                </div>


              );
            })}
          </div>
        )}
      </div>

      <div className="sidebar__footer">
        <div className="sidebar__footer-badge">Philippine public works</div>
        <div className="sidebar__footer-note">Light workspace for DPWH conversations</div>
      </div>
    </aside>
  );
}
