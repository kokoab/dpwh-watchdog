import { useState } from "react";
import "./App.css";
import { ChatWindow } from "./components/ChatWindow";
import { ContractDrawer } from "./components/ContractDrawer";
import { InputBar } from "./components/InputBar";
import { Sidebar } from "./components/Sidebar";
import { useChat } from "./hooks/useChat";

const SIDEBAR_COLLAPSED_KEY = "dpwh_watchdog_sidebar_collapsed";

function readStoredSidebarCollapsed() {
  if (typeof window === "undefined") {
    return false;
  }
  return window.localStorage.getItem(SIDEBAR_COLLAPSED_KEY) === "true";
}

function persistSidebarCollapsed(value) {
  if (typeof window === "undefined") {
    return;
  }
  window.localStorage.setItem(SIDEBAR_COLLAPSED_KEY, String(value));
}

function humanizeFilterTitle(title) {
  const normalized = title.replace(/\s+/g, " ").trim();
  const filterText = normalized.match(/\bwhere\s+(.+?)(?:\s+are:?|$)/i)?.[1];
  if (!filterText || !/[a-z_]+=/i.test(filterText)) {
    return normalized;
  }

  const filters = {};
  for (const part of filterText.split(/\s+AND\s+|,\s*/i)) {
    const [rawKey, ...rawValue] = part.split("=");
    const key = rawKey?.trim().toLowerCase();
    const value = rawValue.join("=").trim();
    if (key && value) {
      filters[key] = value;
    }
  }

  const subject = filters.category ? `${filters.category} projects` : "Filtered contracts";
  const location = filters.province || filters.region;
  const status = filters.status ? `${filters.status} ` : "";

  if (location) {
    return `${status}${subject} in ${location}`;
  }
  return `${status}${subject}`;
}

function getThreadHeading(activeThreadId, threads) {
  const activeThread = threads.find((thread) => thread.thread_id === activeThreadId);
  if (!activeThread) {
    return "New chat";
  }

  const title = activeThread.title || activeThread.last_message_content || "DPWH chat";
  return humanizeFilterTitle(title) || "DPWH chat";
}



export default function App() {
  const {
    messages,
    threads,
    activeThreadId,
    isStreaming,
    isLoadingThreads,
    sendMessage,
    startNewChat,
    loadThread,
    removeThread, 
  } = useChat();
  const [selectedContract, setSelectedContract] = useState(null);
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [sidebarCollapsed, setSidebarCollapsed] = useState(() => readStoredSidebarCollapsed());

  async function handleDeleteThread(thread) {
    const threadId = thread?.thread_id;
    if (!threadId) return;
    // Edge case: accidental delete
    const ok = window.confirm("Delete this chat permanently?");
    if (!ok) return;
    try {
      await removeThread(threadId);
    } catch (error) {
      console.error(error);
      window.alert("Could not delete this chat. Please try again.");
      // No startNewChat, no sidebar change — hook threw before mutating state
    }
  }


  function handleNewChat() {
    setSelectedContract(null);
    startNewChat();
    setSidebarOpen(false);
  }

  function handleLoadThread(threadId) {
    setSelectedContract(null);
    loadThread(threadId);
    setSidebarOpen(false);
  }

  function handleToggleSidebar() {
    setSidebarCollapsed((previous) => {
      const next = !previous;
      persistSidebarCollapsed(next);
      return next;
    });
  }

  return (
    <div className={`app-shell ${sidebarCollapsed ? "app-shell--sidebar-collapsed" : ""}`}>
      <Sidebar
        threads={threads}
        activeThreadId={activeThreadId}
        isLoading={isLoadingThreads}
        disabled={isStreaming}
        isOpen={sidebarOpen}
        isCollapsed={sidebarCollapsed}
        onClose={() => setSidebarOpen(false)}
        onToggleCollapse={handleToggleSidebar}
        onNewChat={handleNewChat}
        onSelectThread={handleLoadThread}
        onDeleteThread={handleDeleteThread}
      />

      <button
        className={`app-overlay ${sidebarOpen ? "app-overlay--visible" : ""}`}
        onClick={() => setSidebarOpen(false)}
        type="button"
        aria-label="Close sidebar"
      />

      <main className="chat-shell">
        <header className="chat-shell__header">
          <div className="chat-shell__header-left">
            <button className="chat-shell__menu" onClick={() => setSidebarOpen(true)} type="button">
              Menu
            </button>
            <div>
              <div className="chat-shell__eyebrow">DPWH Watchdog</div>
              <div className="chat-shell__title">{getThreadHeading(activeThreadId, threads)}</div>
            </div>
          </div>
        </header>

        <div className="chat-shell__body">
          <ChatWindow
            messages={messages}
            onSourceClick={setSelectedContract}
            onSuggestionClick={sendMessage}
          />

          <InputBar onSend={sendMessage} disabled={isStreaming} hasMessages={messages.length > 0} />
        </div>
      </main>

      <ContractDrawer contract={selectedContract} onClose={() => setSelectedContract(null)} />
    </div>
  );
}
