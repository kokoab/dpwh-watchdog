import { useState } from "react";
import { ActiveResultBar } from "./components/ActiveResultBar";
import { ChatWindow } from "./components/ChatWindow";
import { ContractDrawer } from "./components/ContractDrawer";
import { InputBar } from "./components/InputBar";
import { Sidebar } from "./components/Sidebar";
import { useChat } from "./hooks/useChat";
import "./App.css";

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

function getThreadHeading(activeThreadId, threads) {
  const activeThread = threads.find((thread) => thread.thread_id === activeThreadId);
  if (!activeThread) {
    return "New chat";
  }

  const title = activeThread.title || activeThread.last_message_content || "DPWH chat";
  return title.replace(/\s+/g, " ").trim() || "DPWH chat";
}

export default function App() {
  const {
    messages,
    activeResult,
    threads,
    activeThreadId,
    isStreaming,
    isLoadingThreads,
    sendMessage,
    startNewChat,
    loadThread,
  } = useChat();
  const [selectedContract, setSelectedContract] = useState(null);
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [sidebarCollapsed, setSidebarCollapsed] = useState(() => readStoredSidebarCollapsed());

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

          <button className="chat-shell__new-chat" onClick={handleNewChat} type="button">
            New chat
          </button>
        </header>

        <div className="chat-shell__body">
          <ActiveResultBar
            result={activeResult}
            onShowResults={() => sendMessage("show them")}
            onSourceClick={setSelectedContract}
          />

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
