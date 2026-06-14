import { useEffect, useState } from "react";
import { Navigate, Route, Routes, useNavigate, useParams } from "react-router-dom";
import "./App.css";
import { LoginPage } from "./features/auth/LoginPage";
import { ProtectedRoute } from "./features/auth/ProtectedRoute";
import { ChatWindow } from "./features/chat/components/ChatWindow";
import { ContractDrawer } from "./features/chat/components/ContractDrawer";
import { InputBar } from "./features/chat/components/InputBar";
import { Sidebar } from "./features/chat/components/Sidebar";
import { useChat } from "./features/chat/hooks/useChat";

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

function ChatPage() {
  const navigate = useNavigate();
  const { threadId } = useParams();

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
  } = useChat({
    onThreadResolved: (nextThreadId) => {
      navigate(`/chat/${nextThreadId}`, { replace: true });
    },
  });

  const [selectedContract, setSelectedContract] = useState(null);
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);

  useEffect(() => {
    if (threadId) {
      void loadThread(threadId);
      return;
    }

    startNewChat();
  }, [threadId, loadThread, startNewChat]);

  async function handleDeleteThread(thread) {
    const nextThreadId = thread?.thread_id;
    if (!nextThreadId) return;

    const ok = window.confirm("Delete this chat permanently?");
    if (!ok) return;

    try {
      await removeThread(nextThreadId);

      if (nextThreadId === activeThreadId && threadId === nextThreadId) {
        navigate("/chat", { replace: true });
      }
    } catch (error) {
      console.error(error);
      window.alert("Could not delete this chat. Please try again.");
    }
  }

  function handleNewChat() {
    setSelectedContract(null);
    startNewChat();
    navigate("/chat");
    setSidebarOpen(false);
  }

  function handleLoadThread(nextThreadId) {
    setSelectedContract(null);
    navigate(`/chat/${nextThreadId}`);
    setSidebarOpen(false);
  }

  function handleToggleSidebar() {
    setSidebarCollapsed((previous) => !previous);
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

export default function App() {
  return (
    <Routes>
      <Route path="/" element={<Navigate to="/login" replace />} />
      <Route path="/login" element={<LoginPage />} />

      <Route
        path="/chat"
        element={
          <ProtectedRoute>
            <ChatPage/>
          </ProtectedRoute>

        }
      />
      <Route
        path="/chat/:threadId"
        element={
          <ProtectedRoute>
            <ChatPage/>
          </ProtectedRoute>
        }
      />

      <Route path="*" element={<Navigate to="/chat" replace />} />
    </Routes>
  );
}