// frontend/src/App.jsx
import { useState } from "react";
import { ActiveResultBar } from "./components/ActiveResultBar";
import { ChatWindow } from "./components/ChatWindow";
import { ContractDrawer } from "./components/ContractDrawer";
import { InputBar } from "./components/InputBar";
import { useChat } from "./hooks/useChat";

export default function App() {
  const { messages, activeResult, isStreaming, sendMessage } = useChat();
  const [selectedContract, setSelectedContract] = useState(null);

  return (
    <div style={{
      display: "flex", flexDirection: "column",
      height: "100vh",
      background: "#13131f",
      fontFamily: "system-ui, sans-serif",
    }}>
      {/* Header */}
      <div style={{
        padding: "16px 24px",
        borderBottom: "1px solid #2a2a3e",
        color: "#f1f5f9",
        fontWeight: "600",
        fontSize: "16px",
      }}>
        DPWH Watchdog
      </div>

      <ActiveResultBar
        result={activeResult}
        onShowResults={() => sendMessage("show them")}
        onSourceClick={setSelectedContract}
      />
      
      <ChatWindow messages={messages} onSourceClick={setSelectedContract} />
      <InputBar onSend={sendMessage} disabled={isStreaming} />

      <ContractDrawer
        contract={selectedContract}
        onClose={() => setSelectedContract(null)}
      />
    </div>
  );
}
