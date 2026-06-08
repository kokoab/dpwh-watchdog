import { startTransition, useCallback, useEffect, useRef, useState } from "react";
import { fetchThreadMessages, fetchThreads, streamChat } from "../api/chat";

const ACTIVE_THREAD_KEY = "dpwh_watchdog_active_thread_id";
const USER_ID_KEY = "dpwh_watchdog_user_id";

function getStorage() {
  if (typeof window === "undefined") {
    return null;
  }
  return window.localStorage;
}

function getOrCreateAnonymousUserId() {
  const storage = getStorage();
  if (!storage) {
    return "anonymous-user";
  }

  const existing = storage.getItem(USER_ID_KEY);
  if (existing) {
    return existing;
  }

  const created = `anon-${crypto.randomUUID()}`;
  storage.setItem(USER_ID_KEY, created);
  return created;
}

function readStoredThreadId() {
  return getStorage()?.getItem(ACTIVE_THREAD_KEY) || null;
}

function persistThreadId(threadId) {
  const storage = getStorage();
  if (!storage) {
    return;
  }

  if (threadId) {
    storage.setItem(ACTIVE_THREAD_KEY, threadId);
  } else {
    storage.removeItem(ACTIVE_THREAD_KEY);
  }
}

function getMessageSources(message) {
  const metadata = message.message_metadata;
  const resultState = metadata?.result_state;
  if (Array.isArray(resultState?.displayed_sources)) {
    return resultState.displayed_sources;
  }
  return [];
}

function mapHistoryMessage(message) {
  const sources = getMessageSources(message);
  const content =
    message.content || (message.role === "assistant" && sources.length > 0 ? "Results ready." : "");

  return {
    id: message.id || `${message.role}-${message.created_at}`,
    role: message.role,
    content,
    sources,
    streaming: false,
    error: false,
  };
}

function extractLatestResultState(messages) {
  for (let index = messages.length - 1; index >= 0; index -= 1) {
    const resultState = messages[index]?.message_metadata?.result_state;
    if (resultState?.result_kind === "contract_set") {
      return resultState;
    }
  }
  return null;
}

export function useChat() {
  const [messages, setMessages] = useState([]);
  const [activeResult, setActiveResult] = useState(null);
  const [threads, setThreads] = useState([]);
  const [activeThreadId, setActiveThreadId] = useState(null);
  const [isStreaming, setIsStreaming] = useState(false);
  const [isLoadingThreads, setIsLoadingThreads] = useState(true);
  const userIdRef = useRef(null);
  const threadIdRef = useRef(null);
  const abortRef = useRef(null);
  const hydratedRef = useRef(false);

  if (!userIdRef.current) {
    userIdRef.current = getOrCreateAnonymousUserId();
  }

  const refreshThreads = useCallback(async () => {
    setIsLoadingThreads(true);
    try {
      const nextThreads = await fetchThreads(userIdRef.current);
      startTransition(() => {
        setThreads(nextThreads);
      });
      return nextThreads;
    } finally {
      setIsLoadingThreads(false);
    }
  }, []);

  const loadThread = useCallback(async (threadId) => {
    if (!threadId || isStreaming) {
      return;
    }

    const historyMessages = await fetchThreadMessages(threadId, userIdRef.current);
    const nextMessages = historyMessages.map(mapHistoryMessage);
    const nextResult = extractLatestResultState(historyMessages);

    threadIdRef.current = threadId;
    persistThreadId(threadId);

    startTransition(() => {
      setActiveThreadId(threadId);
      setMessages(nextMessages);
      setActiveResult(nextResult);
    });
  }, [isStreaming]);

  const startNewChat = useCallback(() => {
    if (isStreaming) {
      return;
    }

    threadIdRef.current = null;
    persistThreadId(null);

    startTransition(() => {
      setActiveThreadId(null);
      setMessages([]);
      setActiveResult(null);
    });
  }, [isStreaming]);

  const sendMessage = useCallback((text) => {
    if (isStreaming || !text.trim()) {
      return;
    }

    const content = text.trim();
    const assistantId = `assistant-${Date.now()}`;
    const userId = userIdRef.current;

    setMessages((prev) => [
      ...prev,
      { id: `user-${Date.now()}`, role: "user", content, sources: [], streaming: false },
      { id: assistantId, role: "assistant", content: "", sources: [], streaming: true },
    ]);

    setIsStreaming(true);

    const abort = streamChat(
      content,
      threadIdRef.current,
      userId,
      {
        onToken: (token) => {
          setMessages((prev) =>
            prev.map((message) =>
              message.id === assistantId
                ? { ...message, content: message.content + token }
                : message
            )
          );
        },
        onSources: (sources) => {
          setMessages((prev) =>
            prev.map((message) =>
              message.id === assistantId ? { ...message, sources } : message
            )
          );
        },
        onResultState: (resultState) => {
          setActiveResult(resultState);
        },
        onDone: async (returnedThreadId) => {
          const resolvedThreadId = returnedThreadId || threadIdRef.current;
          if (resolvedThreadId) {
            threadIdRef.current = resolvedThreadId;
            persistThreadId(resolvedThreadId);
          }

          startTransition(() => {
            setActiveThreadId(resolvedThreadId || null);
            setMessages((prev) =>
              prev.map((message) =>
                message.id === assistantId ? { ...message, streaming: false } : message
              )
            );
            setIsStreaming(false);
          });

          await refreshThreads();
        },
        onError: async (errorMessage) => {
          setMessages((prev) =>
            prev.map((message) =>
              message.id === assistantId
                ? {
                    ...message,
                    content: `Error: ${errorMessage}`,
                    streaming: false,
                    error: true,
                  }
                : message
            )
          );
          setIsStreaming(false);
          await refreshThreads();
        },
      }
    );

    abortRef.current = abort;
  }, [isStreaming, refreshThreads]);

  useEffect(() => {
    if (hydratedRef.current) {
      return;
    }
    hydratedRef.current = true;

    let cancelled = false;

    async function bootstrap() {
      const storedThreadId = readStoredThreadId();
      const nextThreads = await refreshThreads();
      if (cancelled) {
        return;
      }

      const shouldRestoreThread =
        storedThreadId &&
        nextThreads.some((thread) => thread.thread_id === storedThreadId);

      if (shouldRestoreThread) {
        await loadThread(storedThreadId);
      }
    }

    bootstrap().catch(() => {
      setIsLoadingThreads(false);
    });

    return () => {
      cancelled = true;
      abortRef.current?.();
    };
  }, [loadThread, refreshThreads]);

  return {
    messages,
    activeResult,
    threads,
    activeThreadId,
    isStreaming,
    isLoadingThreads,
    sendMessage,
    startNewChat,
    loadThread,
    refreshThreads,
  };
}
