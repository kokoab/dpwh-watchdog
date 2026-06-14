import {
  startTransition,
  useCallback,
  useEffect,
  useRef,
  useState,
} from "react";
import { useAuth } from "../../auth/UseAuth";
import {
  deleteThread as deleteThreadApi,
  fetchThreadMessages,
  fetchThreads,
  streamChat,
} from "../api/chat";

function getMessageSources(message) {
  const metadata = message.message_metadata;
  const resultState = metadata?.result_state;
  if (Array.isArray(resultState?.displayed_sources)) {
    return resultState.displayed_sources;
  }
  return [];
}

function getResponseSource(message) {
  const source = message.message_metadata?.response_source;
  return typeof source === "string" && source.trim() ? source.trim() : null;
}

function mapHistoryMessage(message) {
  const sources = getMessageSources(message);
  const resultState = message.message_metadata?.result_state;
  const content = message.content || "";

  return {
    id: message.id || `${message.role}-${message.created_at}`,
    role: message.role,
    content,
    sources,
    resultState: resultState || null,
    responseSource: getResponseSource(message),
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

export function useChat({ onThreadResolved }) {
  const [messages, setMessages] = useState([]);
  const [activeResult, setActiveResult] = useState(null);
  const [threads, setThreads] = useState([]);
  const [activeThreadId, setActiveThreadId] = useState(null);
  const [isStreaming, setIsStreaming] = useState(false);
  const [isLoadingThreads, setIsLoadingThreads] = useState(true);
  const { accessToken } = useAuth();
  const threadIdRef = useRef(null);
  const abortRef = useRef(null);

  const refreshThreads = useCallback(async () => {
    if (!accessToken) return;
    setIsLoadingThreads(true);
    try {
      const nextThreads = await fetchThreads(accessToken);
      startTransition(() => {
        setThreads(nextThreads);
      });
      return nextThreads;
    } finally {
      setIsLoadingThreads(false);
    }
  }, [accessToken]);

  const loadThread = useCallback(
    async (threadId) => {
      if (!accessToken) return;
      if (!threadId || isStreaming) {
        return;
      }

      const historyMessages = await fetchThreadMessages(threadId, accessToken);
      const nextMessages = historyMessages.map(mapHistoryMessage);
      const nextResult = extractLatestResultState(historyMessages);

      threadIdRef.current = threadId;

      startTransition(() => {
        setActiveThreadId(threadId);
        setMessages(nextMessages);
        setActiveResult(nextResult);
      });
    },
    [isStreaming, accessToken],
  );

  const startNewChat = useCallback(() => {
    if (!accessToken) return;
    if (isStreaming) {
      return;
    }

    threadIdRef.current = null;

    startTransition(() => {
      setActiveThreadId(null);
      setMessages([]);
      setActiveResult(null);
    });
  }, [isStreaming, accessToken]);

  const removeThread = useCallback(
    async (threadId) => {
      if (!accessToken) return;
      if (!threadId || isStreaming) {
        return;
      }

      await deleteThreadApi(threadId, accessToken);

      if (threadId === activeThreadId) {
        startNewChat();
      }
      await refreshThreads();
    },
    [isStreaming, refreshThreads, accessToken, activeThreadId, startNewChat],
  );

  const sendMessage = useCallback(
    (text) => {
      if (!accessToken) return;
      if (isStreaming || !text.trim()) {
        return;
      }

      const content = text.trim();
      const assistantId = `assistant-${Date.now()}`;
      setMessages((prev) => [
        ...prev,
        {
          id: `user-${Date.now()}`,
          role: "user",
          content,
          sources: [],
          streaming: false,
        },
        {
          id: assistantId,
          role: "assistant",
          content: "",
          sources: [],
          streaming: true,
        },
      ]);

      setIsStreaming(true);

      const resolvedThreadId = threadIdRef.current || crypto.randomUUID();
      if (resolvedThreadId) {
        threadIdRef.current = resolvedThreadId;
        onThreadResolved?.(resolvedThreadId);
      }

      const abort = streamChat(content, resolvedThreadId, accessToken, {
        onToken: (token) => {
          setMessages((prev) =>
            prev.map((message) =>
              message.id === assistantId
                ? {
                    ...message,
                    content: message.content + token,
                    responseSource: message.responseSource || "llm",
                  }
                : message,
            ),
          );
        },
        onSources: (sources) => {
          setMessages((prev) =>
            prev.map((message) =>
              message.id === assistantId ? { ...message, sources } : message,
            ),
          );
        },
        onResultState: (resultState) => {
          setActiveResult(resultState);
          setMessages((prev) =>
            prev.map((message) =>
              message.id === assistantId
                ? {
                    ...message,
                    resultState,
                    sources: Array.isArray(resultState?.displayed_sources)
                      ? resultState.displayed_sources
                      : message.sources,
                    responseSource:
                      resultState?.result_kind === "contract_set" ||
                      resultState?.result_kind === "contract_detail" ||
                      resultState?.result_kind === "contract_compare"
                        ? "structured"
                        : message.responseSource || "llm",
                  }
                : message,
            ),
          );
        },
        onDone: async (returnedThreadId) => {
          const resolvedThreadId = returnedThreadId || threadIdRef.current;
          if (resolvedThreadId) {
            threadIdRef.current = resolvedThreadId;
            startTransition(() => {
              setActiveThreadId(resolvedThreadId)
            })
            onThreadResolved?.(resolvedThreadId);
          }

          startTransition(() => {
            setActiveThreadId(resolvedThreadId || null);
            setMessages((prev) =>
              prev.map((message) =>
                message.id === assistantId
                  ? { ...message, streaming: false }
                  : message,
              ),
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
                : message,
            ),
          );
          setIsStreaming(false);
          await refreshThreads();
        },
      });

      abortRef.current = abort;
    },
    [isStreaming, refreshThreads, accessToken, onThreadResolved],
  );

  useEffect(() => {
    let cancelled = false;

    async function bootstrap() {
      await refreshThreads();
      if (cancelled) {
        return;
      }
      setIsLoadingThreads(false);
    }

    bootstrap().catch(() => {
      setIsLoadingThreads(false);
    });

    return () => {
      cancelled = true;
      abortRef.current?.();
    };
  }, [refreshThreads]);

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
    removeThread,
  };
}
