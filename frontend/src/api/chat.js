const BASE_URL = "";

async function parseJsonResponse(response) {
  if (!response.ok) {
    throw new Error(`Request failed with status ${response.status}`);
  }
  return response.json();
}

export async function fetchThreads(userId) {
  const params = new URLSearchParams();
  if (userId) {
    params.set("user_id", userId);
  }

  const response = await fetch(`${BASE_URL}/chat/threads?${params.toString()}`);
  const payload = await parseJsonResponse(response);
  return Array.isArray(payload.threads) ? payload.threads : [];
}

export async function fetchThreadMessages(threadId, userId) {
  const params = new URLSearchParams();
  if (userId) {
    params.set("user_id", userId);
  }

  const response = await fetch(
    `${BASE_URL}/chat/threads/${encodeURIComponent(threadId)}/messages?${params.toString()}`
  );
  const payload = await parseJsonResponse(response);
  return Array.isArray(payload.messages) ? payload.messages : [];
}

export function streamChat(
  message,
  threadId,
  userId,
  { onToken, onSources, onResultState, onDone, onError }
) {
  const controller = new AbortController();

  fetch(`${BASE_URL}/chat/stream`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ message, thread_id: threadId, user_id: userId }),
    signal: controller.signal,
  })
    .then(async (response) => {
      if (!response.ok || !response.body) {
        throw new Error(`Request failed with status ${response.status}`);
      }

      const returnedThreadId = response.headers.get("X-Thread-Id");
      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) {
          break;
        }

        buffer += decoder.decode(value, { stream: true });

        const parts = buffer.split("\n\n");
        buffer = parts.pop() || "";

        for (const part of parts) {
          if (!part.startsWith("data: ")) {
            continue;
          }

          const rawJson = part.slice(6);
          try {
            const event = JSON.parse(rawJson);
            if (event.type === "token") {
              onToken(event.content);
            }
            if (event.type === "sources") {
              onSources(event.content);
            }
            if (event.type === "result_state" && onResultState) {
              onResultState(event.content);
            }
            if (event.type === "done") {
              onDone(returnedThreadId);
            }
            if (event.type === "error") {
              onError(event.content);
            }
          } catch {
            // Ignore malformed SSE chunks and keep streaming.
          }
        }
      }
    })
    .catch((error) => {
      if (error.name !== "AbortError") {
        onError(error.message);
      }
    });

  return () => controller.abort();
}
