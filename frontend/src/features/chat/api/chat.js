const BASE_URL = "";

function authHeaders(accessToken, extra = {}) {
  return {
    ...extra,
    Authorization: `Bearer ${accessToken}`
  }
}

async function parseJsonResponse(response) {
  if (!response.ok) {
    throw new Error(`Request failed with status ${response.status}`);
  }
  return response.json();
}

export async function fetchThreads(accessToken) {
  const response = await fetch(`${BASE_URL}/chat/threads`, {
    headers: authHeaders(accessToken),
  });
  const payload = await parseJsonResponse(response);
  return Array.isArray(payload.threads) ? payload.threads : [];
}

export async function fetchThreadMessages(threadId, accessToken) {
  const response = await fetch(
    `${BASE_URL}/chat/threads/${encodeURIComponent(threadId)}/messages`, {
      headers: authHeaders(accessToken)
    }
  );
  const payload = await parseJsonResponse(response);
  return Array.isArray(payload.messages) ? payload.messages : [];
}

export async function deleteThread(threadId, accessToken) {
  const response = await fetch (
    `${BASE_URL}/chat/threads/${encodeURIComponent(threadId)}`,
    {
      method: "DELETE",
      headers: authHeaders(accessToken)
    }
  );
  if (!response.ok) {
    throw new Error(`Request failed with status ${response.status}`);
  }

  return true;
}

export function streamChat(
  message,
  threadId,
  accessToken,
  { onToken, onSources, onResultState, onDone, onError }
) {
  const controller = new AbortController();

  fetch(`${BASE_URL}/chat/stream`, {
    method: "POST",
    headers: authHeaders(accessToken, { "Content-Type": "application/json" }),
    body: JSON.stringify({ message, thread_id: threadId }),
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
