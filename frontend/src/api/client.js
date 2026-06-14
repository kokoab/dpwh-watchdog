const BASE_URL = import.meta.env.VITE_API_BASE_URL;

function buildUrl(path) {
  return `${BASE_URL}${path}`;
}

function authHeaders(accessToken, extra = {}) {
  return {
    ...extra,
    Authorization: `Bearer ${accessToken}`,
  };
}

async function parseJsonResponse(response) {
  if (!response.ok) {
    throw new Error(`Request failed with status ${response.status}`);
  }

  return response.json();
}

async function request(path, accessToken, options = {}) {
  const response = await fetch(buildUrl(path), {
    ...options,
    headers: authHeaders(accessToken, options.headers),
  });

  return parseJsonResponse(response);
}

export const chatApi = {
  async fetchThreads(accessToken) {
    const payload = await request("/chat/threads", accessToken);
    return Array.isArray(payload.threads) ? payload.threads : [];
  },

  async fetchThreadMessages(threadId, accessToken) {
    const encodedThreadId = encodeURIComponent(threadId);
    const payload = await request(
      `/chat/threads/${encodedThreadId}/messages`,
      accessToken,
    );

    return Array.isArray(payload.messages) ? payload.messages : [];
  },

  async deleteThread(threadId, accessToken) {
    const encodedThreadId = encodeURIComponent(threadId);

    await fetch(buildUrl(`/chat/threads/${encodedThreadId}`), {
      method: "DELETE",
      headers: authHeaders(accessToken),
    }).then((response) => {
      if (!response.ok) {
        throw new Error(`Request failed with status ${response.status}`);
      }
    });

    return true;
  },
};

export function streamChat(
  message,
  threadId,
  accessToken,
  { onToken, onSources, onResultState, onDone, onError },
) {
  const controller = new AbortController();

  fetch(buildUrl("/chat/stream"), {
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

          try {
            const event = JSON.parse(part.slice(6));

            if (event.type === "token") onToken(event.content);
            if (event.type === "sources") onSources(event.content);
            if (event.type === "result_state" && onResultState) {
              onResultState(event.content);
            }
            if (event.type === "done") onDone(returnedThreadId);
            if (event.type === "error") onError(event.content);
          } catch {
            // Ignore malformed stream chunks and continue reading.
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