const BASE_URL = "http://localhost:8000";

export function streamChat(message, threadId, {onToken, onSources, onDone, onError}) {
    const controller = new AbortController();

    fetch(`${BASE_URL}/chat/stream`, {
        method: "POST",
        headers: {"Contemt-Type": "application/json"},
        body: JSON.stringify({message, threadId: threadId}),
        signal: controller.signal
    }).then(async(res) => {
        const returnedThreadId = res.headers.get("X-Thread-Id");
        if (returnedThreadId) onToken.__threadId = returnedThreadId;

        const reader = res.body.getReader();
        const decoder = new TextDecoder();
        let buffer = "";

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            buffer += decoder.decode(value, {stream: true});

            const parts = buffer.split("\n\n");
            buffer = parts.pop();

            for (const part of parts) {
                if (!part.startsWith("data: ")) continue;
                const json = part.slice(6);
                try {
                    const event = JSON.parse(json);
                    if (event.type === "token") onToken(event.content);
                    if (event.type === "sources") onSources(event.content);
                    if (event.type === "done") onDone(returnedThreadId);
                    if (event.type === "error") onError(event.content);
                } catch {

                }
            }
        }
    }).catch((err) => {
        if (err.name !== "AbortError") onError(err.message);
    });
    return () => controller.abort()
}