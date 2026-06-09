import { SourceChip } from "./SourceChip";

function formatBudget(value) {
  if (value == null || value === "") {
    return "N/A";
  }
  const amount = Number(value);
  if (Number.isFinite(amount)) {
    return `PHP ${amount.toLocaleString()}`;
  }
  return "N/A";
}

function collectDocumentLinks(sources) {
  const links = [];
  sources.forEach((source) => {
    const documentLinks = source?.documentLinks;
    if (!documentLinks || typeof documentLinks !== "object") {
      return;
    }

    Object.entries(documentLinks).forEach(([label, url]) => {
      const trimmedUrl = String(url || "").trim();
      if (!trimmedUrl) {
        return;
      }

      links.push({
        contractId: String(source?.contractId || "").trim(),
        label: String(label || "Document").trim(),
        url: trimmedUrl,
      });
    });
  });
  return links;
}

function buildStructuredResultText(sources) {
  return sources
    .map((source, index) => {
      const description = String(source?.description || "N/A").trim();
      const contractId = String(source?.contractId || "N/A").trim();
      const contractor = String(source?.contractor || "N/A").trim();
      const status = String(source?.status || "N/A").trim();

      return [
        `${index + 1}. **[${contractId}]** ${description}`,
        `- **Contractor:** ${contractor}`,
        `- **Status:** ${status}`,
        `- **Budget:** ${formatBudget(source?.budget)}`,
      ].join("\n");
    })
    .join("\n\n");
}

function humanizeRawFilterLine(line) {
  const text = String(line || "");
  const filterText = text.match(/\bwhere\s+(.+?)(?:\s+are:?|$)/i)?.[1];
  if (!filterText || !/[a-z_]+=/i.test(filterText)) {
    return text;
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

  const subject = filters.category ? `${filters.category} projects` : "contracts";
  const location = filters.province || filters.region;
  return location
    ? `The matching ${subject} in ${location} are:`
    : `The matching ${subject} are:`;
}

function formatResultFilters(filters = {}) {
  const order = ["region", "province", "status", "category", "contractor", "infra_year"];
  return order.map((key) => filters[key]).filter(Boolean);
}

function formatResponseSource(responseSource) {
  if (responseSource === "structured") {
    return "Structured";
  }
  if (responseSource === "llm") {
    return "LLM";
  }
  return null;
}

function normalizeMarkdownText(text) {
  return String(text || "")
    .split("\n")
    .map((line) => humanizeRawFilterLine(line))
    .join("\n");
}

function renderInlineMarkdown(text, keyPrefix) {
  const parts = [];
  const pattern = /\*\*(.+?)\*\*/g;
  let lastIndex = 0;
  let match;
  let index = 0;

  while ((match = pattern.exec(text)) !== null) {
    if (match.index > lastIndex) {
      parts.push(text.slice(lastIndex, match.index));
    }
    parts.push(
      <strong key={`${keyPrefix}-strong-${index}`}>
        {match[1]}
      </strong>
    );
    lastIndex = pattern.lastIndex;
    index += 1;
  }

  if (lastIndex < text.length) {
    parts.push(text.slice(lastIndex));
  }

  return parts.length > 0 ? parts : text;
}

function parseMarkdownBlocks(text) {
  const blocks = [];
  const lines = normalizeMarkdownText(text).split("\n");
  let paragraphLines = [];
  let listBlock = null;

  function flushParagraph() {
    if (paragraphLines.length > 0) {
      blocks.push({ type: "p", lines: paragraphLines });
      paragraphLines = [];
    }
  }

  function flushList() {
    if (listBlock && listBlock.items.length > 0) {
      blocks.push(listBlock);
    }
    listBlock = null;
  }

  lines.forEach((line) => {
    const trimmed = line.trim();
    if (!trimmed) {
      flushParagraph();
      flushList();
      return;
    }

    const orderedMatch = trimmed.match(/^(\d+)\.\s+(.+)$/);
    const unorderedMatch = trimmed.match(/^(?:[-*•])\s+(.+)$/);

    if (orderedMatch) {
      flushParagraph();
      if (!listBlock || listBlock.type !== "ol") {
        flushList();
        listBlock = {
          type: "ol",
          start: Number(orderedMatch[1]) || 1,
          items: [],
        };
      }
      listBlock.items.push(orderedMatch[2]);
      return;
    }

    if (unorderedMatch) {
      flushParagraph();
      if (!listBlock || listBlock.type !== "ul") {
        flushList();
        listBlock = { type: "ul", items: [] };
      }
      listBlock.items.push(unorderedMatch[1]);
      return;
    }

    flushList();
    paragraphLines.push(trimmed);
  });

  flushParagraph();
  flushList();
  return blocks;
}

function MarkdownContent({ messageId, text, isStreaming }) {
  const blocks = parseMarkdownBlocks(text);

  return (
    <div className="message-bubble__text synthesis-content">
      {blocks.map((block, blockIndex) => {
        if (block.type === "p") {
          return (
            <p key={`${messageId}-p-${blockIndex}`}>
              {block.lines.map((line, lineIndex) => (
                <span key={`${messageId}-p-${blockIndex}-${lineIndex}`}>
                  {renderInlineMarkdown(line, `${messageId}-p-${blockIndex}-${lineIndex}`)}
                  {lineIndex < block.lines.length - 1 ? <br /> : null}
                </span>
              ))}
            </p>
          );
        }

        const ListTag = block.type === "ol" ? "ol" : "ul";
        const listProps = block.type === "ol" ? { start: block.start } : {};
        return (
          <ListTag
            key={`${messageId}-${block.type}-${blockIndex}`}
            className={`synthesis-content__list synthesis-content__list--${block.type}`}
            {...listProps}
          >
            {block.items.map((item, itemIndex) => (
              <li key={`${messageId}-${block.type}-${blockIndex}-${itemIndex}`}>
                {renderInlineMarkdown(item, `${messageId}-${block.type}-${blockIndex}-${itemIndex}`)}
              </li>
            ))}
          </ListTag>
        );
      })}
      {isStreaming ? <span className="message-bubble__cursor">▋</span> : null}
    </div>
  );
}

function MessageResultSummary({ result, responseSource }) {
  if (!result && !responseSource) {
    return null;
  }

  const resultKind = result?.result_kind || "";
  const filters = resultKind === "contract_set" ? formatResultFilters(result.filters) : [];
  const responseSourceLabel = formatResponseSource(responseSource);
  const eyebrow = resultKind === "contract_set" ? "RESULTS" : resultKind === "contract_detail" ? "DETAILS" : "RESPONSE";

  if (filters.length === 0 && !responseSourceLabel) {
    return null;
  }

  return (
    <div className="message-result">
      <div className="message-result__eyebrow">{eyebrow}</div>
      <div className="message-result__filters">
        {filters.map((value) => (
          <span key={value} className="message-result__pill">
            {value}
          </span>
        ))}
        {responseSourceLabel ? (
          <span
            className={`message-result__pill message-result__pill--source message-result__pill--${responseSource}`}
          >
            {responseSourceLabel}
          </span>
        ) : null}
      </div>
    </div>
  );
}

function MessageSources({ sources, onSourceClick }) {
  if (sources.length === 0) {
    return null;
  }

  return (
    <div className="message-bubble__sources">
      <div className="message-bubble__sources-label">Sources</div>
      <div className="message-bubble__sources-list">
        {sources.map((source, index) => (
          <SourceChip
            key={`${source.contractId || "source"}-${index}`}
            source={source}
            onClick={onSourceClick}
          />
        ))}
      </div>
    </div>
  );
}

export function MessageBubble({ message, onSourceClick }) {
  const isUser = message.role === "user";
  const availableSources = Array.isArray(message.sources)
    ? message.sources.filter((source) => source?.contractId)
    : [];
  const documentLinks = collectDocumentLinks(availableSources);
  const resultStateSources =
    message.resultState?.result_kind === "contract_set" ? availableSources : [];
  const contentText = String(message.content || "");
  const shouldUseStructuredFallback =
    !message.streaming && resultStateSources.length > 0 && !contentText.trim();
  const displayContent = shouldUseStructuredFallback
    ? buildStructuredResultText(resultStateSources)
    : contentText;

  return (
    <div className={`message-row ${isUser ? "message-row--user" : ""}`}>
      <div className={`message-bubble ${isUser ? "message-bubble--user" : ""} ${message.error ? "message-bubble--error" : ""}`}>
        {!isUser && (message.resultState || message.responseSource) ? (
          <MessageResultSummary
            result={message.resultState}
            responseSource={message.responseSource}
          />
        ) : null}

        <MarkdownContent
          messageId={message.id}
          text={displayContent}
          isStreaming={Boolean(message.streaming)}
        />

        {!isUser && !message.streaming ? (
          <MessageSources sources={availableSources} onSourceClick={onSourceClick} />
        ) : null}

        {!isUser && !message.streaming && documentLinks.length > 0 ? (
          <div className="message-bubble__documents">
            <div className="message-bubble__documents-label">Document links</div>
            <div className="message-bubble__documents-list">
              {documentLinks.map((link, index) => (
                <a
                  key={`${message.id}-${link.contractId}-${link.label}-${index}`}
                  className="message-bubble__document-link"
                  href={link.url}
                  target="_blank"
                  rel="noreferrer"
                >
                  <span className="message-bubble__document-link-label">{link.label}</span>
                  {link.contractId ? (
                    <span className="message-bubble__document-link-contract">{link.contractId}</span>
                  ) : null}
                </a>
              ))}
            </div>
          </div>
        ) : null}
      </div>
    </div>
  );
}
