import { SourceChip } from "./SourceChip";

function matchSourcesForLine(line, sources) {
  const normalizedLine = String(line || "").trim().toLowerCase();
  if (!normalizedLine) {
    return [];
  }

  return sources.filter((source) => {
    const contractId = String(source.contractId || "").trim().toLowerCase();
    return contractId && normalizedLine.includes(contractId);
  });
}

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
        `${index + 1}. ${description} (${contractId})`,
        `• Contractor: ${contractor}`,
        `• Status: ${status}`,
        `• Budget: ${formatBudget(source?.budget)}`,
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

function parseContractHeader(line) {
  const text = String(line || "").trim();
  const parentheticalMatch = text.match(/^(?:(\d+)\.\s+)?(.+?)\s+\(([A-Za-z0-9_-]+)\)\s*$/);
  if (parentheticalMatch) {
    return {
      indexLabel: parentheticalMatch[1] || null,
      title: parentheticalMatch[2].trim(),
      contractId: parentheticalMatch[3].trim(),
    };
  }

  const bracketMatch = text.match(/^(?:(\d+)\.\s+)?\[([A-Za-z0-9_-]+)\]\s+(.+)$/);
  if (!bracketMatch) {
    return null;
  }

  return {
    indexLabel: bracketMatch[1] || null,
    title: bracketMatch[3].trim(),
    contractId: bracketMatch[2].trim(),
  };
}

function parseBulletLine(line) {
  const text = String(line || "").trim();
  const match = text.match(/^(?:[-*•])\s+(.+)$/);
  if (!match) {
    return null;
  }

  const content = match[1].trim();
  const labelMatch = content.match(/^([^:]+):\s*(.+)$/);
  if (!labelMatch) {
    return { label: "", value: content };
  }

  return {
    label: labelMatch[1].trim(),
    value: labelMatch[2].trim(),
  };
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

function buildLineModels(textLines, availableSources, isUser, isStreaming) {
  const matchedContractIds = new Set();
  const lineModels = [];
  let activeContractId = null;

  textLines.forEach((line, index) => {
    const displayLine = humanizeRawFilterLine(line);
    const contractHeader = parseContractHeader(displayLine);
    const bullet = parseBulletLine(displayLine);
    const headerSource = contractHeader
      ? availableSources.find(
          (source) =>
            String(source.contractId || "").trim().toLowerCase() ===
            contractHeader.contractId.toLowerCase()
        ) || null
      : null;

    if (contractHeader) {
      activeContractId = contractHeader.contractId;
    } else if (!String(line || "").trim()) {
      activeContractId = null;
    }

    const suppressLine =
      Boolean(bullet) &&
      bullet.label.toLowerCase() === "description" &&
      Boolean(activeContractId);

    const lineMatches = suppressLine
      ? []
      : !isUser &&
          !isStreaming &&
          activeContractId &&
          !matchedContractIds.has(activeContractId)
        ? availableSources.filter(
            (source) =>
              String(source.contractId || "").trim().toLowerCase() ===
              activeContractId.toLowerCase()
          )
        : !isUser && !isStreaming
          ? matchSourcesForLine(displayLine, availableSources)
          : [];

    lineMatches.forEach((source) => {
      matchedContractIds.add(source.contractId);
    });
    if (headerSource) {
      matchedContractIds.add(headerSource.contractId);
    }

    lineModels.push({
      index,
      displayLine,
      contractHeader,
      bullet,
      headerSource,
      lineMatches,
      suppressLine,
    });
  });

  return { lineModels, matchedContractIds };
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

export function MessageBubble({ message, onSourceClick }) {
  const isUser = message.role === "user";
  const availableSources = Array.isArray(message.sources) ? message.sources : [];
  const documentLinks = collectDocumentLinks(availableSources);
  const resultStateSources =
    message.resultState?.result_kind === "contract_set" ? availableSources : [];
  const contentText = String(message.content || "");
  const hasStructuredResultContent =
    resultStateSources.length > 0 &&
    (!contentText.trim() ||
      !resultStateSources.some((source) => {
        const contractId = String(source.contractId || "").trim().toLowerCase();
        return contractId && contentText.toLowerCase().includes(contractId);
      }));
  const displayContent = hasStructuredResultContent
    ? buildStructuredResultText(resultStateSources)
    : contentText;
  const textLines = displayContent.split("\n");
  const { lineModels, matchedContractIds } = buildLineModels(
    textLines,
    availableSources,
    isUser,
    message.streaming
  );

  return (
    <div className={`message-row ${isUser ? "message-row--user" : ""}`}>
      <div className={`message-bubble ${isUser ? "message-bubble--user" : ""} ${message.error ? "message-bubble--error" : ""}`}>
        {!isUser && (message.resultState || message.responseSource) ? (
          <MessageResultSummary
            result={message.resultState}
            responseSource={message.responseSource}
          />
        ) : null}

        <div className="message-bubble__text">
          {lineModels.map(({ index, displayLine, contractHeader, bullet, headerSource, lineMatches, suppressLine }) => {
            if (suppressLine) {
              return null;
            }
            return (
              <div key={`${message.id}-${index}`} className="message-bubble__line-group">
                {contractHeader ? (
                  <div className="message-bubble__contract">
                    {contractHeader.indexLabel ? (
                      <span className="message-bubble__contract-index">
                        {contractHeader.indexLabel}
                      </span>
                    ) : null}
                    <div className="message-bubble__contract-main">
                      <span className="message-bubble__contract-title">
                        {contractHeader.title}
                      </span>
                      {headerSource ? (
                        <SourceChip source={headerSource} onClick={onSourceClick} />
                      ) : (
                        <span className="message-bubble__contract-id">
                          {contractHeader.contractId}
                        </span>
                      )}
                    </div>
                  </div>
                ) : bullet ? (
                  <div className="message-bubble__bullet">
                    <span className="message-bubble__bullet-mark">•</span>
                    <span className="message-bubble__bullet-copy">
                      {bullet.label ? (
                        <>
                          <strong>{bullet.label}:</strong> {bullet.value}
                        </>
                      ) : (
                        bullet.value
                      )}
                      {message.streaming && index === textLines.length - 1 ? (
                        <span className="message-bubble__cursor">▋</span>
                      ) : null}
                      {lineMatches.length > 0 ? (
                        <span className="message-bubble__line-sources">
                          {lineMatches.map((source) => (
                            <SourceChip
                              key={`${message.id}-${source.contractId}-${index}`}
                              source={source}
                              onClick={onSourceClick}
                            />
                          ))}
                        </span>
                      ) : null}
                    </span>
                  </div>
                ) : (
                  <div className="message-bubble__line">
                    <span className="message-bubble__line-text">
                      {displayLine ? displayLine : <span className="message-bubble__line-break" />}
                      {message.streaming && index === textLines.length - 1 ? (
                        <span className="message-bubble__cursor">▋</span>
                      ) : null}
                      {lineMatches.length > 0 ? (
                        <span className="message-bubble__line-sources">
                          {lineMatches.map((source) => (
                            <SourceChip
                              key={`${message.id}-${source.contractId}-${index}`}
                              source={source}
                              onClick={onSourceClick}
                            />
                          ))}
                        </span>
                      ) : null}
                    </span>
                  </div>
                )}
              </div>
            );
          })}
        </div>

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

        {!message.streaming &&
        availableSources.length > 0 &&
        availableSources.some((source) => !matchedContractIds.has(source.contractId)) ? (
          <div className="message-bubble__sources">
            <div className="message-bubble__sources-label">Sources</div>
            <div className="message-bubble__sources-list">
              {availableSources
                .filter((source) => !matchedContractIds.has(source.contractId))
                .map((source) => (
                  <SourceChip key={source.contractId} source={source} onClick={onSourceClick} />
                ))}
            </div>
          </div>
        ) : null}
      </div>
    </div>
  );
}
