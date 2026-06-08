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

function MessageResultSummary({ result }) {
  if (!result || result.result_kind !== "contract_set") {
    return null;
  }

  const filters = formatResultFilters(result.filters);
  if (filters.length === 0) {
    return null;
  }

  return (
    <div className="message-result">
      <div className="message-result__eyebrow">RESULTS</div>
      <div className="message-result__filters">
        {filters.map((value) => (
          <span key={value} className="message-result__pill">
            {value}
          </span>
        ))}
      </div>
    </div>
  );
}

export function MessageBubble({ message, onSourceClick }) {
  const isUser = message.role === "user";
  const availableSources = Array.isArray(message.sources) ? message.sources : [];
  const textLines = String(message.content || "").split("\n");
  const matchedContractIds = new Set();
  let activeContractId = null;

  return (
    <div className={`message-row ${isUser ? "message-row--user" : ""}`}>
      <div className={`message-bubble ${isUser ? "message-bubble--user" : ""} ${message.error ? "message-bubble--error" : ""}`}>
        {!isUser && message.resultState ? (
          <MessageResultSummary
            result={message.resultState}
          />
        ) : null}

        <div className="message-bubble__text">
          {textLines.map((line, index) => {
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

            const isDescriptionLine =
              bullet && bullet.label.toLowerCase() === "description";
            const lineMatches =
              !isUser && !message.streaming && isDescriptionLine && activeContractId
              ? availableSources.filter(
                    (source) =>
                      String(source.contractId || "").trim().toLowerCase() ===
                      activeContractId.toLowerCase()
                  )
                : !isUser && !message.streaming
                  ? matchSourcesForLine(displayLine, availableSources)
                  : [];

            lineMatches.forEach((source) => {
              matchedContractIds.add(source.contractId);
            });
            if (headerSource) {
              matchedContractIds.add(headerSource.contractId);
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
                    </span>
                  </div>
                )}
              </div>
            );
          })}
        </div>

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
