import { SourceChip } from "./SourceChip";

function formatFilters(filters = {}) {
  const order = ["region", "province", "status", "category", "contractor", "infra_year"];
  return order
    .map((key) => filters[key])
    .filter(Boolean);
}

export function ActiveResultBar({ result, onShowResults, onSourceClick }) {
  if (!result || result.result_kind !== "contract_set") return null;

  const count = Number(result.count || 0);
  const filters = formatFilters(result.filters);
  const displayedSources = Array.isArray(result.displayed_sources) ? result.displayed_sources : [];
  const countLabel = count === 1 ? "1 match" : `${count.toLocaleString()} matches`;

  return (
    <div style={{
      padding: "12px 24px",
      borderBottom: "1px solid #2a2a3e",
      background: "#171827",
      color: "#dbe4f0",
    }}>
      <div style={{
        display: "flex",
        gap: "12px",
        alignItems: "center",
        justifyContent: "space-between",
        flexWrap: "wrap",
      }}>
        <div style={{
          display: "flex",
          gap: "8px",
          alignItems: "center",
          flexWrap: "wrap",
          minWidth: 0,
        }}>
          <div style={{
            fontSize: "12px",
            fontWeight: 700,
            letterSpacing: "0.04em",
            textTransform: "uppercase",
            color: "#8fa3bf",
          }}>
            Active Result
          </div>
          <div style={{ fontSize: "14px", color: "#f8fafc" }}>
            {countLabel}
          </div>
          {filters.map((value) => (
            <div
              key={value}
              style={{
                padding: "3px 8px",
                borderRadius: "999px",
                border: "1px solid #314158",
                fontSize: "12px",
                color: "#bfd1e8",
              }}
            >
              {value}
            </div>
          ))}
        </div>

        <button
          onClick={onShowResults}
          style={{
            border: "1px solid #4f8cff",
            background: "transparent",
            color: "#9ec1ff",
            borderRadius: "8px",
            padding: "7px 12px",
            fontSize: "13px",
            cursor: "pointer",
          }}
        >
          Show results
        </button>
      </div>

      {displayedSources.length > 0 && (
        <div style={{ marginTop: "10px" }}>
          {displayedSources.map((source) => (
            <SourceChip key={source.contractId} source={source} onClick={onSourceClick} />
          ))}
        </div>
      )}
    </div>
  );
}
