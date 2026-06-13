import { SourceChip } from "./SourceChip";

function formatFilters(filters = {}) {
  const order = ["region", "province", "status", "category", "contractor", "infra_year"];
  return order.map((key) => filters[key]).filter(Boolean);
}

export function ActiveResultBar({ result, onShowResults, onSourceClick }) {
  if (!result || result.result_kind !== "contract_set") {
    return null;
  }

  const count = Number(result.count || 0);
  const filters = formatFilters(result.filters);
  const displayedSources = Array.isArray(result.displayed_sources) ? result.displayed_sources : [];
  const countLabel = count === 1 ? "1 match" : `${count.toLocaleString()} matches`;

  return (
    <div className="active-result">
      <div className="active-result__header">
        <div className="active-result__summary">
          <div className="active-result__eyebrow">Active result</div>
          <div className="active-result__count">{countLabel}</div>
          <div className="active-result__filters">
            {filters.map((value) => (
              <span key={value} className="active-result__pill">
                {value}
              </span>
            ))}
          </div>
        </div>

        <button className="active-result__button" onClick={onShowResults} type="button">
          Show results
        </button>
      </div>

      {displayedSources.length > 0 ? (
        <div className="active-result__sources">
          {displayedSources.map((source) => (
            <SourceChip key={source.contractId} source={source} onClick={onSourceClick} />
          ))}
        </div>
      ) : null}
    </div>
  );
}
