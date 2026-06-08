export function SourceChip({ source, onClick }) {
  return (
    <button className="source-chip" onClick={() => onClick(source)} type="button">
      <span className="source-chip__label">{source.contractId}</span>
    </button>
  );
}
