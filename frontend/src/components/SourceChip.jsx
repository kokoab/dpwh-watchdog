// frontend/src/components/SourceChip.jsx
export function SourceChip({ source, onClick }) {
  return (
    <button
      onClick={() => onClick(source)}
      style={{
        display: "inline-flex",
        alignItems: "center",
        gap: "6px",
        padding: "4px 10px",
        borderRadius: "999px",
        border: "1px solid #3b82f6",
        background: "transparent",
        color: "#3b82f6",
        fontSize: "12px",
        cursor: "pointer",
        marginRight: "6px",
        marginTop: "6px",
      }}
    >
      📄 {source.contractId}
    </button>
  );
}