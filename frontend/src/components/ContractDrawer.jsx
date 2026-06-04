// frontend/src/components/ContractDrawer.jsx
export function ContractDrawer({ contract, onClose }) {
  if (!contract) return null;

  const fields = [
    ["Contract ID",    contract.contractId],
    ["Contractor",     contract.contractor],
    ["Status",         contract.status],
    ["Progress",       contract.progress != null ? `${contract.progress}%` : "N/A"],
    ["Budget",         contract.budget != null ? `PHP ${Number(contract.budget).toLocaleString()}` : "N/A"],
    ["Amount Paid",    contract.amountPaid != null ? `PHP ${Number(contract.amountPaid).toLocaleString()}` : "N/A"],
    ["Region",         contract.region],
    ["Province",       contract.province],
    ["Category",       contract.category],
    ["Program",        contract.programName],
    ["Infra Year",     contract.infraYear],
  ];

  return (
    <>
      {/* Backdrop */}
      <div
        onClick={onClose}
        style={{
          position: "fixed", inset: 0,
          background: "rgba(0,0,0,0.4)",
          zIndex: 40,
        }}
      />

      {/* Drawer */}
      <div style={{
        position: "fixed", top: 0, right: 0,
        height: "100vh", width: "380px",
        background: "#1e1e2e",
        borderLeft: "1px solid #333",
        zIndex: 50,
        overflowY: "auto",
        padding: "24px",
        boxSizing: "border-box",
      }}>
        <button
          onClick={onClose}
          style={{
            background: "none", border: "none",
            color: "#aaa", fontSize: "20px",
            cursor: "pointer", marginBottom: "16px",
          }}
        >
          ✕
        </button>

        <h2 style={{ color: "#fff", fontSize: "16px", marginBottom: "20px" }}>
          Contract Detail
        </h2>

        {fields.map(([label, value]) => (
          <div key={label} style={{ marginBottom: "14px" }}>
            <div style={{ color: "#888", fontSize: "11px", textTransform: "uppercase" }}>
              {label}
            </div>
            <div style={{ color: "#e2e8f0", fontSize: "14px", marginTop: "2px" }}>
              {value || "N/A"}
            </div>
          </div>
        ))}
      </div>
    </>
  );
}