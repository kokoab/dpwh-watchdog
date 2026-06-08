export function ContractDrawer({ contract, onClose }) {
  if (!contract) {
    return null;
  }

  const budget = Number(contract.budget || 0);
  const awardAmount = Number(contract.awardAmount || 0);
  const awardToBudgetRatio =
    budget > 0 && awardAmount > 0 ? `${((awardAmount / budget) * 100).toFixed(1)}%` : "N/A";

  const fields = [
    ["Contract ID", contract.contractId],
    ["Contractor", contract.contractor],
    ["Status", contract.status],
    ["Progress", contract.progress != null ? `${contract.progress}%` : "N/A"],
    ["Budget", contract.budget != null ? `PHP ${Number(contract.budget).toLocaleString()}` : "N/A"],
    ["Award Amount", awardAmount > 0 ? `PHP ${awardAmount.toLocaleString()}` : "N/A"],
    ["Award-to-Budget Ratio", awardToBudgetRatio],
    ["Region", contract.region],
    ["Province", contract.province],
    ["Category", contract.category],
    ["Program", contract.programName],
    ["Infra Year", contract.infraYear],
  ];

  return (
    <>
      <button className="drawer-backdrop" onClick={onClose} type="button" aria-label="Close contract detail" />

      <aside className="drawer">
        <div className="drawer__header">
          <div>
            <div className="drawer__eyebrow">Contract detail</div>
            <h2 className="drawer__title">{contract.contractId || "Selected contract"}</h2>
          </div>
          <button className="drawer__close" onClick={onClose} type="button">
            Close
          </button>
        </div>

        <div className="drawer__description">{contract.description || "No description available."}</div>

        <div className="drawer__fields">
          {fields.map(([label, value]) => (
            <div key={label} className="drawer__field">
              <div className="drawer__field-label">{label}</div>
              <div className="drawer__field-value">{value || "N/A"}</div>
            </div>
          ))}
        </div>
      </aside>
    </>
  );
}
