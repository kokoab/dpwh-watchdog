function formatMoney(value) {
  if (value == null || value === "") {
    return "N/A";
  }
  const amount = Number(value);
  if (Number.isFinite(amount)) {
    return `PHP ${amount.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
  }
  return String(value);
}

function formatPercent(value) {
  if (value == null || value === "") {
    return "N/A";
  }
  const amount = Number(value);
  if (Number.isFinite(amount)) {
    return `${amount.toFixed(1)}%`;
  }
  return String(value);
}

function formatValue(value) {
  if (value == null || value === "") {
    return "N/A";
  }
  return String(value);
}

function formatDbField(key, value) {
  if (value == null || value === "") {
    return "N/A";
  }

  if (["budget", "amountPaid", "awardAmount"].includes(key)) {
    return formatMoney(value);
  }

  if (["awardToBudgetRatio", "progress"].includes(key)) {
    return formatPercent(value);
  }

  return formatValue(value);
}

function formatComponentLocation(component) {
  const parts = [component.region, component.province].filter(
    (part) => part != null && String(part).trim() && String(part).trim().toUpperCase() !== "N/A"
  );
  return parts.length > 0 ? parts.join(", ") : "N/A";
}

function getPrettyRawJson(contract) {
  const rawJson = contract.rawJson || contract.raw_json || contract.rawJSON || {};
  try {
    return JSON.stringify(rawJson, null, 2);
  } catch {
    return "{}";
  }
}

export function ContractDrawer({ contract, onClose }) {
  if (!contract) {
    return null;
  }

  const budget = Number(contract.budget || 0);
  const awardAmount = Number(contract.awardAmount || 0);
  const awardToBudgetRatio =
    budget > 0 && awardAmount > 0 ? `${((awardAmount / budget) * 100).toFixed(1)}%` : "N/A";
  const documentLinks =
    contract.documentLinks && typeof contract.documentLinks === "object"
      ? contract.documentLinks
      : {};
  const documentLinkEntries = Object.entries(documentLinks).filter(
    ([, url]) => String(url || "").trim()
  );
  const components = Array.isArray(contract.components) ? contract.components : [];
  const dbFields = contract.dbFields && typeof contract.dbFields === "object" ? contract.dbFields : {};

  const summaryFields = [
    ["Contract ID", contract.contractId],
    ["Contractor", contract.contractor],
    ["Status", contract.status],
    ["Category", contract.category],
    ["Budget", contract.budget != null ? `PHP ${Number(contract.budget).toLocaleString()}` : "N/A"],
    ["Award Amount", awardAmount > 0 ? `PHP ${awardAmount.toLocaleString()}` : "N/A"],
    ["Award-to-Budget Ratio", awardToBudgetRatio],
    ["Amount Paid", contract.amountPaid != null ? `PHP ${Number(contract.amountPaid).toLocaleString()}` : "N/A"],
    ["Progress", contract.progress != null ? `${contract.progress}%` : "N/A"],
    ["Region", contract.region],
    ["Province", contract.province],
    ["Program", contract.programName],
    ["Source of Funds", contract.sourceOfFunds],
    ["Infra Year", contract.infraYear],
    ["Advertisement Date", contract.advertisementDate],
    ["Bid Submission Deadline", contract.bidSubmissionDeadline],
    ["Start Date", contract.startDate],
    ["Completion Date", contract.completionDate],
    ["Expiry Date", contract.expiryDate],
    ["Contract Duration", contract.contractDuration],
  ];

  const dbFieldOrder = [
    "contractId",
    "description",
    "category",
    "status",
    "budget",
    "amountPaid",
    "awardAmount",
    "awardToBudgetRatio",
    "progress",
    "region",
    "province",
    "latitude",
    "longitude",
    "contractor",
    "advertisementDate",
    "bidSubmissionDeadline",
    "startDate",
    "completionDate",
    "expiryDate",
    "infraYear",
    "programName",
    "sourceOfFunds",
    "contractDuration",
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

        <section className="drawer__section">
          <div className="drawer__section-heading">
            <div className="drawer__section-title">Overview</div>
            <div className="drawer__section-copy">Key contract facts from the selected record.</div>
          </div>
          <div className="drawer__fields">
            {summaryFields.map(([label, value]) => (
              <div key={label} className="drawer__field">
                <div className="drawer__field-label">{label}</div>
                <div className="drawer__field-value">{formatValue(value)}</div>
              </div>
            ))}
          </div>
        </section>

        <section className="drawer__section">
          <div className="drawer__section-heading">
            <div className="drawer__section-title">Document Links</div>
            <div className="drawer__section-copy">Direct files from the source payload.</div>
          </div>
          {documentLinkEntries.length > 0 ? (
            <div className="drawer__links">
              {documentLinkEntries.map(([label, url]) => (
                <a
                  key={label}
                  className="drawer__link"
                  href={url}
                  target="_blank"
                  rel="noreferrer"
                >
                  <span className="drawer__link-label">{label}</span>
                  <span className="drawer__link-url">{url}</span>
                </a>
              ))}
            </div>
          ) : (
            <div className="drawer__empty">No document links were stored for this contract.</div>
          )}
        </section>

        <section className="drawer__section">
          <div className="drawer__section-heading">
            <div className="drawer__section-title">Components</div>
            <div className="drawer__section-copy">Child work items tied to the contract.</div>
          </div>
          {components.length > 0 ? (
            <div className="drawer__components">
              {components.map((component, index) => (
                <div key={`${component.componentId || index}`} className="drawer__component">
                  <div className="drawer__component-title">
                    {component.componentId || `Component ${index + 1}`}
                  </div>
                  <div className="drawer__component-meta">
                    {component.typeOfWork || "N/A"} | {component.infraType || "N/A"}
                  </div>
                  <div className="drawer__component-copy">
                    {component.description || "No component description available."}
                  </div>
                  <div className="drawer__component-footnote">
                    {formatComponentLocation(component)}
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <div className="drawer__empty">No component rows were stored for this contract.</div>
          )}
        </section>

        <section className="drawer__section">
          <div className="drawer__section-heading">
            <div className="drawer__section-title">DB Fields</div>
            <div className="drawer__section-copy">Flat schema-backed values returned by the API.</div>
          </div>
          <div className="drawer__fields">
            {dbFieldOrder.map((key) => (
              <div key={key} className="drawer__field">
                <div className="drawer__field-label">{key}</div>
                <div className="drawer__field-value">{formatDbField(key, dbFields[key])}</div>
              </div>
            ))}
          </div>
        </section>

        <section className="drawer__section">
          <div className="drawer__section-heading">
            <div className="drawer__section-title">Raw JSON</div>
            <div className="drawer__section-copy">The original source payload as stored in the database.</div>
          </div>
          <pre className="drawer__json">{getPrettyRawJson(contract)}</pre>
        </section>
      </aside>
    </>
  );
}
