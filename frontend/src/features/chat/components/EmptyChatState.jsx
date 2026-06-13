const SUGGESTIONS = [
  "Any ongoing road projects in Region XI?",
  "Show flood control projects in Leyte",
  "How many completed bridges are in CAR?",
  "How many contracts does Sunwest Construction have?",
];

export function EmptyChatState({ onSuggestionClick }) {
  return (
    <div className="empty-state">
      <div className="empty-state__eyebrow">DPWH Watchdog</div>
      <h1 className="empty-state__title">Where should we begin?</h1>
      <p className="empty-state__subtitle">
        Explore contracts, regions, contractors, and result follow-ups in one chat.
      </p>

      <div className="empty-state__suggestions">
        {SUGGESTIONS.map((suggestion) => (
          <button
            key={suggestion}
            className="empty-state__chip"
            onClick={() => onSuggestionClick(suggestion)}
            type="button"
          >
            {suggestion}
          </button>
        ))}
      </div>
    </div>
  );
}
