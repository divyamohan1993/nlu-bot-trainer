"use client";

interface ConfidenceBarProps {
  label: string;
  confidence: number;
  color?: string;
  isTop?: boolean;
}

export default function ConfidenceBar({
  label,
  confidence,
  color = "#4c6ef5",
  isTop = false,
}: ConfidenceBarProps) {
  const pct = Math.round(confidence * 100);

  return (
    <div className={`flex items-center gap-3 ${isTop ? "mb-1" : ""}`}>
      <span
        className={`text-xs font-mono w-28 truncate ${
          isTop ? "text-white font-semibold" : "text-gray-400"
        }`}
      >
        {label}
      </span>
      <div className="flex-1 h-2 bg-surface-3 rounded-full overflow-hidden">
        <div
          className="h-full rounded-full confidence-bar"
          style={{
            width: `${pct}%`,
            backgroundColor: isTop ? color : `${color}60`,
          }}
        />
      </div>
      <span
        className={`text-xs font-mono w-12 text-right ${
          isTop ? "text-white font-semibold" : "text-gray-500"
        }`}
      >
        {pct}%
      </span>
    </div>
  );
}
