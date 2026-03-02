"use client";

import { useState, useCallback, useRef } from "react";
import Sidebar from "@/components/Sidebar";
import { loadData, saveData } from "@/lib/store";
import { v4 as uuidv4 } from "uuid";
import type { TrainingData, Intent, TrainingExample } from "@/types";

type ImportFormat = "csv" | "json_rasa" | "json_flat" | "text";

interface ParsedRow {
  text: string;
  intent: string;
}

const COLORS = [
  "#4c6ef5", "#be4bdb", "#f76707", "#40c057", "#15aabf",
  "#e64980", "#f59f00", "#7950f2", "#fd7e14", "#12b886",
  "#845ef7", "#ff6b6b", "#20c997", "#339af0", "#fcc419",
];

function detectFormat(content: string): ImportFormat {
  const trimmed = content.trim();
  if (trimmed.startsWith("{") || trimmed.startsWith("[")) {
    try {
      const parsed = JSON.parse(trimmed);
      if (parsed.nlu_data || parsed.rasa_nlu_data) return "json_rasa";
      if (Array.isArray(parsed) && parsed[0]?.text !== undefined) return "json_flat";
    } catch { /* not valid JSON */ }
  }
  if (trimmed.includes(",") && trimmed.split("\n")[0].includes(",")) return "csv";
  return "text";
}

function parseCSV(content: string): ParsedRow[] {
  const lines = content.trim().split("\n");
  const rows: ParsedRow[] = [];
  // Skip header if detected
  const firstLine = lines[0].toLowerCase();
  const startIdx = (firstLine.includes("text") || firstLine.includes("intent") || firstLine.includes("label")) ? 1 : 0;
  for (let i = startIdx; i < lines.length; i++) {
    const line = lines[i].trim();
    if (!line) continue;
    // Support both comma and tab delimiters
    const sep = line.includes("\t") ? "\t" : ",";
    const parts = line.split(sep);
    if (parts.length >= 2) {
      const text = parts[0].replace(/^["']|["']$/g, "").trim();
      const intent = parts[1].replace(/^["']|["']$/g, "").trim();
      if (text && intent) rows.push({ text, intent });
    }
  }
  return rows;
}

function parseRasa(content: string): ParsedRow[] {
  const parsed = JSON.parse(content);
  const nluData = parsed.nlu_data || parsed.rasa_nlu_data || parsed;
  const examples = nluData?.common_examples || nluData?.examples || [];
  return examples
    .filter((e: { text?: string; intent?: string }) => e.text && e.intent)
    .map((e: { text: string; intent: string }) => ({ text: e.text, intent: e.intent }));
}

function parseFlatJSON(content: string): ParsedRow[] {
  const parsed = JSON.parse(content);
  const arr = Array.isArray(parsed) ? parsed : parsed.data || parsed.examples || [];
  return arr
    .filter((e: { text?: string; intent?: string; label?: string }) => e.text && (e.intent || e.label))
    .map((e: { text: string; intent?: string; label?: string }) => ({
      text: e.text,
      intent: e.intent || e.label || "unknown",
    }));
}

function parsePlainText(content: string): ParsedRow[] {
  // Each line is a user utterance; assign "uncategorized" intent
  return content.trim().split("\n")
    .map((l) => l.trim())
    .filter((l) => l.length > 0)
    .map((text) => ({ text, intent: "uncategorized" }));
}

export default function ImportPage() {
  const [rows, setRows] = useState<ParsedRow[]>([]);
  const [format, setFormat] = useState<ImportFormat | null>(null);
  const [imported, setImported] = useState(false);
  const [stats, setStats] = useState<{ intents: number; examples: number } | null>(null);
  const [dragOver, setDragOver] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const processContent = useCallback((content: string, filename?: string) => {
    setError(null);
    setImported(false);
    const detected = detectFormat(content);
    setFormat(detected);

    try {
      let parsed: ParsedRow[];
      switch (detected) {
        case "csv": parsed = parseCSV(content); break;
        case "json_rasa": parsed = parseRasa(content); break;
        case "json_flat": parsed = parseFlatJSON(content); break;
        case "text": parsed = parsePlainText(content); break;
      }
      if (parsed.length === 0) {
        setError("No valid examples found. Check the file format.");
        return;
      }
      setRows(parsed);
      const intents = new Set(parsed.map((r) => r.intent));
      setStats({ intents: intents.size, examples: parsed.length });
    } catch (e) {
      setError(`Parse error: ${(e as Error).message}`);
    }
  }, []);

  const handleFileSelect = useCallback((files: FileList | null) => {
    if (!files || files.length === 0) return;
    const file = files[0];
    const reader = new FileReader();
    reader.onload = (e) => {
      const content = e.target?.result as string;
      processContent(content, file.name);
    };
    reader.readAsText(file);
  }, [processContent]);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setDragOver(false);
    handleFileSelect(e.dataTransfer.files);
  }, [handleFileSelect]);

  const handleMerge = useCallback(() => {
    if (rows.length === 0) return;

    const data = loadData();
    if (!data) return;

    // Group imported rows by intent
    const byIntent: Record<string, string[]> = {};
    for (const row of rows) {
      if (!byIntent[row.intent]) byIntent[row.intent] = [];
      byIntent[row.intent].push(row.text);
    }

    let addedExamples = 0;
    let addedIntents = 0;

    for (const [intentName, texts] of Object.entries(byIntent)) {
      let existingIntent = data.intents.find((i) => i.name === intentName);

      if (!existingIntent) {
        // Create new intent
        const colorIdx = (data.intents.length + addedIntents) % COLORS.length;
        const newIntent: Intent = {
          id: uuidv4(),
          name: intentName,
          description: `Imported intent`,
          examples: [],
          color: COLORS[colorIdx],
        };
        data.intents.push(newIntent);
        existingIntent = newIntent;
        addedIntents++;
      }

      // Add examples, skip exact duplicates
      const existingTexts = new Set(existingIntent.examples.map((e) => e.text.toLowerCase().trim()));
      for (const text of texts) {
        if (!existingTexts.has(text.toLowerCase().trim())) {
          const example: TrainingExample = {
            id: uuidv4(),
            text,
            intent: intentName,
            entities: [],
          };
          existingIntent.examples.push(example);
          existingTexts.add(text.toLowerCase().trim());
          addedExamples++;
        }
      }
    }

    saveData(data);
    setImported(true);
    setStats({ intents: addedIntents, examples: addedExamples });
  }, [rows]);

  const intentGroups = rows.reduce<Record<string, number>>((acc, r) => {
    acc[r.intent] = (acc[r.intent] || 0) + 1;
    return acc;
  }, {});

  return (
    <div className="flex">
      <Sidebar />
      <main id="main-content" className="md:ml-64 flex-1 min-h-screen p-4 pt-14 md:pt-8 md:p-8" role="main">
        <header className="mb-8">
          <h1 className="text-3xl font-bold text-white tracking-tight">Import Data</h1>
          <p className="text-gray-500 mt-1">
            Import conversation logs or training data from CSV, Rasa JSON, flat JSON, or plain text
          </p>
        </header>

        {/* Drop zone */}
        <section
          onDrop={handleDrop}
          onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
          onDragLeave={() => setDragOver(false)}
          className={`glass rounded-xl p-12 text-center mb-6 transition-all border-2 border-dashed ${
            dragOver ? "border-brand-400 bg-brand-500/5" : "border-white/10"
          }`}
        >
          <input
            ref={fileInputRef}
            type="file"
            accept=".csv,.json,.jsonl,.txt,.tsv"
            onChange={(e) => handleFileSelect(e.target.files)}
            className="hidden"
            aria-label="Upload training data file"
          />
          <div className="w-12 h-12 rounded-2xl bg-surface-3 flex items-center justify-center mx-auto mb-4" aria-hidden="true">
            <svg className="w-6 h-6 text-gray-500" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
              <path strokeLinecap="round" strokeLinejoin="round" d="M3 16.5v2.25A2.25 2.25 0 005.25 21h13.5A2.25 2.25 0 0021 18.75V16.5m-13.5-9L12 3m0 0l4.5 4.5M12 3v13.5" />
            </svg>
          </div>
          <p className="text-gray-400 mb-2">Drag & drop a file or</p>
          <button
            onClick={() => fileInputRef.current?.click()}
            className="px-4 py-2 bg-brand-600 hover:bg-brand-700 text-white text-sm font-semibold rounded-lg transition-colors"
          >
            Choose File
          </button>
          <div className="mt-4 flex justify-center gap-3 text-[10px] text-gray-600 uppercase tracking-wider">
            <span>CSV</span>
            <span>Rasa JSON</span>
            <span>JSON Array</span>
            <span>Plain Text</span>
          </div>
        </section>

        {error && (
          <div className="glass rounded-xl p-4 mb-6 border border-red-500/20" role="alert">
            <p className="text-sm text-red-400">{error}</p>
          </div>
        )}

        {/* Preview */}
        {rows.length > 0 && !imported && (
          <>
            <section className="glass rounded-xl p-6 mb-6">
              <div className="flex items-center justify-between mb-4">
                <div>
                  <h2 className="text-lg font-semibold text-white">Preview</h2>
                  <p className="text-xs text-gray-500 mt-0.5">
                    Detected format: <span className="text-brand-400 font-mono">{format}</span>
                    {stats && ` — ${stats.examples} examples across ${stats.intents} intent(s)`}
                  </p>
                </div>
                <button
                  onClick={handleMerge}
                  className="px-4 py-2 bg-green-600 hover:bg-green-700 text-white text-sm font-semibold rounded-lg transition-colors"
                >
                  Merge into Training Data
                </button>
              </div>

              {/* Intent breakdown */}
              <div className="flex flex-wrap gap-2 mb-4">
                {Object.entries(intentGroups).map(([intent, count], idx) => (
                  <span
                    key={intent}
                    className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-medium"
                    style={{
                      backgroundColor: `${COLORS[idx % COLORS.length]}15`,
                      color: COLORS[idx % COLORS.length],
                      border: `1px solid ${COLORS[idx % COLORS.length]}30`,
                    }}
                  >
                    {intent}
                    <span className="text-[10px] opacity-60">{count}</span>
                  </span>
                ))}
              </div>

              {/* Sample rows */}
              <div className="overflow-x-auto max-h-64 overflow-y-auto">
                <table className="text-xs w-full">
                  <thead>
                    <tr className="border-b border-white/5">
                      <th className="text-left p-2 text-gray-500 font-medium w-8">#</th>
                      <th className="text-left p-2 text-gray-500 font-medium">Text</th>
                      <th className="text-left p-2 text-gray-500 font-medium w-32">Intent</th>
                    </tr>
                  </thead>
                  <tbody>
                    {rows.slice(0, 50).map((row, idx) => (
                      <tr key={idx} className="border-b border-white/[0.03]">
                        <td className="p-2 text-gray-600 font-mono">{idx + 1}</td>
                        <td className="p-2 text-gray-300">{row.text}</td>
                        <td className="p-2">
                          <span
                            className="text-[10px] font-mono px-1.5 py-0.5 rounded"
                            style={{
                              backgroundColor: `${COLORS[Object.keys(intentGroups).indexOf(row.intent) % COLORS.length]}15`,
                              color: COLORS[Object.keys(intentGroups).indexOf(row.intent) % COLORS.length],
                            }}
                          >
                            {row.intent}
                          </span>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
                {rows.length > 50 && (
                  <p className="text-xs text-gray-500 p-2">Showing 50 of {rows.length} examples</p>
                )}
              </div>
            </section>
          </>
        )}

        {/* Success message */}
        {imported && stats && (
          <div className="glass rounded-xl p-6 border border-green-500/20" role="status">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 rounded-full bg-green-500/10 flex items-center justify-center" aria-hidden="true">
                <svg className="w-5 h-5 text-green-400" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                  <path strokeLinecap="round" strokeLinejoin="round" d="M4.5 12.75l6 6 9-13.5" />
                </svg>
              </div>
              <div>
                <p className="text-sm font-semibold text-green-400">Import Successful</p>
                <p className="text-xs text-gray-400 mt-0.5">
                  Added {stats.examples} new example(s) and {stats.intents} new intent(s). Duplicates were skipped.
                </p>
              </div>
              <a
                href="/intents"
                className="ml-auto px-3 py-1.5 bg-surface-3 hover:bg-surface-4 text-gray-300 text-xs rounded-lg transition-colors"
              >
                View Intents
              </a>
              <a
                href="/train"
                className="px-3 py-1.5 bg-brand-600 hover:bg-brand-700 text-white text-xs font-semibold rounded-lg transition-colors"
              >
                Train Model
              </a>
            </div>
          </div>
        )}

        {/* Format guide */}
        <section className="glass rounded-xl p-6 mt-6">
          <h2 className="text-lg font-semibold text-white mb-4">Supported Formats</h2>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {[
              {
                name: "CSV / TSV",
                desc: "text,intent per line. Header row optional.",
                example: `text,intent\nhello there,greet\nbye bye,farewell`,
              },
              {
                name: "Rasa JSON",
                desc: "Rasa NLU training format (v2/v3).",
                example: `{"rasa_nlu_data":{"common_examples":[{"text":"hi","intent":"greet"}]}}`,
              },
              {
                name: "JSON Array",
                desc: 'Array of {text, intent} objects.',
                example: `[{"text":"hello","intent":"greet"}]`,
              },
              {
                name: "Plain Text",
                desc: "One utterance per line. All assigned to 'uncategorized' for manual labeling.",
                example: `where is my order\nI want a refund\nthank you`,
              },
            ].map((fmt) => (
              <div key={fmt.name} className="bg-surface-3/30 rounded-lg p-3">
                <p className="text-sm font-medium text-white mb-1">{fmt.name}</p>
                <p className="text-[10px] text-gray-500 mb-2">{fmt.desc}</p>
                <pre className="text-[10px] text-gray-400 bg-surface-3/50 rounded p-2 overflow-x-auto">{fmt.example}</pre>
              </div>
            ))}
          </div>
        </section>
      </main>
    </div>
  );
}
