"use client";

import { useEffect, useState, useCallback } from "react";
import Sidebar from "@/components/Sidebar";
import EmptyState from "@/components/EmptyState";
import { loadData, saveData, getNextIntentColor } from "@/lib/store";
import type { TrainingData, Intent, TrainingExample } from "@/types";
import { v4 as uuidv4 } from "uuid";

export default function IntentsPage() {
  const [data, setData] = useState<TrainingData | null>(null);
  const [selectedIntent, setSelectedIntent] = useState<string | null>(null);
  const [newIntentName, setNewIntentName] = useState("");
  const [newIntentDesc, setNewIntentDesc] = useState("");
  const [newExample, setNewExample] = useState("");
  const [showCreateModal, setShowCreateModal] = useState(false);
  const [editingIntent, setEditingIntent] = useState<string | null>(null);
  const [editName, setEditName] = useState("");
  const [editDesc, setEditDesc] = useState("");
  const [generating, setGenerating] = useState(false);
  const [genCount, setGenCount] = useState(20);

  useEffect(() => {
    const loaded = loadData();
    setData(loaded);
    if (loaded.intents.length > 0) {
      setSelectedIntent(loaded.intents[0].id);
    }
  }, []);

  const persist = useCallback((updated: TrainingData) => {
    setData(updated);
    saveData(updated);
  }, []);

  const createIntent = () => {
    if (!data || !newIntentName.trim()) return;
    const intent: Intent = {
      id: `intent_${uuidv4().slice(0, 8)}`,
      name: newIntentName.trim().toLowerCase().replace(/\s+/g, "_"),
      description: newIntentDesc.trim(),
      color: getNextIntentColor(data.intents),
      examples: [],
    };
    const updated = { ...data, intents: [...data.intents, intent] };
    persist(updated);
    setSelectedIntent(intent.id);
    setNewIntentName("");
    setNewIntentDesc("");
    setShowCreateModal(false);
  };

  const deleteIntent = (intentId: string) => {
    if (!data) return;
    const updated = {
      ...data,
      intents: data.intents.filter((i) => i.id !== intentId),
    };
    persist(updated);
    if (selectedIntent === intentId) {
      setSelectedIntent(updated.intents[0]?.id || null);
    }
  };

  const addExample = () => {
    if (!data || !selectedIntent || !newExample.trim()) return;
    const intent = data.intents.find((i) => i.id === selectedIntent);
    if (!intent) return;

    const example: TrainingExample = {
      id: `ex_${uuidv4().slice(0, 8)}`,
      text: newExample.trim(),
      intent: intent.name,
      entities: [],
    };

    const updated = {
      ...data,
      intents: data.intents.map((i) =>
        i.id === selectedIntent
          ? { ...i, examples: [...i.examples, example] }
          : i
      ),
    };
    persist(updated);
    setNewExample("");
  };

  const deleteExample = (exampleId: string) => {
    if (!data || !selectedIntent) return;
    const updated = {
      ...data,
      intents: data.intents.map((i) =>
        i.id === selectedIntent
          ? { ...i, examples: i.examples.filter((e) => e.id !== exampleId) }
          : i
      ),
    };
    persist(updated);
  };

  const startEditIntent = (intent: Intent) => {
    setEditingIntent(intent.id);
    setEditName(intent.name);
    setEditDesc(intent.description);
  };

  const saveEditIntent = () => {
    if (!data || !editingIntent || !editName.trim()) return;
    const oldIntent = data.intents.find((i) => i.id === editingIntent);
    const newName = editName.trim().toLowerCase().replace(/\s+/g, "_");
    const updated = {
      ...data,
      intents: data.intents.map((i) =>
        i.id === editingIntent
          ? {
              ...i,
              name: newName,
              description: editDesc.trim(),
              examples: i.examples.map((e) =>
                e.intent === oldIntent?.name ? { ...e, intent: newName } : e
              ),
            }
          : i
      ),
    };
    persist(updated);
    setEditingIntent(null);
  };

  const [genError, setGenError] = useState<string | null>(null);

  const generateExamples = async () => {
    const intent = data?.intents.find((i) => i.id === selectedIntent);
    if (!data || !intent || generating) return;
    setGenerating(true);
    setGenError(null);
    try {
      const res = await fetch("/api/generate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          intentName: intent.name,
          intentDescription: intent.description,
          existingExamples: intent.examples.map((e) => e.text),
          count: genCount,
        }),
      });
      const result = await res.json();
      if (!res.ok) {
        setGenError(result.error || "Generation failed");
        return;
      }
      const newExamples: TrainingExample[] = (result.examples as string[]).map(
        (text: string) => ({
          id: `ex_${uuidv4().slice(0, 8)}`,
          text: text.trim(),
          intent: intent.name,
          entities: [],
        })
      );
      const updated = {
        ...data,
        intents: data.intents.map((i) =>
          i.id === intent.id
            ? { ...i, examples: [...i.examples, ...newExamples] }
            : i
        ),
      };
      persist(updated);
    } catch (err) {
      setGenError(`Network error: ${err instanceof Error ? err.message : String(err)}`);
    } finally {
      setGenerating(false);
    }
  };

  if (!data) return null;

  const currentIntent = data.intents.find((i) => i.id === selectedIntent);

  return (
    <div className="flex">
      <Sidebar />
      <main id="main-content" className="ml-64 flex-1 min-h-screen" role="main">
        <div className="flex h-screen">
          {/* Intent List */}
          <div className="w-72 border-r border-white/5 bg-surface-1/50 flex flex-col">
            <div className="p-4 border-b border-white/5">
              <div className="flex items-center justify-between mb-3">
                <h2 className="text-sm font-semibold text-white">Intents</h2>
                <button
                  onClick={() => setShowCreateModal(true)}
                  className="w-7 h-7 rounded-lg bg-brand-600 hover:bg-brand-700 flex items-center justify-center transition-colors"
                >
                  <svg className="w-4 h-4 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                    <path strokeLinecap="round" strokeLinejoin="round" d="M12 4.5v15m7.5-7.5h-15" />
                  </svg>
                </button>
              </div>
              <p className="text-xs text-gray-500">{data.intents.length} intents defined</p>
            </div>

            <div className="flex-1 overflow-y-auto p-2 space-y-1">
              {data.intents.map((intent) => (
                <button
                  key={intent.id}
                  onClick={() => setSelectedIntent(intent.id)}
                  className={`w-full text-left px-3 py-2.5 rounded-lg transition-all text-sm ${
                    selectedIntent === intent.id
                      ? "bg-surface-3 text-white"
                      : "text-gray-400 hover:bg-surface-2 hover:text-gray-300"
                  }`}
                >
                  <div className="flex items-center gap-2">
                    <span
                      className="w-2 h-2 rounded-full flex-shrink-0"
                      style={{ backgroundColor: intent.color }}
                    />
                    <span className="truncate font-medium">{intent.name}</span>
                    <span className="ml-auto text-xs text-gray-600">
                      {intent.examples.length}
                    </span>
                  </div>
                </button>
              ))}
            </div>
          </div>

          {/* Intent Detail */}
          <div className="flex-1 flex flex-col overflow-hidden">
            {currentIntent ? (
              <>
                {/* Intent Header */}
                <div className="p-6 border-b border-white/5">
                  {editingIntent === currentIntent.id ? (
                    <div className="space-y-3">
                      <input
                        value={editName}
                        onChange={(e) => setEditName(e.target.value)}
                        className="bg-surface-2 border border-white/10 rounded-lg px-3 py-2 text-white text-lg font-bold w-full focus:outline-none focus:border-brand-500"
                        placeholder="Intent name"
                      />
                      <input
                        value={editDesc}
                        onChange={(e) => setEditDesc(e.target.value)}
                        className="bg-surface-2 border border-white/10 rounded-lg px-3 py-1.5 text-gray-400 text-sm w-full focus:outline-none focus:border-brand-500"
                        placeholder="Description"
                      />
                      <div className="flex gap-2">
                        <button
                          onClick={saveEditIntent}
                          className="px-3 py-1.5 bg-brand-600 text-white text-xs font-medium rounded-lg hover:bg-brand-700"
                        >
                          Save
                        </button>
                        <button
                          onClick={() => setEditingIntent(null)}
                          className="px-3 py-1.5 bg-surface-3 text-gray-400 text-xs font-medium rounded-lg hover:bg-surface-4"
                        >
                          Cancel
                        </button>
                      </div>
                    </div>
                  ) : (
                    <div className="flex items-start justify-between">
                      <div>
                        <div className="flex items-center gap-3">
                          <span
                            className="w-3 h-3 rounded-full"
                            style={{ backgroundColor: currentIntent.color }}
                          />
                          <h2 className="text-xl font-bold text-white">
                            {currentIntent.name}
                          </h2>
                        </div>
                        <p className="text-sm text-gray-500 mt-1 ml-6">
                          {currentIntent.description || "No description"}
                        </p>
                      </div>
                      <div className="flex gap-2">
                        <button
                          onClick={() => startEditIntent(currentIntent)}
                          className="p-2 rounded-lg hover:bg-surface-3 text-gray-500 hover:text-gray-300 transition-colors"
                          title="Edit intent"
                        >
                          <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
                            <path strokeLinecap="round" strokeLinejoin="round" d="M16.862 4.487l1.687-1.688a1.875 1.875 0 112.652 2.652L10.582 16.07a4.5 4.5 0 01-1.897 1.13L6 18l.8-2.685a4.5 4.5 0 011.13-1.897l8.932-8.931zm0 0L19.5 7.125M18 14v4.75A2.25 2.25 0 0115.75 21H5.25A2.25 2.25 0 013 18.75V8.25A2.25 2.25 0 015.25 6H10" />
                          </svg>
                        </button>
                        <button
                          onClick={() => {
                            if (confirm(`Delete intent "${currentIntent.name}"?`)) {
                              deleteIntent(currentIntent.id);
                            }
                          }}
                          className="p-2 rounded-lg hover:bg-red-500/10 text-gray-500 hover:text-red-400 transition-colors"
                          title="Delete intent"
                        >
                          <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
                            <path strokeLinecap="round" strokeLinejoin="round" d="M14.74 9l-.346 9m-4.788 0L9.26 9m9.968-3.21c.342.052.682.107 1.022.166m-1.022-.165L18.16 19.673a2.25 2.25 0 01-2.244 2.077H8.084a2.25 2.25 0 01-2.244-2.077L4.772 5.79m14.456 0a48.108 48.108 0 00-3.478-.397m-12 .562c.34-.059.68-.114 1.022-.165m0 0a48.11 48.11 0 013.478-.397m7.5 0v-.916c0-1.18-.91-2.164-2.09-2.201a51.964 51.964 0 00-3.32 0c-1.18.037-2.09 1.022-2.09 2.201v.916m7.5 0a48.667 48.667 0 00-7.5 0" />
                          </svg>
                        </button>
                      </div>
                    </div>
                  )}
                </div>

                {/* Add Example */}
                <div className="p-4 border-b border-white/5">
                  <form
                    onSubmit={(e) => {
                      e.preventDefault();
                      addExample();
                    }}
                    className="flex gap-2"
                  >
                    <input
                      value={newExample}
                      onChange={(e) => setNewExample(e.target.value)}
                      placeholder='Add training example, e.g., "hello there"'
                      className="flex-1 bg-surface-2 border border-white/10 rounded-lg px-4 py-2.5 text-sm text-white placeholder-gray-600 focus:outline-none focus:border-brand-500 transition-colors"
                    />
                    <button
                      type="submit"
                      disabled={!newExample.trim()}
                      className="px-4 py-2.5 bg-brand-600 hover:bg-brand-700 disabled:opacity-30 disabled:cursor-not-allowed text-white text-sm font-medium rounded-lg transition-colors"
                    >
                      Add
                    </button>
                  </form>
                </div>

                {/* Examples List */}
                <div className="flex-1 overflow-y-auto p-4">
                  <div className="flex items-center justify-between mb-3">
                    <h3 className="text-xs text-gray-500 uppercase tracking-wider font-medium">
                      Training Examples ({currentIntent.examples.length})
                    </h3>
                    <div className="flex items-center gap-2">
                      <select
                        value={genCount}
                        onChange={(e) => setGenCount(Number(e.target.value))}
                        className="bg-surface-3 border border-white/10 rounded-md px-2 py-1 text-xs text-gray-400 focus:outline-none"
                        aria-label="Number of examples to generate"
                      >
                        <option value={10}>10</option>
                        <option value={20}>20</option>
                        <option value={30}>30</option>
                        <option value={50}>50</option>
                      </select>
                      <button
                        onClick={generateExamples}
                        disabled={generating}
                        className="flex items-center gap-1.5 px-3 py-1.5 bg-amber-600/20 hover:bg-amber-600/30 border border-amber-500/30 text-amber-400 text-xs font-medium rounded-lg transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                        title="Generate synthetic examples with Gemini AI (requires GOOGLE_API_KEY in .env.local)"
                      >
                        {generating ? (
                          <>
                            <svg className="w-3 h-3 animate-spin" viewBox="0 0 24 24" fill="none">
                              <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                              <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
                            </svg>
                            Generating...
                          </>
                        ) : (
                          <>
                            <svg className="w-3 h-3" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                              <path strokeLinecap="round" strokeLinejoin="round" d="M9.813 15.904L9 18.75l-.813-2.846a4.5 4.5 0 00-3.09-3.09L2.25 12l2.846-.813a4.5 4.5 0 003.09-3.09L9 5.25l.813 2.846a4.5 4.5 0 003.09 3.09L15.75 12l-2.846.813a4.5 4.5 0 00-3.09 3.09zM18.259 8.715L18 9.75l-.259-1.035a3.375 3.375 0 00-2.455-2.456L14.25 6l1.036-.259a3.375 3.375 0 002.455-2.456L18 2.25l.259 1.035a3.375 3.375 0 002.455 2.456L21.75 6l-1.036.259a3.375 3.375 0 00-2.455 2.456z" />
                            </svg>
                            Generate with AI
                          </>
                        )}
                      </button>
                    </div>
                  </div>
                  {genError && (
                    <div className="mb-3 p-2.5 rounded-lg bg-red-500/10 border border-red-500/20 text-xs text-red-400 flex items-start gap-2">
                      <svg className="w-3.5 h-3.5 flex-shrink-0 mt-0.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                        <path strokeLinecap="round" strokeLinejoin="round" d="M12 9v3.75m9-.75a9 9 0 11-18 0 9 9 0 0118 0zm-9 3.75h.008v.008H12v-.008z" />
                      </svg>
                      <span>{genError}</span>
                      <button onClick={() => setGenError(null)} className="ml-auto text-red-400/60 hover:text-red-400">
                        <svg className="w-3 h-3" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2}>
                          <path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12" />
                        </svg>
                      </button>
                    </div>
                  )}
                  {currentIntent.examples.length === 0 ? (
                    <div className="text-center py-12">
                      <p className="text-sm text-gray-600">
                        No examples yet. Add training sentences above.
                      </p>
                    </div>
                  ) : (
                    <div className="space-y-1">
                      {currentIntent.examples.map((example, idx) => (
                        <div
                          key={example.id}
                          className="flex items-center gap-3 px-3 py-2 rounded-lg hover:bg-surface-2 group animate-fade-in"
                        >
                          <span className="text-xs text-gray-600 w-6 text-right font-mono">
                            {idx + 1}
                          </span>
                          <span className="flex-1 text-sm text-gray-300">
                            {example.text}
                          </span>
                          <button
                            onClick={() => deleteExample(example.id)}
                            className="opacity-0 group-hover:opacity-100 p-1 rounded hover:bg-red-500/10 text-gray-600 hover:text-red-400 transition-all"
                          >
                            <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                              <path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12" />
                            </svg>
                          </button>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              </>
            ) : (
              <EmptyState
                icon={
                  <svg className="w-8 h-8" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1}>
                    <path strokeLinecap="round" strokeLinejoin="round" d="M9.568 3H5.25A2.25 2.25 0 003 5.25v4.318c0 .597.237 1.17.659 1.591l9.581 9.581c.699.699 1.78.872 2.607.33a18.095 18.095 0 005.223-5.223c.542-.827.369-1.908-.33-2.607L11.16 3.66A2.25 2.25 0 009.568 3z" />
                  </svg>
                }
                title="No intents yet"
                description="Create your first intent to start building your NLU training data"
                action={{
                  label: "Create Intent",
                  onClick: () => setShowCreateModal(true),
                }}
              />
            )}
          </div>
        </div>

        {/* Create Intent Modal */}
        {showCreateModal && (
          <div className="fixed inset-0 bg-black/60 backdrop-blur-sm flex items-center justify-center z-50 animate-fade-in">
            <div className="bg-surface-2 border border-white/10 rounded-2xl w-full max-w-md p-6 shadow-2xl animate-slide-up">
              <h3 className="text-lg font-bold text-white mb-4">Create New Intent</h3>
              <div className="space-y-4">
                <div>
                  <label className="text-xs text-gray-500 uppercase tracking-wider font-medium">
                    Name
                  </label>
                  <input
                    value={newIntentName}
                    onChange={(e) => setNewIntentName(e.target.value)}
                    placeholder="e.g., book_flight"
                    className="w-full mt-1 bg-surface-3 border border-white/10 rounded-lg px-3 py-2 text-white text-sm focus:outline-none focus:border-brand-500"
                    autoFocus
                  />
                </div>
                <div>
                  <label className="text-xs text-gray-500 uppercase tracking-wider font-medium">
                    Description
                  </label>
                  <input
                    value={newIntentDesc}
                    onChange={(e) => setNewIntentDesc(e.target.value)}
                    placeholder="e.g., User wants to book a flight"
                    className="w-full mt-1 bg-surface-3 border border-white/10 rounded-lg px-3 py-2 text-white text-sm focus:outline-none focus:border-brand-500"
                  />
                </div>
              </div>
              <div className="flex justify-end gap-2 mt-6">
                <button
                  onClick={() => {
                    setShowCreateModal(false);
                    setNewIntentName("");
                    setNewIntentDesc("");
                  }}
                  className="px-4 py-2 text-sm text-gray-400 hover:text-gray-300 transition-colors"
                >
                  Cancel
                </button>
                <button
                  onClick={createIntent}
                  disabled={!newIntentName.trim()}
                  className="px-4 py-2 bg-brand-600 hover:bg-brand-700 disabled:opacity-30 text-white text-sm font-medium rounded-lg transition-colors"
                >
                  Create
                </button>
              </div>
            </div>
          </div>
        )}
      </main>
    </div>
  );
}
