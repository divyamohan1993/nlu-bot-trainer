"use client";

import { useEffect, useState, useRef } from "react";
import Sidebar from "@/components/Sidebar";
import { loadData, loadEnsembleModel, saveEnsembleModel } from "@/lib/store";
import {
  runSelfLearningLoop,
  DEFAULT_SELF_LEARN_CONFIG,
  type SelfLearnConfig,
  type SelfLearnIteration,
  type SelfLearnResult,
} from "@/lib/self-learn/autonomous-loop";
import { registerModel } from "@/lib/enterprise/model-registry";
import type { TrainingData } from "@/types";
import type { EnsembleModel } from "@/lib/engine/ensemble";

type LearnStatus = "idle" | "running" | "complete" | "error";

export default function SelfLearnPage() {
  const [data, setData] = useState<TrainingData | null>(null);
  const [model, setModel] = useState<EnsembleModel | null>(null);
  const [status, setStatus] = useState<LearnStatus>("idle");
  const [iterations, setIterations] = useState<SelfLearnIteration[]>([]);
  const [result, setResult] = useState<SelfLearnResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [config, setConfig] = useState<SelfLearnConfig>({ ...DEFAULT_SELF_LEARN_CONFIG });
  const iterEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    setData(loadData());
    setModel(loadEnsembleModel());
  }, []);

  useEffect(() => {
    iterEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [iterations]);

  const startSelfLearning = async () => {
    if (!data) return;

    setStatus("running");
    setIterations([]);
    setResult(null);
    setError(null);

    const allExamples = data.intents.flatMap((intent) =>
      intent.examples.map((ex) => ({ text: ex.text, intent: intent.name }))
    );

    if (allExamples.length < 10) {
      setError("Need at least 10 training examples for self-learning.");
      setStatus("error");
      return;
    }

    // Run async to allow UI updates
    await new Promise((r) => setTimeout(r, 100));

    try {
      const res = runSelfLearningLoop(allExamples, config, (iteration) => {
        setIterations((prev) => [...prev, iteration]);
      });

      setResult(res);

      // Register and save the improved model
      if (res.totalImprovement > 0) {
        saveEnsembleModel(res.finalModel);
        setModel(res.finalModel);
        registerModel(res.finalModel, "patch", `Self-learned: +${(res.totalImprovement * 100).toFixed(1)}% in ${res.iterations.length} iterations`);
      }

      setStatus("complete");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Self-learning failed");
      setStatus("error");
    }
  };

  if (!data) return null;

  return (
    <div className="flex">
      <Sidebar />
      <main id="main-content" className="ml-64 flex-1 min-h-screen p-8" role="main">
        <div className="max-w-4xl">
          <header className="mb-8">
            <h1 className="text-3xl font-bold text-white tracking-tight">Self-Learning</h1>
            <p className="text-gray-500 mt-1">Autonomous recursive improvement: augment, self-train, validate, repeat</p>
          </header>

          {!model ? (
            <div className="glass rounded-xl p-12 text-center">
              <p className="text-gray-400">Train a model first on the <a href="/train" className="text-brand-400 hover:underline">Train</a> page.</p>
            </div>
          ) : (
            <>
              {/* Current Model Stats */}
              <section className="glass rounded-xl p-6 mb-6">
                <h2 className="text-sm font-semibold text-white mb-3">Current Model</h2>
                <div className="grid grid-cols-4 gap-4">
                  <div>
                    <p className="text-2xl font-bold text-white tabular-nums">{Math.round(model.metrics.accuracy * 100)}%</p>
                    <p className="text-xs text-gray-500">Accuracy</p>
                  </div>
                  <div>
                    <p className="text-2xl font-bold text-white tabular-nums">{model.metrics.totalExamples}</p>
                    <p className="text-xs text-gray-500">Examples</p>
                  </div>
                  <div>
                    <p className="text-2xl font-bold text-white tabular-nums">{model.classes.length}</p>
                    <p className="text-xs text-gray-500">Classes</p>
                  </div>
                  <div>
                    <p className="text-2xl font-bold text-white tabular-nums">{model.metrics.vocabularySize.toLocaleString()}</p>
                    <p className="text-xs text-gray-500">Vocab Size</p>
                  </div>
                </div>
              </section>

              {/* Configuration */}
              <section className="glass rounded-xl p-6 mb-6">
                <h2 className="text-sm font-semibold text-white mb-4">Self-Learning Configuration</h2>
                <div className="grid grid-cols-2 lg:grid-cols-3 gap-4">
                  <label className="block">
                    <span className="text-xs text-gray-500">Max Iterations</span>
                    <input
                      type="number"
                      min={1}
                      max={50}
                      value={config.maxIterations}
                      onChange={(e) => setConfig({ ...config, maxIterations: Number(e.target.value) })}
                      className="mt-1 w-full bg-surface-3 border border-white/10 rounded-lg px-3 py-2 text-sm text-white focus:outline-none focus:border-brand-500"
                      disabled={status === "running"}
                    />
                  </label>
                  <label className="block">
                    <span className="text-xs text-gray-500">Pseudo-Label Threshold</span>
                    <input
                      type="number"
                      min={0.5}
                      max={0.99}
                      step={0.01}
                      value={config.pseudoLabelThreshold}
                      onChange={(e) => setConfig({ ...config, pseudoLabelThreshold: Number(e.target.value) })}
                      className="mt-1 w-full bg-surface-3 border border-white/10 rounded-lg px-3 py-2 text-sm text-white focus:outline-none focus:border-brand-500"
                      disabled={status === "running"}
                    />
                  </label>
                  <label className="block">
                    <span className="text-xs text-gray-500">Max Augment Ratio</span>
                    <input
                      type="number"
                      min={0.5}
                      max={5}
                      step={0.5}
                      value={config.maxAugmentRatio}
                      onChange={(e) => setConfig({ ...config, maxAugmentRatio: Number(e.target.value) })}
                      className="mt-1 w-full bg-surface-3 border border-white/10 rounded-lg px-3 py-2 text-sm text-white focus:outline-none focus:border-brand-500"
                      disabled={status === "running"}
                    />
                  </label>
                </div>
                <div className="flex gap-4 mt-4">
                  {[
                    { key: "enableAugmentation" as const, label: "Data Augmentation" },
                    { key: "enablePseudoLabeling" as const, label: "Pseudo-Labeling" },
                    { key: "enableCurriculum" as const, label: "Curriculum Learning" },
                  ].map(({ key, label }) => (
                    <label key={key} className="flex items-center gap-2">
                      <input
                        type="checkbox"
                        checked={config[key]}
                        onChange={(e) => setConfig({ ...config, [key]: e.target.checked })}
                        className="w-3.5 h-3.5 accent-brand-500"
                        disabled={status === "running"}
                      />
                      <span className="text-xs text-gray-400">{label}</span>
                    </label>
                  ))}
                </div>
              </section>

              {/* Run Button */}
              <section className="glass rounded-xl p-6 mb-6">
                <div className="flex items-center justify-between">
                  <div>
                    <h2 className="text-sm font-semibold text-white">
                      {status === "complete" ? "Self-Learning Complete" : "Run Self-Learning Loop"}
                    </h2>
                    <p className="text-xs text-gray-500 mt-1">
                      {status === "running"
                        ? `Iteration ${iterations.length}/${config.maxIterations}...`
                        : "Augment weak intents, pseudo-label, curriculum order, retrain, validate"}
                    </p>
                  </div>
                  <button
                    onClick={startSelfLearning}
                    disabled={status === "running"}
                    className={`px-6 py-2.5 rounded-lg text-sm font-semibold transition-all ${
                      status === "complete"
                        ? "bg-green-600 hover:bg-green-700 text-white"
                        : status === "error"
                        ? "bg-red-600 hover:bg-red-700 text-white"
                        : "bg-brand-600 hover:bg-brand-700 text-white disabled:opacity-50"
                    }`}
                  >
                    {status === "idle" && "Start Self-Learning"}
                    {status === "running" && "Running..."}
                    {status === "complete" && "Run Again"}
                    {status === "error" && "Retry"}
                  </button>
                </div>

                {/* Progress */}
                {status === "running" && (
                  <div className="mt-4" role="progressbar" aria-valuenow={iterations.length} aria-valuemin={0} aria-valuemax={config.maxIterations}>
                    <div className="h-2 bg-surface-3 rounded-full overflow-hidden">
                      <div
                        className="h-full rounded-full bg-brand-500 transition-all duration-500"
                        style={{ width: `${(iterations.length / config.maxIterations) * 100}%` }}
                      />
                    </div>
                  </div>
                )}

                {error && (
                  <div role="alert" className="mt-4 p-3 rounded-lg bg-red-500/10 border border-red-500/20 text-sm text-red-400">
                    {error}
                  </div>
                )}
              </section>

              {/* Iterations Log */}
              {iterations.length > 0 && (
                <section className="glass rounded-xl p-6 mb-6">
                  <h2 className="text-sm font-semibold text-white mb-4">Iteration History</h2>
                  <div className="space-y-2 max-h-96 overflow-y-auto">
                    {iterations.map((iter) => (
                      <div key={iter.iteration} className="bg-surface-3/50 rounded-lg p-3 flex items-center gap-4">
                        <div className="w-8 h-8 rounded-full bg-surface-4 flex items-center justify-center text-xs font-bold text-gray-400">
                          {iter.iteration}
                        </div>
                        <div className="flex-1 min-w-0">
                          <div className="flex items-center gap-2">
                            <span className="text-sm font-medium text-white">{(iter.accuracy * 100).toFixed(1)}%</span>
                            <span className={`text-xs font-mono ${iter.improvement >= 0 ? "text-green-400" : "text-red-400"}`}>
                              {iter.improvement >= 0 ? "+" : ""}{(iter.improvement * 100).toFixed(2)}%
                            </span>
                          </div>
                          <p className="text-xs text-gray-500 truncate mt-0.5">{iter.action}</p>
                        </div>
                        <div className="text-right">
                          <p className="text-xs text-gray-500">{iter.augmentedExamples} aug</p>
                          <p className="text-xs text-gray-500">{iter.pseudoLabeledExamples} pseudo</p>
                        </div>
                      </div>
                    ))}
                    <div ref={iterEndRef} />
                  </div>
                </section>
              )}

              {/* Results Summary */}
              {result && (
                <section className="glass rounded-xl p-6">
                  <h2 className="text-sm font-semibold text-white mb-4">Results</h2>
                  <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
                    <div className="bg-surface-3/50 rounded-lg p-3">
                      <p className="text-xs text-gray-500">Initial Accuracy</p>
                      <p className="text-lg font-bold text-white">{(result.initialAccuracy * 100).toFixed(1)}%</p>
                    </div>
                    <div className="bg-surface-3/50 rounded-lg p-3">
                      <p className="text-xs text-gray-500">Final Accuracy</p>
                      <p className="text-lg font-bold text-green-400">{(result.finalAccuracy * 100).toFixed(1)}%</p>
                    </div>
                    <div className="bg-surface-3/50 rounded-lg p-3">
                      <p className="text-xs text-gray-500">Total Improvement</p>
                      <p className={`text-lg font-bold ${result.totalImprovement >= 0 ? "text-green-400" : "text-red-400"}`}>
                        {result.totalImprovement >= 0 ? "+" : ""}{(result.totalImprovement * 100).toFixed(2)}%
                      </p>
                    </div>
                    <div className="bg-surface-3/50 rounded-lg p-3">
                      <p className="text-xs text-gray-500">Duration</p>
                      <p className="text-lg font-bold text-white">{(result.durationMs / 1000).toFixed(1)}s</p>
                    </div>
                  </div>
                  <div className="mt-4 pt-4 border-t border-white/5">
                    <p className="text-xs text-gray-500">
                      {result.converged ? "Converged" : "Stopped"} after {result.iterations.length} iterations.
                      Reason: {result.stoppedReason.replace("_", " ")}.
                      {result.totalNewExamples > 0 && ` Added ${result.totalNewExamples} synthetic examples.`}
                    </p>
                  </div>
                </section>
              )}
            </>
          )}
        </div>
      </main>
    </div>
  );
}
