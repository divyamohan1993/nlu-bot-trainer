"use client";

import { useEffect, useState, useMemo } from "react";
import Sidebar from "@/components/Sidebar";
import { loadData, loadEnsembleModel } from "@/lib/store";
import { getDriftReport, loadDriftState } from "@/lib/enterprise/drift-detector";
import type { TrainingData } from "@/types";
import type { EnsembleModel } from "@/lib/engine/ensemble";

export default function AnalyticsPage() {
  const [data, setData] = useState<TrainingData | null>(null);
  const [model, setModel] = useState<EnsembleModel | null>(null);

  useEffect(() => {
    setData(loadData());
    setModel(loadEnsembleModel());
  }, []);

  const driftReport = useMemo(() => {
    if (typeof window === "undefined") return null;
    return getDriftReport(loadDriftState());
  }, []);

  const f1Entries = useMemo(() => {
    if (!model) return [];
    return Object.entries(model.metrics.perClassF1).sort((a, b) => a[1] - b[1]);
  }, [model]);

  const confusionClasses = useMemo(() => {
    if (!model) return [];
    return Object.keys(model.metrics.confusionMatrix);
  }, [model]);

  if (!data) return null;

  return (
    <div className="flex">
      <Sidebar />
      <main id="main-content" className="ml-64 flex-1 min-h-screen p-8" role="main">
        <header className="mb-8">
          <h1 className="text-3xl font-bold text-white tracking-tight">Analytics</h1>
          <p className="text-gray-500 mt-1">Model performance, confusion matrix, and drift monitoring</p>
        </header>

        {!model ? (
          <div className="glass rounded-xl p-12 text-center">
            <p className="text-gray-400">No trained model. Go to <a href="/train" className="text-brand-400 hover:underline">Train</a> first.</p>
          </div>
        ) : (
          <>
            {/* Summary Cards */}
            <section className="grid grid-cols-2 lg:grid-cols-5 gap-4 mb-8">
              {[
                { label: "Accuracy", value: `${Math.round(model.metrics.accuracy * 100)}%`, color: model.metrics.accuracy >= 0.9 ? "#40c057" : "#f59f00" },
                { label: "Weighted F1", value: `${Math.round(f1Entries.reduce((s, [, v]) => s + v, 0) / (f1Entries.length || 1) * 100)}%`, color: "#4c6ef5" },
                { label: "Training Time", value: `${model.metrics.trainingTimeMs.toFixed(0)}ms`, color: "#be4bdb" },
                { label: "Vocabulary", value: model.metrics.vocabularySize.toLocaleString(), color: "#15aabf" },
                { label: "Drift", value: driftReport?.overallStatus || "N/A", color: driftReport?.overallStatus === "healthy" ? "#40c057" : driftReport?.overallStatus === "warning" ? "#f59f00" : "#fa5252" },
              ].map((s) => (
                <div key={s.label} className="glass rounded-xl p-4">
                  <p className="text-xs text-gray-500 uppercase tracking-wider">{s.label}</p>
                  <p className="text-xl font-bold mt-1" style={{ color: s.color }}>{s.value}</p>
                </div>
              ))}
            </section>

            {/* Per-Intent F1 Scores */}
            <section className="glass rounded-xl p-6 mb-6">
              <h2 className="text-lg font-semibold text-white mb-4">Per-Intent F1 Scores</h2>
              <div className="space-y-2">
                {f1Entries.map(([intent, f1]) => {
                  const pct = Math.round(f1 * 100);
                  const color = f1 >= 0.95 ? "#40c057" : f1 >= 0.8 ? "#f59f00" : "#fa5252";
                  const intentData = data.intents.find((i) => i.name === intent);
                  return (
                    <div key={intent} className="flex items-center gap-3">
                      <span className="text-sm text-gray-300 w-36 truncate">{intent}</span>
                      <div className="flex-1 h-2 bg-surface-3 rounded-full overflow-hidden">
                        <div
                          className="h-full rounded-full transition-all duration-700"
                          style={{ width: `${pct}%`, backgroundColor: intentData?.color || color }}
                        />
                      </div>
                      <span className="text-xs font-mono w-12 text-right" style={{ color }}>
                        {pct}%
                      </span>
                    </div>
                  );
                })}
              </div>
            </section>

            {/* Confusion Matrix */}
            <section className="glass rounded-xl p-6 mb-6">
              <h2 className="text-lg font-semibold text-white mb-4">Confusion Matrix</h2>
              <div className="overflow-x-auto">
                <table className="text-xs w-full" role="grid" aria-label="Confusion matrix">
                  <thead>
                    <tr>
                      <th className="p-2 text-left text-gray-500 font-medium">Actual \ Predicted</th>
                      {confusionClasses.map((cls) => (
                        <th key={cls} className="p-2 text-center text-gray-400 font-medium max-w-[80px] truncate" title={cls}>
                          {cls.length > 8 ? cls.slice(0, 7) + "..." : cls}
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {confusionClasses.map((actual) => (
                      <tr key={actual}>
                        <td className="p-2 text-gray-400 font-medium max-w-[100px] truncate" title={actual}>
                          {actual.length > 10 ? actual.slice(0, 9) + "..." : actual}
                        </td>
                        {confusionClasses.map((predicted) => {
                          const count = model.metrics.confusionMatrix[actual]?.[predicted] || 0;
                          const isCorrect = actual === predicted;
                          const total = Object.values(model.metrics.confusionMatrix[actual] || {}).reduce((s, c) => s + c, 0);
                          const intensity = total > 0 ? count / total : 0;
                          return (
                            <td
                              key={predicted}
                              className="p-2 text-center font-mono tabular-nums"
                              style={{
                                backgroundColor: isCorrect
                                  ? `rgba(64, 192, 87, ${intensity * 0.4})`
                                  : count > 0
                                  ? `rgba(250, 82, 82, ${intensity * 0.4})`
                                  : "transparent",
                                color: count > 0 ? (isCorrect ? "#40c057" : "#fa5252") : "#333",
                              }}
                              title={`${actual} -> ${predicted}: ${count}`}
                            >
                              {count || "·"}
                            </td>
                          );
                        })}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </section>

            {/* Meta-Weights */}
            <section className="glass rounded-xl p-6 mb-6">
              <h2 className="text-lg font-semibold text-white mb-4">Ensemble Meta-Weights</h2>
              <p className="text-xs text-gray-500 mb-4">Learned via 3-fold cross-validated grid search. Higher weight = more influence on final prediction.</p>
              <div className="grid grid-cols-5 gap-4">
                {[
                  { name: "Logistic Regression", weight: model.metaWeights[0], color: "#4c6ef5" },
                  { name: "Complement NB", weight: model.metaWeights[1], color: "#be4bdb" },
                  { name: "Linear SVM", weight: model.metaWeights[2], color: "#f76707" },
                  { name: "Neural Net MLP", weight: model.metaWeights[3], color: "#e64980" },
                  { name: "Gradient Boost", weight: model.metaWeights[4], color: "#40c057" },
                ].map((m) => (
                  <div key={m.name} className="text-center">
                    <div
                      className="w-16 h-16 rounded-full flex items-center justify-center mx-auto text-sm font-bold"
                      style={{ backgroundColor: `${m.color}20`, color: m.color, border: `2px solid ${m.color}40` }}
                    >
                      {Math.round(m.weight * 100)}%
                    </div>
                    <p className="text-xs text-gray-400 mt-2">{m.name}</p>
                  </div>
                ))}
              </div>
            </section>

            {/* Drift Monitoring */}
            <section className="glass rounded-xl p-6">
              <h2 className="text-lg font-semibold text-white mb-4">Drift Monitoring</h2>
              {driftReport && driftReport.predictionCount > 0 ? (
                <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
                  <div className="bg-surface-3/50 rounded-lg p-3">
                    <p className="text-xs text-gray-500">Concept Drift</p>
                    <p className={`text-sm font-medium mt-1 ${driftReport.conceptDrift.detected ? "text-red-400" : "text-green-400"}`}>
                      {driftReport.conceptDrift.detected ? "Detected" : "None"} ({(driftReport.conceptDrift.severity * 100).toFixed(0)}%)
                    </p>
                  </div>
                  <div className="bg-surface-3/50 rounded-lg p-3">
                    <p className="text-xs text-gray-500">Error Drift (DDM)</p>
                    <p className={`text-sm font-medium mt-1 ${
                      driftReport.errorDrift.state === "in_control" ? "text-green-400" : driftReport.errorDrift.state === "warning" ? "text-yellow-400" : "text-red-400"
                    }`}>
                      {driftReport.errorDrift.state.replace("_", " ")}
                    </p>
                  </div>
                  <div className="bg-surface-3/50 rounded-lg p-3">
                    <p className="text-xs text-gray-500">Recent Accuracy</p>
                    <p className="text-sm font-medium text-white mt-1">{(driftReport.recentAccuracy * 100).toFixed(1)}%</p>
                  </div>
                  <div className="bg-surface-3/50 rounded-lg p-3">
                    <p className="text-xs text-gray-500">Predictions Tracked</p>
                    <p className="text-sm font-medium text-white mt-1">{driftReport.predictionCount}</p>
                  </div>
                </div>
              ) : (
                <p className="text-sm text-gray-500">No prediction data yet. Use the Test page to generate predictions for drift monitoring.</p>
              )}
            </section>
          </>
        )}
      </main>
    </div>
  );
}
