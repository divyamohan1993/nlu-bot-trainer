"use client";

import { useEffect, useState, useMemo } from "react";
import Sidebar from "@/components/Sidebar";
import { loadData, loadEnsembleModel } from "@/lib/store";
import { getDriftReport, loadDriftState } from "@/lib/enterprise/drift-detector";
import { learnTemperature, type CalibrationResult } from "@/lib/engine/confidence-calibration";
import { validateDataQuality, type DataQualityReport } from "@/lib/engine/data-quality";
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

  // Calibration is expensive (~80 ensemble predictions + grid search) — on-demand only
  const [calibration, setCalibration] = useState<CalibrationResult | null>(null);
  const [calibrationLoading, setCalibrationLoading] = useState(false);
  const runCalibration = () => {
    if (!model || !data || calibrationLoading) return;
    const flat = data.intents.flatMap((i) => i.examples.map((e) => ({ text: e.text, intent: i.name })));
    if (flat.length < 5) return;
    // Sample up to 36 examples stratified (3 per intent for 12 intents)
    const sampled = flat.length <= 36 ? flat : (() => {
      const byIntent: Record<string, typeof flat> = {};
      for (const item of flat) {
        if (!byIntent[item.intent]) byIntent[item.intent] = [];
        byIntent[item.intent].push(item);
      }
      const intents = Object.keys(byIntent);
      const perIntent = Math.max(2, Math.floor(36 / intents.length));
      const result: typeof flat = [];
      for (const intent of intents) {
        result.push(...byIntent[intent].slice(0, perIntent));
      }
      return result;
    })();
    setCalibrationLoading(true);
    setTimeout(() => {
      setCalibration(learnTemperature(model, sampled));
      setCalibrationLoading(false);
    }, 16);
  };

  // Data quality: structural checks run instantly, mislabel detection is on-demand
  const [dataQuality, setDataQuality] = useState<DataQualityReport | null>(null);
  const [qualityLoading, setQualityLoading] = useState(false);
  const [mislabelRun, setMislabelRun] = useState(false);
  // Run structural checks (no model) immediately on load
  useEffect(() => {
    if (!data) return;
    const flat = data.intents.flatMap((i) => i.examples.map((e) => ({ text: e.text, intent: i.name })));
    setDataQuality(validateDataQuality(flat)); // no model = no mislabel detection = instant
  }, [data]);
  // Mislabel detection: on-demand with model
  const runMislabelDetection = () => {
    if (!data || !model || qualityLoading) return;
    const flat = data.intents.flatMap((i) => i.examples.map((e) => ({ text: e.text, intent: i.name })));
    setQualityLoading(true);
    setTimeout(() => {
      setDataQuality(validateDataQuality(flat, model));
      setQualityLoading(false);
      setMislabelRun(true);
    }, 16);
  };

  if (!data) return null;

  return (
    <div className="flex">
      <Sidebar />
      <main id="main-content" className="md:ml-64 flex-1 min-h-screen p-4 pt-14 md:pt-8 md:p-8" role="main">
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

            {/* Confidence Calibration — Guo et al. 2017 */}
            {!calibration && (
              <section className="glass rounded-xl p-6 mb-6">
                <div className="flex items-center justify-between">
                  <div>
                    <h2 className="text-lg font-semibold text-white mb-1">Confidence Calibration</h2>
                    <p className="text-xs text-gray-500">Temperature scaling (Guo et al. 2017). Runs ~36 ensemble predictions.</p>
                  </div>
                  <button
                    onClick={runCalibration}
                    disabled={calibrationLoading}
                    className="px-4 py-2 bg-brand-600 hover:bg-brand-700 disabled:opacity-50 text-white text-sm font-semibold rounded-lg transition-colors"
                  >
                    {calibrationLoading ? "Computing..." : "Run Calibration"}
                  </button>
                </div>
              </section>
            )}
            {calibration && calibration.numExamples > 0 && (
              <section className="glass rounded-xl p-6 mb-6">
                <h2 className="text-lg font-semibold text-white mb-2">Confidence Calibration</h2>
                <p className="text-xs text-gray-500 mb-4">
                  Temperature scaling (Guo et al. 2017). T={calibration.temperature.toFixed(2)} — {calibration.temperature > 1 ? "softens overconfident" : calibration.temperature < 1 ? "sharpens underconfident" : "no change to"} probabilities.
                </p>
                <div className="grid grid-cols-2 lg:grid-cols-4 gap-4 mb-4">
                  <div className="bg-surface-3/50 rounded-lg p-3">
                    <p className="text-xs text-gray-500">Temperature (T)</p>
                    <p className="text-sm font-bold text-brand-400 mt-1">{calibration.temperature.toFixed(2)}</p>
                  </div>
                  <div className="bg-surface-3/50 rounded-lg p-3">
                    <p className="text-xs text-gray-500">ECE Before</p>
                    <p className="text-sm font-medium text-yellow-400 mt-1">{(calibration.eceBefore * 100).toFixed(1)}%</p>
                  </div>
                  <div className="bg-surface-3/50 rounded-lg p-3">
                    <p className="text-xs text-gray-500">ECE After</p>
                    <p className="text-sm font-medium text-green-400 mt-1">{(calibration.eceAfter * 100).toFixed(1)}%</p>
                  </div>
                  <div className="bg-surface-3/50 rounded-lg p-3">
                    <p className="text-xs text-gray-500">Improvement</p>
                    <p className={`text-sm font-medium mt-1 ${calibration.eceAfter < calibration.eceBefore ? "text-green-400" : "text-gray-400"}`}>
                      {calibration.eceBefore > 0 ? `${((1 - calibration.eceAfter / calibration.eceBefore) * 100).toFixed(0)}%` : "N/A"}
                    </p>
                  </div>
                </div>

                {/* Reliability diagram — CSS bar chart */}
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
                  {[
                    { title: "Before Calibration", bins: calibration.reliabilityBefore },
                    { title: "After Calibration", bins: calibration.reliabilityAfter },
                  ].map(({ title, bins }) => (
                    <div key={title}>
                      <p className="text-xs text-gray-400 mb-2">{title}</p>
                      <div className="bg-surface-3/30 rounded-lg p-3">
                        <div className="flex items-end gap-0.5 h-24" aria-label={`Reliability diagram: ${title}`}>
                          {bins.map((bin, idx) => {
                            const gap = Math.abs(bin.avgAccuracy - bin.avgConfidence);
                            const barColor = gap < 0.05 ? "#40c057" : gap < 0.15 ? "#f59f00" : "#fa5252";
                            return (
                              <div key={idx} className="flex-1 flex flex-col items-center gap-0.5">
                                <div className="w-full relative" style={{ height: "96px" }}>
                                  {/* Perfect calibration line (diagonal) */}
                                  <div
                                    className="absolute bottom-0 left-0 right-0 bg-white/10"
                                    style={{ height: `${bin.binCenter * 100}%` }}
                                  />
                                  {/* Actual accuracy bar */}
                                  <div
                                    className="absolute bottom-0 left-[15%] right-[15%] rounded-t-sm"
                                    style={{
                                      height: `${bin.count > 0 ? bin.avgAccuracy * 100 : 0}%`,
                                      backgroundColor: barColor,
                                      opacity: bin.count > 0 ? 0.8 : 0.15,
                                    }}
                                  />
                                </div>
                                <span className="text-[8px] text-gray-600 tabular-nums">{(bin.binCenter * 100).toFixed(0)}</span>
                              </div>
                            );
                          })}
                        </div>
                        <div className="flex justify-between mt-1">
                          <span className="text-[9px] text-gray-600">Confidence %</span>
                          <span className="text-[9px] text-gray-600">
                            <span className="inline-block w-2 h-2 bg-white/10 mr-1 align-middle" />ideal
                            <span className="inline-block w-2 h-2 rounded-sm ml-2 mr-1 align-middle" style={{ backgroundColor: "#40c057" }} />actual
                          </span>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </section>
            )}

            {/* Data Quality Report */}
            {dataQuality && (
              <section className="glass rounded-xl p-6 mb-6">
                <div className="flex items-center justify-between mb-2">
                  <div>
                    <h2 className="text-lg font-semibold text-white">Data Quality</h2>
                    <p className="text-xs text-gray-500 mt-0.5">Automated checks based on Sambasivan et al. (2021) &ldquo;Data Cascades in ML&rdquo;</p>
                  </div>
                  {model && !mislabelRun && (
                    <button
                      onClick={runMislabelDetection}
                      disabled={qualityLoading}
                      className="px-3 py-1.5 bg-yellow-600 hover:bg-yellow-700 disabled:opacity-50 text-white text-xs font-semibold rounded-lg transition-colors"
                    >
                      {qualityLoading ? "Scanning..." : "Run Mislabel Detection"}
                    </button>
                  )}
                </div>

                <div className="grid grid-cols-2 lg:grid-cols-5 gap-4 mb-4">
                  <div className="bg-surface-3/50 rounded-lg p-3">
                    <p className="text-xs text-gray-500">Score</p>
                    <p className={`text-xl font-bold mt-1 ${
                      dataQuality.score >= 80 ? "text-green-400" : dataQuality.score >= 60 ? "text-yellow-400" : "text-red-400"
                    }`}>
                      {dataQuality.score}<span className="text-sm text-gray-500">/100</span>
                    </p>
                  </div>
                  <div className="bg-surface-3/50 rounded-lg p-3">
                    <p className="text-xs text-gray-500">Grade</p>
                    <p className={`text-xl font-bold mt-1 ${
                      dataQuality.grade === "A" ? "text-green-400" : dataQuality.grade === "B" ? "text-blue-400" : dataQuality.grade === "C" ? "text-yellow-400" : "text-red-400"
                    }`}>
                      {dataQuality.grade}
                    </p>
                  </div>
                  <div className="bg-surface-3/50 rounded-lg p-3">
                    <p className="text-xs text-gray-500">Duplicates</p>
                    <p className={`text-sm font-medium mt-1 ${dataQuality.stats.duplicateCount > 0 ? "text-yellow-400" : "text-green-400"}`}>
                      {dataQuality.stats.duplicateCount}
                    </p>
                  </div>
                  <div className="bg-surface-3/50 rounded-lg p-3">
                    <p className="text-xs text-gray-500">Mislabels</p>
                    <p className={`text-sm font-medium mt-1 ${dataQuality.stats.suspectedMislabels > 0 ? "text-red-400" : "text-green-400"}`}>
                      {dataQuality.stats.suspectedMislabels}
                    </p>
                  </div>
                  <div className="bg-surface-3/50 rounded-lg p-3">
                    <p className="text-xs text-gray-500">Vocabulary</p>
                    <p className="text-sm font-medium text-white mt-1">{dataQuality.stats.vocabularySize}</p>
                  </div>
                </div>

                {/* Issues list */}
                {dataQuality.issues.length > 0 && (
                  <div className="space-y-1.5 max-h-48 overflow-y-auto">
                    {dataQuality.issues.slice(0, 20).map((issue, idx) => (
                      <div key={idx} className={`flex items-start gap-2 px-3 py-2 rounded-lg text-xs ${
                        issue.severity === "error" ? "bg-red-500/5 border border-red-500/10" :
                        issue.severity === "warning" ? "bg-yellow-500/5 border border-yellow-500/10" :
                        "bg-blue-500/5 border border-blue-500/10"
                      }`}>
                        <span className={`font-mono text-[10px] uppercase px-1 py-0.5 rounded ${
                          issue.severity === "error" ? "bg-red-500/15 text-red-400" :
                          issue.severity === "warning" ? "bg-yellow-500/15 text-yellow-400" :
                          "bg-blue-500/15 text-blue-400"
                        }`}>
                          {issue.severity}
                        </span>
                        <span className="text-gray-300 flex-1">{issue.message}</span>
                      </div>
                    ))}
                    {dataQuality.issues.length > 20 && (
                      <p className="text-xs text-gray-500 pl-3">+{dataQuality.issues.length - 20} more issues</p>
                    )}
                  </div>
                )}
              </section>
            )}

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
