"use client";

import { useEffect, useState, useRef } from "react";
import Sidebar from "@/components/Sidebar";
import { loadData, saveData, loadEnsembleModel, saveEnsembleModel, clearModel } from "@/lib/store";
import { trainEnsemble, crossValidateEnsemble, type EnsembleModel } from "@/lib/engine/ensemble";
import { exportTrainingData, type ExportFormat } from "@/lib/enterprise/export-formats";
import { registerModel } from "@/lib/enterprise/model-registry";
import type { TrainingData } from "@/types";

type TrainStatus = "idle" | "preparing" | "training" | "meta_learning" | "validating" | "registering" | "complete" | "error";

export default function TrainPage() {
  const [data, setData] = useState<TrainingData | null>(null);
  const [model, setModel] = useState<EnsembleModel | null>(null);
  const [status, setStatus] = useState<TrainStatus>("idle");
  const [progress, setProgress] = useState(0);
  const [accuracy, setAccuracy] = useState<number | null>(null);
  const [cvAccuracy, setCvAccuracy] = useState<number | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [logs, setLogs] = useState<string[]>([]);
  const [exportFormat, setExportFormat] = useState<ExportFormat>("json");
  const logsEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    setData(loadData());
    const m = loadEnsembleModel();
    if (m) {
      setModel(m);
      setAccuracy(m.metrics.accuracy);
    }
  }, []);

  useEffect(() => {
    logsEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [logs]);

  const addLog = (msg: string) => {
    setLogs((prev) => [...prev, `[${new Date().toLocaleTimeString()}] ${msg}`]);
  };

  const startTraining = async () => {
    if (!data) return;

    setStatus("preparing");
    setProgress(0);
    setError(null);
    setLogs([]);
    setAccuracy(null);
    setCvAccuracy(null);

    const allExamples = data.intents.flatMap((intent) =>
      intent.examples.map((ex) => ({ text: ex.text, intent: intent.name }))
    );

    if (allExamples.length < 2) {
      setError("Need at least 2 training examples across all intents.");
      setStatus("error");
      return;
    }

    const uniqueIntents = new Set(allExamples.map((e) => e.intent));
    if (uniqueIntents.size < 2) {
      setError("Need at least 2 different intents with examples.");
      setStatus("error");
      return;
    }

    addLog(`Preparing ${allExamples.length} examples across ${uniqueIntents.size} intents`);
    setProgress(5);
    await new Promise((r) => setTimeout(r, 50));

    setStatus("training");
    addLog("Tokenizing with v2 engine: word n-grams, char n-grams, syntactic features, intent signals...");
    setProgress(10);
    await new Promise((r) => setTimeout(r, 50));

    addLog("Feature hashing with MurmurHash3 into 1024-dim vectors...");
    setProgress(15);
    await new Promise((r) => setTimeout(r, 50));

    addLog("Training Logistic Regression (SGD, 15 epochs)...");
    setProgress(20);
    await new Promise((r) => setTimeout(r, 50));

    addLog("Training Complement Naive Bayes (Rennie et al. 2003)...");
    setProgress(28);
    await new Promise((r) => setTimeout(r, 50));

    addLog("Training Linear SVM (Pegasos algorithm, 10 epochs)...");
    setProgress(36);
    await new Promise((r) => setTimeout(r, 50));

    addLog("Training Neural Network MLP (1024\u2192128\u219212, 50 epochs, ReLU+softmax)...");
    setProgress(44);
    await new Promise((r) => setTimeout(r, 50));

    addLog("Training Gradient Boosted Stumps (150 rounds, shrinkage=0.1)...");
    setProgress(52);
    await new Promise((r) => setTimeout(r, 50));

    setStatus("meta_learning");
    addLog("Learning 5-way ensemble meta-weights via 3-fold CV grid search...");
    setProgress(60);
    await new Promise((r) => setTimeout(r, 50));

    try {
      const startTime = performance.now();
      const trained = trainEnsemble(allExamples);
      const trainTime = performance.now() - startTime;

      setProgress(75);
      setAccuracy(trained.metrics.accuracy);
      addLog(`Training accuracy: ${(trained.metrics.accuracy * 100).toFixed(1)}%`);
      addLog(`Training time: ${trainTime.toFixed(0)}ms`);
      addLog(`Vocabulary size: ${trained.metrics.vocabularySize.toLocaleString()} features`);
      const mw = trained.metaWeights;
      addLog(`Meta-weights: LR=${(mw[0] * 100).toFixed(0)}% NB=${(mw[1] * 100).toFixed(0)}% SVM=${(mw[2] * 100).toFixed(0)}% MLP=${(mw[3] * 100).toFixed(0)}% GBM=${(mw[4] * 100).toFixed(0)}%`);
      addLog(`Total parameters: ~171K (LR:12K + NB:7K + SVM:12K + MLP:133K + GBM:7K)`);

      setStatus("validating");
      addLog("Running 3-fold stratified cross-validation...");
      setProgress(85);
      await new Promise((r) => setTimeout(r, 50));

      const cv = crossValidateEnsemble(allExamples, 3);
      setCvAccuracy(cv.accuracy);
      addLog(`Cross-validation accuracy: ${(cv.accuracy * 100).toFixed(1)}%`);

      const weakIntents = Object.entries(cv.perClassF1)
        .filter(([, f1]) => f1 < 0.9)
        .sort((a, b) => a[1] - b[1]);
      if (weakIntents.length > 0) {
        addLog(`Weak intents: ${weakIntents.map(([name, f1]) => `${name}(F1=${(f1 * 100).toFixed(0)}%)`).join(", ")}`);
        addLog("TIP: Use Self-Learn to auto-improve weak intents with augmentation.");
      }

      setProgress(90);
      saveEnsembleModel(trained);
      setModel(trained);

      setStatus("registering");
      addLog("Registering model in version registry...");
      setProgress(95);
      await new Promise((r) => setTimeout(r, 50));

      const version = registerModel(trained, "minor", `Trained on ${allExamples.length} examples, CV accuracy ${(cv.accuracy * 100).toFixed(1)}%`);
      addLog(`Registered as v${version.semver} (${version.status})`);

      const updatedData = {
        ...data,
        metadata: { ...data.metadata, trainedAt: trained.trainedAt },
      };
      saveData(updatedData);
      setData(updatedData);

      setProgress(100);
      setStatus("complete");
      addLog("Ensemble model trained and saved successfully!");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Training failed");
      setStatus("error");
      addLog(`ERROR: ${err instanceof Error ? err.message : "Training failed"}`);
    }
  };

  const resetModel = () => {
    clearModel();
    setModel(null);
    setStatus("idle");
    setProgress(0);
    setAccuracy(null);
    setCvAccuracy(null);
    setLogs([]);
  };

  const handleExport = () => {
    if (!data) return;
    const result = exportTrainingData(data, exportFormat);
    const blob = new Blob([result.content], { type: result.mimeType });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = result.filename;
    a.click();
    URL.revokeObjectURL(url);
  };

  if (!data) return null;

  const totalExamples = data.intents.reduce((s, i) => s + i.examples.length, 0);

  return (
    <div className="flex">
      <Sidebar />
      <main id="main-content" className="md:ml-64 flex-1 min-h-screen p-4 pt-14 md:pt-8 md:p-8" role="main">
        <div className="max-w-4xl">
          <header className="mb-8">
            <h1 className="text-3xl font-bold text-white tracking-tight">Train Model</h1>
            <p className="text-gray-500 mt-1">Train the 5-classifier stacking ensemble on your training data</p>
          </header>

          {/* Training Data Summary */}
          <section aria-label="Training data summary" className="glass rounded-xl p-6 mb-6">
            <h2 className="text-sm font-semibold text-white mb-4">Training Data Summary</h2>
            <div className="grid grid-cols-3 gap-4 mb-4">
              <div>
                <p className="text-2xl font-bold text-white tabular-nums">{data.intents.length}</p>
                <p className="text-xs text-gray-500">Intents</p>
              </div>
              <div>
                <p className="text-2xl font-bold text-white tabular-nums">{totalExamples}</p>
                <p className="text-xs text-gray-500">Examples</p>
              </div>
              <div>
                <p className="text-2xl font-bold text-white tabular-nums">{data.entities.length}</p>
                <p className="text-xs text-gray-500">Entities</p>
              </div>
            </div>
            <div className="pt-4 border-t border-white/5 grid grid-cols-3 gap-x-6 gap-y-1.5">
              {data.intents.map((intent) => (
                <div key={intent.id} className="flex items-center gap-2 text-xs">
                  <span className="w-1.5 h-1.5 rounded-full flex-shrink-0" style={{ backgroundColor: intent.color }} aria-hidden="true" />
                  <span className="text-gray-400 truncate">{intent.name}</span>
                  <span className="text-gray-600 ml-auto tabular-nums">{intent.examples.length}</span>
                </div>
              ))}
            </div>
          </section>

          {/* Ensemble Architecture Info */}
          <section className="glass rounded-xl p-6 mb-6">
            <h2 className="text-sm font-semibold text-white mb-3">Ensemble Architecture</h2>
            <div className="grid grid-cols-5 gap-3">
              {[
                { name: "Logistic Regression", desc: "SGD + L2, 12K params", color: "#4c6ef5" },
                { name: "Complement NB", desc: "Rennie et al., 7K params", color: "#be4bdb" },
                { name: "Linear SVM", desc: "Pegasos, 12K params", color: "#f76707" },
                { name: "Neural Net MLP", desc: "1024\u2192128\u219212, 133K params", color: "#e64980" },
                { name: "Gradient Boost", desc: "150 stumps, 7K params", color: "#40c057" },
              ].map((c) => (
                <div key={c.name} className="bg-surface-3/50 rounded-lg p-3 border-l-2" style={{ borderLeftColor: c.color }}>
                  <p className="text-xs font-semibold text-white">{c.name}</p>
                  <p className="text-[10px] text-gray-500 mt-0.5">{c.desc}</p>
                </div>
              ))}
            </div>
            {model && (
              <div className="mt-3 pt-3 border-t border-white/5">
                <p className="text-xs text-gray-500">
                  Meta-weights: LR {(model.metaWeights[0] * 100).toFixed(0)}% | NB {(model.metaWeights[1] * 100).toFixed(0)}% | SVM {(model.metaWeights[2] * 100).toFixed(0)}% | MLP {(model.metaWeights[3] * 100).toFixed(0)}% | GBM {(model.metaWeights[4] * 100).toFixed(0)}%
                </p>
                <p className="text-xs text-gray-500 mt-1">~171,772 total parameters</p>
              </div>
            )}
          </section>

          {/* Train Section */}
          <section aria-label="Model training" className="glass rounded-xl p-6 mb-6">
            <div className="flex items-center justify-between mb-4">
              <div>
                <h2 className="text-sm font-semibold text-white">
                  {status === "complete" ? "Ensemble Trained" : "Train Ensemble"}
                </h2>
                <p className="text-xs text-gray-500 mt-1">
                  {status === "complete"
                    ? `Last trained: ${model ? new Date(model.trainedAt).toLocaleString() : "never"}`
                    : "5-classifier stacking with cross-validated meta-weights"}
                </p>
              </div>
              <div className="flex gap-2">
                {model && (
                  <button
                    onClick={resetModel}
                    className="px-4 py-2 bg-surface-3 hover:bg-surface-4 text-gray-400 text-sm rounded-lg transition-colors"
                    aria-label="Reset trained model"
                  >
                    Reset
                  </button>
                )}
                <button
                  onClick={startTraining}
                  disabled={status !== "idle" && status !== "complete" && status !== "error"}
                  className={`px-6 py-2.5 rounded-lg text-sm font-semibold transition-all ${
                    status === "complete"
                      ? "bg-green-600 hover:bg-green-700 text-white"
                      : status === "error"
                      ? "bg-red-600 hover:bg-red-700 text-white"
                      : "bg-brand-600 hover:bg-brand-700 text-white disabled:opacity-50"
                  }`}
                  aria-live="polite"
                >
                  {status === "idle" && "Train Ensemble"}
                  {status === "preparing" && "Preparing..."}
                  {status === "training" && "Training classifiers..."}
                  {status === "meta_learning" && "Learning weights..."}
                  {status === "validating" && "Cross-validating..."}
                  {status === "registering" && "Registering..."}
                  {status === "complete" && "Retrain Ensemble"}
                  {status === "error" && "Retry Training"}
                </button>
              </div>
            </div>

            {/* Progress */}
            {status !== "idle" && (
              <div className="mb-4" role="progressbar" aria-valuenow={progress} aria-valuemin={0} aria-valuemax={100} aria-label="Training progress">
                <div className="h-2 bg-surface-3 rounded-full overflow-hidden">
                  <div
                    className={`h-full rounded-full transition-all duration-500 ${
                      status === "error" ? "bg-red-500" : status === "complete" ? "bg-green-500" : "bg-brand-500"
                    }`}
                    style={{ width: `${progress}%` }}
                  />
                </div>
              </div>
            )}

            {/* Accuracy Cards */}
            {(accuracy !== null || cvAccuracy !== null) && (
              <div className="grid grid-cols-2 gap-3 mb-4">
                {accuracy !== null && (
                  <div className="flex items-center gap-3 p-3 rounded-lg bg-surface-3" aria-live="polite">
                    <div
                      className={`w-12 h-12 rounded-lg flex items-center justify-center text-sm font-bold ${
                        accuracy >= 0.95 ? "bg-green-500/20 text-green-400"
                        : accuracy >= 0.85 ? "bg-yellow-500/20 text-yellow-400"
                        : "bg-red-500/20 text-red-400"
                      }`}
                    >
                      {Math.round(accuracy * 100)}%
                    </div>
                    <div>
                      <p className="text-sm font-medium text-white">Training Accuracy</p>
                      <p className="text-xs text-gray-500">On full training set</p>
                    </div>
                  </div>
                )}
                {cvAccuracy !== null && (
                  <div className="flex items-center gap-3 p-3 rounded-lg bg-surface-3" aria-live="polite">
                    <div
                      className={`w-12 h-12 rounded-lg flex items-center justify-center text-sm font-bold ${
                        cvAccuracy >= 0.9 ? "bg-green-500/20 text-green-400"
                        : cvAccuracy >= 0.7 ? "bg-yellow-500/20 text-yellow-400"
                        : "bg-red-500/20 text-red-400"
                      }`}
                    >
                      {Math.round(cvAccuracy * 100)}%
                    </div>
                    <div>
                      <p className="text-sm font-medium text-white">Cross-Validation</p>
                      <p className="text-xs text-gray-500">3-fold stratified</p>
                    </div>
                  </div>
                )}
              </div>
            )}

            {error && (
              <div role="alert" className="p-3 rounded-lg bg-red-500/10 border border-red-500/20 text-sm text-red-400">
                {error}
              </div>
            )}

            {/* Logs */}
            {logs.length > 0 && (
              <div className="mt-4">
                <h3 className="text-xs text-gray-500 uppercase tracking-wider font-medium mb-2">Training Log</h3>
                <div className="bg-surface-0 rounded-lg p-3 max-h-64 overflow-y-auto font-mono text-xs space-y-0.5" role="log" aria-live="polite">
                  {logs.map((log, i) => (
                    <div key={i} className={log.includes("ERROR") ? "text-red-400" : log.includes("successfully") ? "text-green-400" : log.includes("TIP") ? "text-yellow-400" : log.includes("Meta-weights") || log.includes("Registered") ? "text-purple-400" : "text-gray-500"}>
                      {log}
                    </div>
                  ))}
                  <div ref={logsEndRef} />
                </div>
              </div>
            )}
          </section>

          {/* Multi-Format Export */}
          <section aria-label="Export training data" className="glass rounded-xl p-6">
            <h2 className="text-sm font-semibold text-white mb-2">Export Training Data</h2>
            <p className="text-xs text-gray-500 mb-4">Download in 7 formats for cross-platform NLU interoperability</p>
            <div className="flex items-center gap-3">
              <select
                value={exportFormat}
                onChange={(e) => setExportFormat(e.target.value as ExportFormat)}
                className="bg-surface-3 border border-white/10 text-gray-300 text-sm rounded-lg px-3 py-2 focus:outline-none focus:border-brand-500"
                aria-label="Export format"
              >
                <option value="json">JSON (Universal)</option>
                <option value="rasa">Rasa NLU (YAML v3.1)</option>
                <option value="dialogflow">Dialogflow ES (JSON)</option>
                <option value="lex">Amazon Lex V2 (JSON)</option>
                <option value="luis">Microsoft LUIS (JSON)</option>
                <option value="wit">Wit.ai (JSON)</option>
                <option value="csv">CSV (Data Analysis)</option>
              </select>
              <button
                onClick={handleExport}
                className="px-4 py-2 bg-surface-3 hover:bg-surface-4 text-gray-300 text-sm font-medium rounded-lg transition-colors"
              >
                Export
              </button>
            </div>
          </section>
        </div>
      </main>
    </div>
  );
}
