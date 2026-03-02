"use client";

import { useEffect, useState } from "react";
import Sidebar from "@/components/Sidebar";
import { loadEnsembleModel, saveEnsembleModel } from "@/lib/store";
import {
  loadRegistry,
  loadModelVersion,
  promoteToChampion,
  deleteVersion,
  startABTest,
  concludeABTest,
  type ModelRegistryState,
  type ModelVersion,
  type ABTest,
} from "@/lib/enterprise/model-registry";
import type { EnsembleModel } from "@/lib/engine/ensemble";

export default function ModelsPage() {
  const [registry, setRegistry] = useState<ModelRegistryState | null>(null);
  const [model, setModel] = useState<EnsembleModel | null>(null);
  const [selectedVersion, setSelectedVersion] = useState<ModelVersion | null>(null);
  const [abTestName, setAbTestName] = useState("");
  const [abTrafficSplit, setAbTrafficSplit] = useState(0.1);

  const refresh = () => {
    setRegistry(loadRegistry());
    setModel(loadEnsembleModel());
  };

  useEffect(() => {
    refresh();
  }, []);

  const handlePromote = (versionId: string) => {
    promoteToChampion(versionId);
    const loadedModel = loadModelVersion(versionId);
    if (loadedModel) saveEnsembleModel(loadedModel);
    refresh();
  };

  const handleDelete = (versionId: string) => {
    deleteVersion(versionId);
    refresh();
  };

  const handleStartABTest = (challengerVersionId: string) => {
    if (!abTestName.trim()) return;
    startABTest(abTestName, challengerVersionId, abTrafficSplit);
    setAbTestName("");
    refresh();
  };

  const handleConcludeTest = (testId: string) => {
    concludeABTest(testId);
    refresh();
  };

  const statusColors: Record<string, string> = {
    champion: "#40c057",
    challenger: "#f59f00",
    staging: "#4c6ef5",
    draft: "#748ffc",
    retired: "#666",
  };

  if (!registry) return null;

  return (
    <div className="flex">
      <Sidebar />
      <main id="main-content" className="md:ml-64 flex-1 min-h-screen p-4 pt-14 md:pt-8 md:p-8" role="main">
        <div className="max-w-5xl">
          <header className="mb-8">
            <h1 className="text-3xl font-bold text-white tracking-tight">Model Registry</h1>
            <p className="text-gray-500 mt-1">Version management, champion/challenger, A/B testing, rollback</p>
          </header>

          {registry.versions.length === 0 ? (
            <div className="glass rounded-xl p-12 text-center">
              <p className="text-gray-400">No model versions. <a href="/train" className="text-brand-400 hover:underline">Train a model</a> to create the first version.</p>
            </div>
          ) : (
            <>
              {/* Version List */}
              <section className="glass rounded-xl p-6 mb-6">
                <h2 className="text-lg font-semibold text-white mb-4">
                  Model Versions ({registry.versions.length})
                </h2>
                <div className="space-y-2">
                  {[...registry.versions].reverse().map((version) => (
                    <div
                      key={version.id}
                      className={`bg-surface-3/50 rounded-lg p-4 flex items-center gap-4 cursor-pointer transition-colors hover:bg-surface-3 ${
                        selectedVersion?.id === version.id ? "ring-1 ring-brand-500" : ""
                      }`}
                      onClick={() => setSelectedVersion(selectedVersion?.id === version.id ? null : version)}
                      role="button"
                      tabIndex={0}
                      onKeyDown={(e) => { if (e.key === "Enter") setSelectedVersion(selectedVersion?.id === version.id ? null : version); }}
                      aria-label={`Version ${version.semver} - ${version.status}`}
                    >
                      {/* Version Badge */}
                      <div
                        className="w-12 h-12 rounded-lg flex items-center justify-center text-sm font-bold"
                        style={{ backgroundColor: `${statusColors[version.status]}20`, color: statusColors[version.status] }}
                      >
                        v{version.semver}
                      </div>

                      {/* Info */}
                      <div className="flex-1 min-w-0">
                        <div className="flex items-center gap-2">
                          <span
                            className="text-xs px-2 py-0.5 rounded-full font-medium"
                            style={{ backgroundColor: `${statusColors[version.status]}20`, color: statusColors[version.status] }}
                          >
                            {version.status}
                          </span>
                          <span className="text-xs text-gray-500">
                            {new Date(version.trainedAt).toLocaleDateString()} {new Date(version.trainedAt).toLocaleTimeString()}
                          </span>
                        </div>
                        <p className="text-xs text-gray-500 mt-1 truncate">{version.notes || "No notes"}</p>
                      </div>

                      {/* Metrics */}
                      <div className="text-right">
                        <p className="text-sm font-bold text-white">{Math.round(version.metrics.accuracy * 100)}%</p>
                        <p className="text-xs text-gray-500">{version.metrics.totalExamples} examples</p>
                      </div>

                      {/* Actions */}
                      <div className="flex gap-2" onClick={(e) => e.stopPropagation()}>
                        {version.status !== "champion" && (
                          <button
                            onClick={() => handlePromote(version.id)}
                            className="px-3 py-1.5 text-xs bg-green-600/20 text-green-400 hover:bg-green-600/30 rounded-lg transition-colors"
                            title="Promote to champion"
                          >
                            Promote
                          </button>
                        )}
                        {version.status !== "champion" && (
                          <button
                            onClick={() => handleDelete(version.id)}
                            className="px-3 py-1.5 text-xs bg-red-500/10 text-red-400 hover:bg-red-500/20 rounded-lg transition-colors"
                            title="Delete version"
                          >
                            Delete
                          </button>
                        )}
                      </div>
                    </div>
                  ))}
                </div>
              </section>

              {/* Selected Version Detail */}
              {selectedVersion && (
                <section className="glass rounded-xl p-6 mb-6">
                  <h2 className="text-lg font-semibold text-white mb-4">v{selectedVersion.semver} Details</h2>
                  <div className="grid grid-cols-2 lg:grid-cols-4 gap-4 mb-4">
                    <div className="bg-surface-3/50 rounded-lg p-3">
                      <p className="text-xs text-gray-500">Accuracy</p>
                      <p className="text-lg font-bold text-white">{Math.round(selectedVersion.metrics.accuracy * 100)}%</p>
                    </div>
                    <div className="bg-surface-3/50 rounded-lg p-3">
                      <p className="text-xs text-gray-500">Weighted F1</p>
                      <p className="text-lg font-bold text-white">{Math.round(selectedVersion.metrics.weightedF1 * 100)}%</p>
                    </div>
                    <div className="bg-surface-3/50 rounded-lg p-3">
                      <p className="text-xs text-gray-500">Training Time</p>
                      <p className="text-lg font-bold text-white">{selectedVersion.metrics.trainingTimeMs.toFixed(0)}ms</p>
                    </div>
                    <div className="bg-surface-3/50 rounded-lg p-3">
                      <p className="text-xs text-gray-500">Vocabulary</p>
                      <p className="text-lg font-bold text-white">{selectedVersion.metrics.vocabularySize.toLocaleString()}</p>
                    </div>
                  </div>

                  {/* Per-Intent F1 */}
                  <h3 className="text-xs text-gray-500 uppercase tracking-wider font-medium mb-2 mt-4">Per-Intent F1</h3>
                  <div className="grid grid-cols-2 lg:grid-cols-3 gap-2">
                    {Object.entries(selectedVersion.metrics.perIntentF1)
                      .sort((a, b) => a[1] - b[1])
                      .map(([intent, f1]) => (
                        <div key={intent} className="flex items-center gap-2 text-xs">
                          <span className="text-gray-400 truncate w-28">{intent}</span>
                          <div className="flex-1 h-1.5 bg-surface-3 rounded-full overflow-hidden">
                            <div
                              className="h-full rounded-full"
                              style={{
                                width: `${f1 * 100}%`,
                                backgroundColor: f1 >= 0.9 ? "#40c057" : f1 >= 0.7 ? "#f59f00" : "#fa5252",
                              }}
                            />
                          </div>
                          <span className="text-gray-500 font-mono w-10 text-right">{Math.round(f1 * 100)}%</span>
                        </div>
                      ))}
                  </div>

                  {/* Ensemble Config */}
                  <h3 className="text-xs text-gray-500 uppercase tracking-wider font-medium mb-2 mt-4">Ensemble Config</h3>
                  <p className="text-xs text-gray-500">
                    Hash dim: {selectedVersion.config.hashDim} |
                    Classifiers: {selectedVersion.config.classifierTypes.join(", ")} |
                    Weights: [{selectedVersion.config.ensembleWeights.map((w) => (w * 100).toFixed(0) + "%").join(", ")}]
                  </p>
                </section>
              )}

              {/* A/B Testing */}
              <section className="glass rounded-xl p-6">
                <h2 className="text-lg font-semibold text-white mb-4">A/B Testing</h2>

                {/* Active Tests */}
                {registry.abTests.filter((t) => t.status === "running").length > 0 && (
                  <div className="mb-4">
                    <h3 className="text-xs text-gray-500 uppercase tracking-wider font-medium mb-2">Active Tests</h3>
                    {registry.abTests.filter((t) => t.status === "running").map((test) => (
                      <div key={test.id} className="bg-surface-3/50 rounded-lg p-4 mb-2">
                        <div className="flex items-center justify-between">
                          <div>
                            <p className="text-sm font-medium text-white">{test.name}</p>
                            <p className="text-xs text-gray-500">
                              Traffic split: {Math.round(test.trafficSplit * 100)}% to challenger |
                              Started: {new Date(test.startedAt).toLocaleDateString()}
                            </p>
                          </div>
                          <button
                            onClick={() => handleConcludeTest(test.id)}
                            className="px-4 py-2 text-xs bg-brand-600 text-white rounded-lg hover:bg-brand-700 transition-colors"
                          >
                            Conclude
                          </button>
                        </div>
                        {test.results && (
                          <div className="grid grid-cols-2 gap-4 mt-3 pt-3 border-t border-white/5">
                            <div>
                              <p className="text-xs text-gray-500">Champion: {test.results.championRequests} requests</p>
                              <p className="text-sm text-white">{(test.results.championAccuracy * 100).toFixed(1)}% accuracy</p>
                            </div>
                            <div>
                              <p className="text-xs text-gray-500">Challenger: {test.results.challengerRequests} requests</p>
                              <p className="text-sm text-white">{(test.results.challengerAccuracy * 100).toFixed(1)}% accuracy</p>
                            </div>
                          </div>
                        )}
                      </div>
                    ))}
                  </div>
                )}

                {/* Start New Test */}
                {registry.championId && registry.versions.filter((v) => v.status === "challenger" || v.status === "staging").length > 0 && (
                  <div>
                    <h3 className="text-xs text-gray-500 uppercase tracking-wider font-medium mb-2">Start New Test</h3>
                    <div className="flex items-end gap-3">
                      <label className="flex-1">
                        <span className="text-xs text-gray-500">Test Name</span>
                        <input
                          value={abTestName}
                          onChange={(e) => setAbTestName(e.target.value)}
                          placeholder="e.g., v1.2 accuracy test"
                          className="mt-1 w-full bg-surface-3 border border-white/10 rounded-lg px-3 py-2 text-sm text-white focus:outline-none focus:border-brand-500"
                        />
                      </label>
                      <label>
                        <span className="text-xs text-gray-500">Traffic %</span>
                        <input
                          type="number"
                          min={1}
                          max={50}
                          value={Math.round(abTrafficSplit * 100)}
                          onChange={(e) => setAbTrafficSplit(Number(e.target.value) / 100)}
                          className="mt-1 w-20 bg-surface-3 border border-white/10 rounded-lg px-3 py-2 text-sm text-white focus:outline-none focus:border-brand-500"
                        />
                      </label>
                      {registry.versions.filter((v) => v.status === "challenger").map((v) => (
                        <button
                          key={v.id}
                          onClick={() => handleStartABTest(v.id)}
                          className="px-4 py-2 bg-brand-600 text-white text-sm rounded-lg hover:bg-brand-700 transition-colors"
                        >
                          Test v{v.semver}
                        </button>
                      ))}
                    </div>
                  </div>
                )}

                {/* Past Tests */}
                {registry.abTests.filter((t) => t.status !== "running").length > 0 && (
                  <div className="mt-4 pt-4 border-t border-white/5">
                    <h3 className="text-xs text-gray-500 uppercase tracking-wider font-medium mb-2">Past Tests</h3>
                    {registry.abTests.filter((t) => t.status !== "running").map((test) => (
                      <div key={test.id} className="bg-surface-3/30 rounded-lg p-3 mb-2 flex items-center gap-4">
                        <span className={`text-xs px-2 py-0.5 rounded-full ${
                          test.status === "concluded" ? "bg-green-500/15 text-green-400" : "bg-red-500/15 text-red-400"
                        }`}>
                          {test.status}
                        </span>
                        <span className="text-sm text-gray-300">{test.name}</span>
                        {test.results && (
                          <span className={`ml-auto text-xs font-medium ${
                            test.results.winner === "challenger" ? "text-green-400" : test.results.winner === "champion" ? "text-yellow-400" : "text-gray-500"
                          }`}>
                            Winner: {test.results.winner}
                          </span>
                        )}
                      </div>
                    ))}
                  </div>
                )}

                {registry.versions.length <= 1 && (
                  <p className="text-sm text-gray-500">Train multiple model versions to enable A/B testing.</p>
                )}
              </section>
            </>
          )}
        </div>
      </main>
    </div>
  );
}
