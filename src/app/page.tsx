"use client";

import { useEffect, useState, useMemo } from "react";
import Sidebar from "@/components/Sidebar";
import StatsCard from "@/components/StatsCard";
import { loadData, loadEnsembleModel } from "@/lib/store";
import { loadRegistry } from "@/lib/enterprise/model-registry";
import { getDriftReport, loadDriftState } from "@/lib/enterprise/drift-detector";
import type { TrainingData } from "@/types";
import type { EnsembleModel } from "@/lib/engine/ensemble";

export default function Dashboard() {
  const [data, setData] = useState<TrainingData | null>(null);
  const [model, setModel] = useState<EnsembleModel | null>(null);

  useEffect(() => {
    setData(loadData());
    setModel(loadEnsembleModel());
  }, []);

  const stats = useMemo(() => {
    if (!data) return null;
    const totalExamples = data.intents.reduce((s, i) => s + i.examples.length, 0);
    const avgExamples = data.intents.length > 0 ? Math.round(totalExamples / data.intents.length) : 0;
    return { totalExamples, avgExamples };
  }, [data]);

  const registry = useMemo(() => {
    if (typeof window === "undefined") return null;
    return loadRegistry();
  }, []);

  const driftReport = useMemo(() => {
    if (typeof window === "undefined") return null;
    const state = loadDriftState();
    return getDriftReport(state);
  }, []);

  if (!data || !stats) return null;

  const driftStatusColor = driftReport?.overallStatus === "healthy" ? "#40c057"
    : driftReport?.overallStatus === "warning" ? "#f59f00" : "#fa5252";

  return (
    <div className="flex">
      <Sidebar />
      <main id="main-content" className="ml-64 flex-1 min-h-screen p-8" role="main">
        {/* Header */}
        <header className="mb-8">
          <h1 className="text-3xl font-bold text-white tracking-tight">Dashboard</h1>
          <p className="text-gray-500 mt-1">Enterprise NLU Bot Trainer — Overview</p>
        </header>

        {/* Stats Grid */}
        <section aria-label="Project statistics" className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-8">
          <StatsCard
            label="Intents"
            value={data.intents.length}
            color="#4c6ef5"
            icon={
              <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5} aria-hidden="true">
                <path strokeLinecap="round" strokeLinejoin="round" d="M9.568 3H5.25A2.25 2.25 0 003 5.25v4.318c0 .597.237 1.17.659 1.591l9.581 9.581c.699.699 1.78.872 2.607.33a18.095 18.095 0 005.223-5.223c.542-.827.369-1.908-.33-2.607L11.16 3.66A2.25 2.25 0 009.568 3z" />
              </svg>
            }
            subtitle="Intent categories"
          />
          <StatsCard
            label="Training Examples"
            value={stats.totalExamples}
            color="#be4bdb"
            icon={
              <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5} aria-hidden="true">
                <path strokeLinecap="round" strokeLinejoin="round" d="M7.5 8.25h9m-9 3H12m-9.75 1.51c0 1.6 1.123 2.994 2.707 3.227 1.129.166 2.27.293 3.423.379.35.026.67.21.865.501L12 21l2.755-4.133a1.14 1.14 0 01.865-.501 48.172 48.172 0 003.423-.379c1.584-.233 2.707-1.626 2.707-3.228V6.741c0-1.602-1.123-2.995-2.707-3.228A48.394 48.394 0 0012 3c-2.392 0-4.744.175-7.043.513C3.373 3.746 2.25 5.14 2.25 6.741v6.018z" />
              </svg>
            }
            subtitle={`~${stats.avgExamples} per intent`}
          />
          <StatsCard
            label="Model Accuracy"
            value={model ? `${Math.round(model.metrics.accuracy * 100)}%` : "—"}
            color={model ? (model.metrics.accuracy >= 0.9 ? "#40c057" : "#f59f00") : "#fa5252"}
            icon={
              <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5} aria-hidden="true">
                <path strokeLinecap="round" strokeLinejoin="round" d="M9.75 3.104v5.714a2.25 2.25 0 01-.659 1.591L5 14.5M9.75 3.104c-.251.023-.501.05-.75.082m.75-.082a24.301 24.301 0 014.5 0m0 0v5.714c0 .597.237 1.17.659 1.591L19 14.5" />
              </svg>
            }
            subtitle={model ? `Trained ${new Date(model.trainedAt).toLocaleDateString()}` : "Not yet trained"}
          />
          <StatsCard
            label="Model Versions"
            value={registry?.versions.length || 0}
            color="#15aabf"
            icon={
              <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5} aria-hidden="true">
                <path strokeLinecap="round" strokeLinejoin="round" d="M20.25 6.375c0 2.278-3.694 4.125-8.25 4.125S3.75 8.653 3.75 6.375m16.5 0c0-2.278-3.694-4.125-8.25-4.125S3.75 4.097 3.75 6.375m16.5 0v11.25c0 2.278-3.694 4.125-8.25 4.125s-8.25-1.847-8.25-4.125V6.375" />
              </svg>
            }
            subtitle={registry?.championId ? "Champion deployed" : "No champion yet"}
          />
        </section>

        {/* Two-column layout */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-6">
          {/* Intent Distribution */}
          <section aria-label="Intent distribution" className="glass rounded-xl p-6">
            <h2 className="text-lg font-semibold text-white mb-4">Intent Distribution</h2>
            <div className="space-y-2.5" role="list">
              {data.intents.map((intent) => {
                const pct = stats.totalExamples > 0 ? (intent.examples.length / stats.totalExamples) * 100 : 0;
                return (
                  <div key={intent.id} className="flex items-center gap-3" role="listitem">
                    <span
                      className="w-2 h-2 rounded-full flex-shrink-0"
                      style={{ backgroundColor: intent.color }}
                      aria-hidden="true"
                    />
                    <span className="text-sm text-gray-300 w-32 truncate">{intent.name}</span>
                    <div className="flex-1 h-1.5 bg-surface-3 rounded-full overflow-hidden" role="progressbar" aria-valuenow={Math.round(pct)} aria-valuemin={0} aria-valuemax={100} aria-label={`${intent.name}: ${intent.examples.length} examples`}>
                      <div
                        className="h-full rounded-full transition-all duration-500"
                        style={{ width: `${pct}%`, backgroundColor: intent.color }}
                      />
                    </div>
                    <span className="text-xs text-gray-500 w-12 text-right tabular-nums">
                      {intent.examples.length}
                    </span>
                  </div>
                );
              })}
            </div>
          </section>

          {/* Engine Info + Drift Status */}
          <section aria-label="Engine information" className="glass rounded-xl p-6">
            <h2 className="text-lg font-semibold text-white mb-3">Engine: Ensemble v3</h2>
            <p className="text-sm text-gray-400 mb-4 leading-relaxed">
              5-classifier stacking ensemble with cross-validated meta-weights. Logistic Regression, Complement Naive Bayes,
              Linear SVM (Pegasos), MLP Neural Network, and Gradient Boosted Stumps — combined through learned weighted averaging.
            </p>
            <div className="grid grid-cols-2 gap-3 mb-4">
              <div className="bg-surface-3/50 rounded-lg p-3">
                <p className="text-xs text-gray-500 mb-1">Architecture</p>
                <p className="text-sm font-medium text-white">5-Model Stacking</p>
              </div>
              <div className="bg-surface-3/50 rounded-lg p-3">
                <p className="text-xs text-gray-500 mb-1">Features</p>
                <p className="text-sm font-medium text-white">MurmurHash3 1024-d</p>
              </div>
              <div className="bg-surface-3/50 rounded-lg p-3">
                <p className="text-xs text-gray-500 mb-1">Export Formats</p>
                <p className="text-sm font-medium text-white">7 (Rasa, LUIS...)</p>
              </div>
              <div className="bg-surface-3/50 rounded-lg p-3">
                <p className="text-xs text-gray-500 mb-1">Drift Status</p>
                <p className="text-sm font-medium" style={{ color: driftStatusColor }}>
                  {driftReport?.overallStatus === "healthy" ? "Healthy" : driftReport?.overallStatus === "warning" ? "Warning" : driftReport?.predictionCount === 0 ? "No data" : "Critical"}
                </p>
              </div>
            </div>
            {model && (
              <div className="pt-3 border-t border-white/5">
                <p className="text-xs text-gray-500">
                  Training time: {model.metrics.trainingTimeMs.toFixed(0)}ms | Vocab: {model.metrics.vocabularySize.toLocaleString()} | Examples: {model.metrics.totalExamples}
                </p>
              </div>
            )}
          </section>
        </div>

        {/* Quick Actions */}
        <section aria-label="Quick actions" className="grid grid-cols-1 md:grid-cols-3 lg:grid-cols-4 gap-4">
          <a href="/train" className="glass rounded-xl p-5 hover:border-green-500/30 transition-all group cursor-pointer">
            <h3 className="font-semibold text-white group-hover:text-green-400 transition-colors">Train Model</h3>
            <p className="text-xs text-gray-500 mt-1">Train the ensemble classifier on your data</p>
          </a>
          <a href="/test" className="glass rounded-xl p-5 hover:border-purple-500/30 transition-all group cursor-pointer">
            <h3 className="font-semibold text-white group-hover:text-purple-400 transition-colors">Test Playground</h3>
            <p className="text-xs text-gray-500 mt-1">Real-time intent classification with per-model breakdown</p>
          </a>
          <a href="/self-learn" className="glass rounded-xl p-5 hover:border-brand-500/30 transition-all group cursor-pointer">
            <h3 className="font-semibold text-white group-hover:text-brand-400 transition-colors">Self-Learn</h3>
            <p className="text-xs text-gray-500 mt-1">Autonomous recursive improvement loop</p>
          </a>
          <a href="/analytics" className="glass rounded-xl p-5 hover:border-orange-500/30 transition-all group cursor-pointer">
            <h3 className="font-semibold text-white group-hover:text-orange-400 transition-colors">Analytics</h3>
            <p className="text-xs text-gray-500 mt-1">Confusion matrix, F1 scores, drift monitoring</p>
          </a>
        </section>
      </main>
    </div>
  );
}
