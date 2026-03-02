"use client";

import { useEffect, useState, useRef, useMemo } from "react";
import Sidebar from "@/components/Sidebar";
import ConfidenceBar from "@/components/ConfidenceBar";
import { loadData, loadEnsembleModel } from "@/lib/store";
import { predictEnsemble, type EnsembleModel } from "@/lib/engine/ensemble";
import { extractEntities } from "@/lib/entity-extractor";
import type { TrainingData, PredictionResult } from "@/types";

interface ChatMessage {
  id: string;
  text: string;
  type: "user" | "bot";
  prediction?: PredictionResult;
  inferenceTimeUs?: number;
  perModelScores?: {
    logReg: Array<{ intent: string; score: number }>;
    naiveBayes: Array<{ intent: string; score: number }>;
    svm: Array<{ intent: string; score: number }>;
    mlp: Array<{ intent: string; score: number }>;
    gradBoost: Array<{ intent: string; score: number }>;
  };
}

const SUGGESTIONS = [
  "Hello there!",
  "Where is my order?",
  "I want to return this laptop",
  "Can I get a refund?",
  "This product is defective",
  "I need help with my account",
  "Connect me to a human",
  "What colors are available?",
  "Cancel my order please",
  "My payment failed",
  "Thank you so much",
  "Goodbye",
];

export default function TestPage() {
  const [data, setData] = useState<TrainingData | null>(null);
  const [model, setModel] = useState<EnsembleModel | null>(null);
  const [input, setInput] = useState("");
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [threshold, setThreshold] = useState(0.65);
  const [showModelBreakdown, setShowModelBreakdown] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    setData(loadData());
    setModel(loadEnsembleModel());
  }, []);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const intentColorMap = useMemo(() => {
    const map: Record<string, string> = {};
    data?.intents.forEach((i) => { map[i.name] = i.color; });
    return map;
  }, [data]);

  const entityColorMap = useMemo(() => {
    const map: Record<string, string> = {};
    data?.entities.forEach((e) => { map[e.name] = e.color; });
    return map;
  }, [data]);

  const handlePredict = (textOverride?: string) => {
    const text = (textOverride || input).trim();
    if (!text || !model || !data) return;

    const result = predictEnsemble(text, model);
    const entities = extractEntities(text, data.entities);

    const userMsg: ChatMessage = { id: `msg_${Date.now()}_u`, text, type: "user" };
    const prediction: PredictionResult = {
      text,
      intent: { name: result.intent, confidence: result.confidence },
      intent_ranking: result.ranking,
      entities,
    };
    const botMsg: ChatMessage = {
      id: `msg_${Date.now()}_b`,
      text: "",
      type: "bot",
      prediction,
      inferenceTimeUs: result.inferenceTimeUs,
      perModelScores: result.perModelScores,
    };

    setMessages((prev) => [...prev, userMsg, botMsg]);
    setInput("");
    inputRef.current?.focus();
  };

  const clearChat = () => {
    setMessages([]);
    inputRef.current?.focus();
  };

  if (!data) return null;

  return (
    <div className="flex">
      <Sidebar />
      <main id="main-content" className="ml-64 flex-1 min-h-screen flex" role="main">
        <div className="flex-1 flex flex-col">
          {/* Header */}
          <header className="p-6 border-b border-white/5 flex items-center justify-between">
            <div>
              <h1 className="text-xl font-bold text-white tracking-tight">Test Playground</h1>
              <p className="text-xs text-gray-500 mt-0.5">
                {model ? `Ensemble v2 — trained ${new Date(model.trainedAt).toLocaleDateString()}` : "No trained model — go to Train first"}
              </p>
            </div>
            <div className="flex items-center gap-3">
              <label className="flex items-center gap-1.5">
                <input
                  type="checkbox"
                  checked={showModelBreakdown}
                  onChange={(e) => setShowModelBreakdown(e.target.checked)}
                  className="w-3.5 h-3.5 accent-brand-500"
                />
                <span className="text-xs text-gray-500">Per-model scores</span>
              </label>
              <label className="flex items-center gap-2">
                <span className="text-xs text-gray-500">Threshold</span>
                <input
                  type="range"
                  min="0"
                  max="100"
                  value={threshold * 100}
                  onChange={(e) => setThreshold(Number(e.target.value) / 100)}
                  className="w-20 h-1 accent-brand-500"
                  aria-label="Confidence threshold"
                />
                <span className="text-xs text-gray-400 font-mono w-8 tabular-nums">{Math.round(threshold * 100)}%</span>
              </label>
              <button
                onClick={clearChat}
                className="px-3 py-1.5 text-xs text-gray-500 hover:text-gray-300 bg-surface-3 hover:bg-surface-4 rounded-lg transition-colors"
              >
                Clear
              </button>
            </div>
          </header>

          {/* Messages */}
          <div className="flex-1 overflow-y-auto p-6 space-y-4" role="log" aria-label="Conversation" aria-live="polite">
            {!model && (
              <div className="flex items-center justify-center h-full">
                <div className="text-center" role="status">
                  <div className="w-16 h-16 rounded-2xl bg-surface-3 flex items-center justify-center mx-auto mb-4" aria-hidden="true">
                    <svg className="w-8 h-8 text-gray-600" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1}>
                      <path strokeLinecap="round" strokeLinejoin="round" d="M9.75 3.104v5.714a2.25 2.25 0 01-.659 1.591L5 14.5M9.75 3.104c-.251.023-.501.05-.75.082m.75-.082a24.301 24.301 0 014.5 0m0 0v5.714c0 .597.237 1.17.659 1.591L19 14.5" />
                    </svg>
                  </div>
                  <p className="text-gray-400 font-medium">Model not trained yet</p>
                  <p className="text-sm text-gray-600 mt-1">
                    Go to the <a href="/train" className="text-brand-400 hover:underline focus:underline">Train</a> page first
                  </p>
                </div>
              </div>
            )}

            {model && messages.length === 0 && (
              <div className="flex items-center justify-center h-full">
                <div className="text-center max-w-lg">
                  <p className="text-gray-500 text-sm mb-4">Type a message or try a suggestion:</p>
                  <div className="flex flex-wrap justify-center gap-2" role="group" aria-label="Quick test suggestions">
                    {SUGGESTIONS.map((s) => (
                      <button
                        key={s}
                        onClick={() => handlePredict(s)}
                        className="px-3 py-1.5 bg-surface-3 hover:bg-surface-4 text-gray-400 hover:text-gray-200 text-xs rounded-full transition-colors"
                      >
                        {s}
                      </button>
                    ))}
                  </div>
                </div>
              </div>
            )}

            {messages.map((msg) => (
              <div
                key={msg.id}
                className={`flex ${msg.type === "user" ? "justify-end" : "justify-start"}`}
              >
                {msg.type === "user" ? (
                  <div className="max-w-md px-4 py-2.5 rounded-2xl rounded-br-sm bg-brand-600 text-white text-sm">
                    {msg.text}
                  </div>
                ) : msg.prediction ? (
                  <div className="max-w-lg w-full" role="status" aria-label={`Predicted intent: ${msg.prediction.intent.name} at ${Math.round(msg.prediction.intent.confidence * 100)}% confidence`}>
                    <div className="glass rounded-2xl rounded-bl-sm p-4">
                      {/* Top intent + inference time */}
                      <div className="flex items-center gap-2 mb-3">
                        <span className="w-2 h-2 rounded-full" style={{ backgroundColor: intentColorMap[msg.prediction.intent.name] || "#4c6ef5" }} aria-hidden="true" />
                        <span className="text-sm font-semibold text-white">{msg.prediction.intent.name}</span>
                        <span className={`ml-auto text-xs font-mono px-2 py-0.5 rounded-full ${
                          msg.prediction.intent.confidence >= threshold ? "bg-green-500/15 text-green-400" : "bg-yellow-500/15 text-yellow-400"
                        }`}>
                          {Math.round(msg.prediction.intent.confidence * 100)}%
                        </span>
                        {msg.inferenceTimeUs !== undefined && (
                          <span className="text-[10px] text-gray-600 font-mono">
                            {msg.inferenceTimeUs < 1000
                              ? `${Math.round(msg.inferenceTimeUs)}us`
                              : `${(msg.inferenceTimeUs / 1000).toFixed(1)}ms`}
                          </span>
                        )}
                      </div>

                      {/* Ranking bars */}
                      <div className="space-y-1.5">
                        {msg.prediction.intent_ranking.slice(0, 5).map((r, i) => (
                          <ConfidenceBar key={r.name} label={r.name} confidence={r.confidence} color={intentColorMap[r.name] || "#4c6ef5"} isTop={i === 0} />
                        ))}
                      </div>

                      {/* Per-model breakdown */}
                      {showModelBreakdown && msg.perModelScores && (
                        <div className="mt-3 pt-3 border-t border-white/5">
                          <p className="text-xs text-gray-500 mb-2">Per-model top prediction:</p>
                          <div className="grid grid-cols-5 gap-2">
                            {[
                              { label: "LogReg", scores: msg.perModelScores.logReg, color: "#4c6ef5" },
                              { label: "CNB", scores: msg.perModelScores.naiveBayes, color: "#be4bdb" },
                              { label: "SVM", scores: msg.perModelScores.svm, color: "#f76707" },
                              { label: "MLP", scores: msg.perModelScores.mlp, color: "#e64980" },
                              { label: "GBM", scores: msg.perModelScores.gradBoost, color: "#40c057" },
                            ].map((m) => (
                              <div key={m.label} className="bg-surface-3/50 rounded p-2">
                                <p className="text-[10px] font-mono" style={{ color: m.color }}>{m.label}</p>
                                <p className="text-xs text-white truncate">{m.scores[0]?.intent}</p>
                                <p className="text-[10px] text-gray-500">{Math.round((m.scores[0]?.score || 0) * 100)}%</p>
                              </div>
                            ))}
                          </div>
                        </div>
                      )}

                      {/* Entities */}
                      {msg.prediction.entities.length > 0 && (
                        <div className="mt-3 pt-3 border-t border-white/5">
                          <p className="text-xs text-gray-500 mb-2">Entities found:</p>
                          <div className="flex flex-wrap gap-1.5">
                            {msg.prediction.entities.map((entity, i) => (
                              <span
                                key={i}
                                className="inline-flex items-center gap-1 px-2 py-1 rounded text-xs font-medium"
                                style={{
                                  backgroundColor: `${entityColorMap[entity.entity] || "#748ffc"}15`,
                                  color: entityColorMap[entity.entity] || "#748ffc",
                                  border: `1px solid ${entityColorMap[entity.entity] || "#748ffc"}30`,
                                }}
                              >
                                <span className="opacity-60">{entity.entity}:</span>
                                {entity.value}
                              </span>
                            ))}
                          </div>
                        </div>
                      )}

                      {msg.prediction.intent.confidence < threshold && (
                        <div className="mt-3 pt-3 border-t border-white/5">
                          <p className="text-xs text-yellow-400" role="alert">
                            Below confidence threshold ({Math.round(threshold * 100)}%). Consider adding more training examples.
                          </p>
                        </div>
                      )}
                    </div>
                  </div>
                ) : null}
              </div>
            ))}
            <div ref={messagesEndRef} />
          </div>

          {/* Input */}
          <div className="p-4 border-t border-white/5">
            <form
              onSubmit={(e) => { e.preventDefault(); handlePredict(); }}
              className="flex gap-2"
              role="search"
              aria-label="Test a message"
            >
              <input
                ref={inputRef}
                value={input}
                onChange={(e) => setInput(e.target.value)}
                placeholder={model ? "Type a message to test..." : "Train a model first"}
                disabled={!model}
                className="flex-1 bg-surface-2 border border-white/10 rounded-xl px-4 py-3 text-sm text-white placeholder-gray-600 focus:outline-none focus:border-brand-500 transition-colors disabled:opacity-50"
                aria-label="Test message input"
              />
              <button
                type="submit"
                disabled={!model || !input.trim()}
                className="px-6 py-3 bg-brand-600 hover:bg-brand-700 disabled:opacity-30 disabled:cursor-not-allowed text-white text-sm font-semibold rounded-xl transition-colors"
              >
                Send
              </button>
            </form>
          </div>
        </div>
      </main>
    </div>
  );
}
