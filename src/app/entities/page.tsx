"use client";

import { useEffect, useState, useCallback } from "react";
import Sidebar from "@/components/Sidebar";
import EmptyState from "@/components/EmptyState";
import { loadData, saveData, getNextEntityColor } from "@/lib/store";
import type { TrainingData, Entity } from "@/types";
import { v4 as uuidv4 } from "uuid";

export default function EntitiesPage() {
  const [data, setData] = useState<TrainingData | null>(null);
  const [selectedEntity, setSelectedEntity] = useState<string | null>(null);
  const [showCreateModal, setShowCreateModal] = useState(false);
  const [newEntityName, setNewEntityName] = useState("");
  const [newEntityDesc, setNewEntityDesc] = useState("");
  const [newValue, setNewValue] = useState("");

  useEffect(() => {
    const loaded = loadData();
    setData(loaded);
    if (loaded.entities.length > 0) {
      setSelectedEntity(loaded.entities[0].id);
    }
  }, []);

  const persist = useCallback((updated: TrainingData) => {
    setData(updated);
    saveData(updated);
  }, []);

  const createEntity = () => {
    if (!data || !newEntityName.trim()) return;
    const entity: Entity = {
      id: `entity_${uuidv4().slice(0, 8)}`,
      name: newEntityName.trim().toLowerCase().replace(/\s+/g, "_"),
      description: newEntityDesc.trim(),
      values: [],
      color: getNextEntityColor(data.entities),
    };
    const updated = { ...data, entities: [...data.entities, entity] };
    persist(updated);
    setSelectedEntity(entity.id);
    setNewEntityName("");
    setNewEntityDesc("");
    setShowCreateModal(false);
  };

  const deleteEntity = (entityId: string) => {
    if (!data) return;
    const updated = {
      ...data,
      entities: data.entities.filter((e) => e.id !== entityId),
    };
    persist(updated);
    if (selectedEntity === entityId) {
      setSelectedEntity(updated.entities[0]?.id || null);
    }
  };

  const addValue = () => {
    if (!data || !selectedEntity || !newValue.trim()) return;
    const updated = {
      ...data,
      entities: data.entities.map((e) =>
        e.id === selectedEntity
          ? { ...e, values: [...e.values, newValue.trim()] }
          : e
      ),
    };
    persist(updated);
    setNewValue("");
  };

  const deleteValue = (value: string) => {
    if (!data || !selectedEntity) return;
    const updated = {
      ...data,
      entities: data.entities.map((e) =>
        e.id === selectedEntity
          ? { ...e, values: e.values.filter((v) => v !== value) }
          : e
      ),
    };
    persist(updated);
  };

  if (!data) return null;

  const currentEntity = data.entities.find((e) => e.id === selectedEntity);

  return (
    <div className="flex">
      <Sidebar />
      <main id="main-content" className="ml-64 flex-1 min-h-screen" role="main">
        <div className="flex h-screen">
          {/* Entity List */}
          <div className="w-72 border-r border-white/5 bg-surface-1/50 flex flex-col">
            <div className="p-4 border-b border-white/5">
              <div className="flex items-center justify-between mb-3">
                <h2 className="text-sm font-semibold text-white">Entities</h2>
                <button
                  onClick={() => setShowCreateModal(true)}
                  className="w-7 h-7 rounded-lg bg-brand-600 hover:bg-brand-700 flex items-center justify-center transition-colors"
                >
                  <svg className="w-4 h-4 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                    <path strokeLinecap="round" strokeLinejoin="round" d="M12 4.5v15m7.5-7.5h-15" />
                  </svg>
                </button>
              </div>
              <p className="text-xs text-gray-500">{data.entities.length} entities defined</p>
            </div>

            <div className="flex-1 overflow-y-auto p-2 space-y-1">
              {data.entities.map((entity) => (
                <button
                  key={entity.id}
                  onClick={() => setSelectedEntity(entity.id)}
                  className={`w-full text-left px-3 py-2.5 rounded-lg transition-all text-sm ${
                    selectedEntity === entity.id
                      ? "bg-surface-3 text-white"
                      : "text-gray-400 hover:bg-surface-2 hover:text-gray-300"
                  }`}
                >
                  <div className="flex items-center gap-2">
                    <span
                      className="w-2 h-2 rounded-full flex-shrink-0"
                      style={{ backgroundColor: entity.color }}
                    />
                    <span className="truncate font-medium">{entity.name}</span>
                    <span className="ml-auto text-xs text-gray-600">
                      {entity.values.length}
                    </span>
                  </div>
                </button>
              ))}
            </div>
          </div>

          {/* Entity Detail */}
          <div className="flex-1 flex flex-col overflow-hidden">
            {currentEntity ? (
              <>
                <div className="p-6 border-b border-white/5">
                  <div className="flex items-start justify-between">
                    <div>
                      <div className="flex items-center gap-3">
                        <span
                          className="w-3 h-3 rounded-full"
                          style={{ backgroundColor: currentEntity.color }}
                        />
                        <h2 className="text-xl font-bold text-white">
                          {currentEntity.name}
                        </h2>
                      </div>
                      <p className="text-sm text-gray-500 mt-1 ml-6">
                        {currentEntity.description || "No description"}
                      </p>
                    </div>
                    <button
                      onClick={() => {
                        if (confirm(`Delete entity "${currentEntity.name}"?`)) {
                          deleteEntity(currentEntity.id);
                        }
                      }}
                      className="p-2 rounded-lg hover:bg-red-500/10 text-gray-500 hover:text-red-400 transition-colors"
                    >
                      <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
                        <path strokeLinecap="round" strokeLinejoin="round" d="M14.74 9l-.346 9m-4.788 0L9.26 9m9.968-3.21c.342.052.682.107 1.022.166m-1.022-.165L18.16 19.673a2.25 2.25 0 01-2.244 2.077H8.084a2.25 2.25 0 01-2.244-2.077L4.772 5.79m14.456 0a48.108 48.108 0 00-3.478-.397m-12 .562c.34-.059.68-.114 1.022-.165m0 0a48.11 48.11 0 013.478-.397m7.5 0v-.916c0-1.18-.91-2.164-2.09-2.201a51.964 51.964 0 00-3.32 0c-1.18.037-2.09 1.022-2.09 2.201v.916m7.5 0a48.667 48.667 0 00-7.5 0" />
                      </svg>
                    </button>
                  </div>
                </div>

                {/* Add Value */}
                <div className="p-4 border-b border-white/5">
                  <form
                    onSubmit={(e) => {
                      e.preventDefault();
                      addValue();
                    }}
                    className="flex gap-2"
                  >
                    <input
                      value={newValue}
                      onChange={(e) => setNewValue(e.target.value)}
                      placeholder='Add entity value, e.g., "New York"'
                      className="flex-1 bg-surface-2 border border-white/10 rounded-lg px-4 py-2.5 text-sm text-white placeholder-gray-600 focus:outline-none focus:border-brand-500 transition-colors"
                    />
                    <button
                      type="submit"
                      disabled={!newValue.trim()}
                      className="px-4 py-2.5 bg-brand-600 hover:bg-brand-700 disabled:opacity-30 disabled:cursor-not-allowed text-white text-sm font-medium rounded-lg transition-colors"
                    >
                      Add
                    </button>
                  </form>
                </div>

                {/* Values List */}
                <div className="flex-1 overflow-y-auto p-4">
                  <h3 className="text-xs text-gray-500 uppercase tracking-wider font-medium mb-3">
                    Values ({currentEntity.values.length})
                  </h3>
                  {currentEntity.values.length === 0 ? (
                    <div className="text-center py-12">
                      <p className="text-sm text-gray-600">
                        No values yet. Add entity values above.
                      </p>
                    </div>
                  ) : (
                    <div className="flex flex-wrap gap-2">
                      {currentEntity.values.map((value) => (
                        <span
                          key={value}
                          className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-sm font-medium animate-fade-in"
                          style={{
                            backgroundColor: `${currentEntity.color}15`,
                            color: currentEntity.color,
                            border: `1px solid ${currentEntity.color}30`,
                          }}
                        >
                          {value}
                          <button
                            onClick={() => deleteValue(value)}
                            className="ml-1 opacity-60 hover:opacity-100 transition-opacity"
                          >
                            <svg className="w-3 h-3" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                              <path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12" />
                            </svg>
                          </button>
                        </span>
                      ))}
                    </div>
                  )}
                </div>
              </>
            ) : (
              <EmptyState
                icon={
                  <svg className="w-8 h-8" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1}>
                    <path strokeLinecap="round" strokeLinejoin="round" d="M5.25 8.25h15m-16.5 7.5h15m-1.8-13.5l-3.9 19.5m-2.1-19.5l-3.9 19.5" />
                  </svg>
                }
                title="No entities yet"
                description="Create entity types to enable named entity recognition in your bot"
                action={{
                  label: "Create Entity",
                  onClick: () => setShowCreateModal(true),
                }}
              />
            )}
          </div>
        </div>

        {/* Create Entity Modal */}
        {showCreateModal && (
          <div className="fixed inset-0 bg-black/60 backdrop-blur-sm flex items-center justify-center z-50 animate-fade-in">
            <div className="bg-surface-2 border border-white/10 rounded-2xl w-full max-w-md p-6 shadow-2xl animate-slide-up">
              <h3 className="text-lg font-bold text-white mb-4">Create New Entity</h3>
              <div className="space-y-4">
                <div>
                  <label className="text-xs text-gray-500 uppercase tracking-wider font-medium">
                    Name
                  </label>
                  <input
                    value={newEntityName}
                    onChange={(e) => setNewEntityName(e.target.value)}
                    placeholder="e.g., city_name"
                    className="w-full mt-1 bg-surface-3 border border-white/10 rounded-lg px-3 py-2 text-white text-sm focus:outline-none focus:border-brand-500"
                    autoFocus
                  />
                </div>
                <div>
                  <label className="text-xs text-gray-500 uppercase tracking-wider font-medium">
                    Description
                  </label>
                  <input
                    value={newEntityDesc}
                    onChange={(e) => setNewEntityDesc(e.target.value)}
                    placeholder="e.g., Names of cities"
                    className="w-full mt-1 bg-surface-3 border border-white/10 rounded-lg px-3 py-2 text-white text-sm focus:outline-none focus:border-brand-500"
                  />
                </div>
              </div>
              <div className="flex justify-end gap-2 mt-6">
                <button
                  onClick={() => {
                    setShowCreateModal(false);
                    setNewEntityName("");
                    setNewEntityDesc("");
                  }}
                  className="px-4 py-2 text-sm text-gray-400 hover:text-gray-300 transition-colors"
                >
                  Cancel
                </button>
                <button
                  onClick={createEntity}
                  disabled={!newEntityName.trim()}
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
