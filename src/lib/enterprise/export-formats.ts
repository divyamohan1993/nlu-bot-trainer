/**
 * Multi-Format Export/Import Engine
 *
 * Supports: Rasa, Dialogflow ES, Amazon Lex, LUIS, Wit.ai, JSON, CSV
 * Cross-platform interoperability for enterprise NLU deployments
 */

import type { TrainingData, Intent, Entity } from "@/types";

export type ExportFormat = "rasa" | "dialogflow" | "lex" | "luis" | "wit" | "json" | "csv";

/**
 * Rasa NLU YAML format (v3.1)
 */
export function exportRasa(data: TrainingData): string {
  const lines: string[] = ['version: "3.1"', "nlu:"];

  for (const intent of data.intents) {
    lines.push(`  - intent: ${intent.name}`);
    lines.push("    examples: |");
    for (const ex of intent.examples) {
      let text = ex.text;
      const sorted = [...ex.entities].sort((a, b) => b.start - a.start);
      for (const entity of sorted) {
        text = text.slice(0, entity.start) + `[${entity.value}](${entity.entity})` + text.slice(entity.end);
      }
      lines.push(`      - ${text}`);
    }
  }

  for (const entity of data.entities) {
    if (entity.values.length > 0) {
      lines.push(`  - lookup: ${entity.name}`);
      lines.push("    examples: |");
      for (const val of entity.values) {
        lines.push(`      - ${val}`);
      }
    }
  }

  return lines.join("\n");
}

/**
 * Dialogflow ES JSON format
 */
export function exportDialogflow(data: TrainingData): object {
  return {
    intents: data.intents.map((intent) => ({
      name: intent.name,
      auto: true,
      contexts: [],
      responses: [{ resetContexts: false, parameters: [], messages: [{ type: 0, lang: "en", speech: [] }] }],
      priority: 500000,
      userSays: intent.examples.map((ex) => ({
        id: ex.id,
        data: ex.entities.length > 0
          ? buildDialogflowParts(ex.text, ex.entities)
          : [{ text: ex.text, userDefined: false }],
        isTemplate: false,
        count: 0,
      })),
    })),
    entities: data.entities.map((entity) => ({
      name: entity.name,
      isOverridable: true,
      isEnum: false,
      automatedExpansion: true,
      entries: entity.values.map((v) => ({ value: v, synonyms: [v] })),
    })),
  };
}

function buildDialogflowParts(text: string, entities: Array<{ start: number; end: number; value: string; entity: string }>) {
  const sorted = [...entities].sort((a, b) => a.start - b.start);
  const parts: Array<{ text: string; alias?: string; meta?: string; userDefined?: boolean }> = [];
  let pos = 0;

  for (const ent of sorted) {
    if (ent.start > pos) {
      parts.push({ text: text.slice(pos, ent.start), userDefined: false });
    }
    parts.push({ text: text.slice(ent.start, ent.end), alias: ent.entity, meta: `@${ent.entity}`, userDefined: true });
    pos = ent.end;
  }
  if (pos < text.length) {
    parts.push({ text: text.slice(pos), userDefined: false });
  }

  return parts;
}

/**
 * Amazon Lex V2 JSON format
 */
export function exportLex(data: TrainingData): object {
  return {
    metadata: { schemaVersion: "1.0", fileFormat: "LexJson", resourceType: "Bot" },
    resource: {
      name: "NLUBotTrainer",
      description: "Exported from NLU Bot Trainer",
      locale: "en_US",
      intents: data.intents.map((intent) => ({
        name: intent.name,
        description: intent.description,
        sampleUtterances: intent.examples.map((ex) => ({ utterance: ex.text })),
      })),
      slotTypes: data.entities.map((entity) => ({
        name: entity.name,
        description: entity.description,
        slotTypeValues: entity.values.map((v) => ({
          sampleValue: { value: v },
          synonyms: [{ value: v }],
        })),
        valueSelectionSetting: { resolutionStrategy: "TopResolution" },
      })),
    },
  };
}

/**
 * Microsoft LUIS JSON format
 */
export function exportLUIS(data: TrainingData): object {
  return {
    luis_schema_version: "7.0.0",
    versionId: data.metadata.version,
    name: "NLU-Bot-Trainer",
    desc: "Exported from NLU Bot Trainer",
    culture: "en-us",
    intents: [
      ...data.intents.map((i) => ({ name: i.name })),
      { name: "None" },
    ],
    entities: data.entities.map((e) => ({ name: e.name, roles: [] })),
    closedLists: data.entities.map((entity) => ({
      name: entity.name,
      subLists: entity.values.map((v) => ({ canonicalForm: v, list: [v] })),
      roles: [],
    })),
    utterances: data.intents.flatMap((intent) =>
      intent.examples.map((ex) => ({
        text: ex.text,
        intent: intent.name,
        entities: ex.entities.map((e) => ({
          entity: e.entity,
          startPos: e.start,
          endPos: e.end - 1,
        })),
      }))
    ),
    patterns: [],
    regex_entities: [],
    prebuiltEntities: [],
    composites: [],
    patternAnyEntities: [],
    phraselists: [],
    regex_features: [],
    settings: [],
  };
}

/**
 * Wit.ai JSON format
 */
export function exportWit(data: TrainingData): object {
  return {
    intents: data.intents.map((i) => ({ id: i.id, name: i.name })),
    entities: data.entities.map((entity) => ({
      id: entity.id,
      name: `${entity.name}:${entity.name}`,
      roles: [entity.name],
      lookups: ["free-text", "keywords"],
      keywords: entity.values.map((v) => ({ keyword: v, synonyms: [v] })),
    })),
    utterances: data.intents.flatMap((intent) =>
      intent.examples.map((ex) => ({
        text: ex.text,
        intent: intent.name,
        entities: ex.entities.map((e) => ({
          entity: `${e.entity}:${e.entity}`,
          start: e.start,
          end: e.end,
          body: e.value,
          value: e.value,
          role: e.entity,
        })),
        traits: [],
      }))
    ),
  };
}

/**
 * CSV format for data analysis
 */
export function exportCSV(data: TrainingData): string {
  const lines: string[] = ["text,intent,entities"];
  for (const intent of data.intents) {
    for (const ex of intent.examples) {
      const entities = ex.entities.map((e) => `${e.entity}:${e.value}`).join("|");
      const escapedText = `"${ex.text.replace(/"/g, '""')}"`;
      lines.push(`${escapedText},${intent.name},${entities || ""}`);
    }
  }
  return lines.join("\n");
}

/**
 * Universal export function
 */
export function exportTrainingData(data: TrainingData, format: ExportFormat): { content: string; filename: string; mimeType: string } {
  switch (format) {
    case "rasa":
      return { content: exportRasa(data), filename: "nlu-training-data.yml", mimeType: "text/yaml" };
    case "dialogflow":
      return { content: JSON.stringify(exportDialogflow(data), null, 2), filename: "dialogflow-export.json", mimeType: "application/json" };
    case "lex":
      return { content: JSON.stringify(exportLex(data), null, 2), filename: "lex-export.json", mimeType: "application/json" };
    case "luis":
      return { content: JSON.stringify(exportLUIS(data), null, 2), filename: "luis-export.json", mimeType: "application/json" };
    case "wit":
      return { content: JSON.stringify(exportWit(data), null, 2), filename: "wit-export.json", mimeType: "application/json" };
    case "csv":
      return { content: exportCSV(data), filename: "nlu-training-data.csv", mimeType: "text/csv" };
    case "json":
    default:
      return {
        content: JSON.stringify({
          intents: data.intents.map((i) => ({
            name: i.name, description: i.description,
            examples: i.examples.map((e) => ({ text: e.text, entities: e.entities })),
          })),
          entities: data.entities.map((e) => ({
            name: e.name, description: e.description, values: e.values,
          })),
          metadata: { ...data.metadata, exportedAt: new Date().toISOString(), format: "nlu-bot-trainer-v2" },
        }, null, 2),
        filename: "nlu-training-data.json",
        mimeType: "application/json",
      };
  }
}

/**
 * Import from JSON format
 */
export function importFromJSON(json: string): Partial<TrainingData> | null {
  try {
    const data = JSON.parse(json);
    if (data.intents && Array.isArray(data.intents)) return data;
    if (data.nlu && Array.isArray(data.nlu)) {
      // Rasa format
      return {
        intents: data.nlu.map((item: { intent: string; examples: string }, idx: number) => ({
          id: `imported_${idx}`,
          name: item.intent,
          description: "",
          color: `hsl(${(idx * 137) % 360}, 70%, 50%)`,
          examples: (item.examples || "").split("\n")
            .filter((l: string) => l.trim().startsWith("- "))
            .map((l: string, i: number) => ({
              id: `ex_${idx}_${i}`,
              text: l.trim().slice(2).replace(/\[([^\]]+)\]\([^)]+\)/g, "$1"),
              intent: item.intent,
              entities: [],
            })),
        })),
        entities: [],
      };
    }
    return null;
  } catch {
    return null;
  }
}
