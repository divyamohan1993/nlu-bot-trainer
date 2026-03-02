/**
 * Data Quality Validator
 *
 * Catches training data problems before they waste training time.
 * Inspired by practices from Google's "Data Cascades in ML" (Sambasivan et al., 2021)
 * and the Cleanlab framework for data-centric AI.
 *
 * Checks:
 * 1. Duplicate detection (exact + near-duplicate via Jaccard similarity)
 * 2. Class imbalance warnings
 * 3. Mislabel detection (examples that classify better under a different intent)
 * 4. Vocabulary coverage gaps (classes with few unique tokens)
 * 5. Short/empty examples
 * 6. High overlap between classes (confusable intents)
 * 7. Low-information examples (stop words only)
 */

import { type EnsembleModel, predictEnsemble } from "./ensemble";

export type IssueSeverity = "error" | "warning" | "info";
export type IssueCategory =
  | "duplicate"
  | "near_duplicate"
  | "class_imbalance"
  | "mislabel"
  | "short_example"
  | "empty_example"
  | "low_vocabulary"
  | "class_overlap"
  | "low_information"
  | "special_characters";

export interface DataQualityIssue {
  severity: IssueSeverity;
  category: IssueCategory;
  message: string;
  /** Affected intent name(s) */
  intents: string[];
  /** Affected example text(s) */
  examples: string[];
  /** Suggested fix */
  suggestion: string;
}

export interface DataQualityReport {
  /** Overall quality score (0-100) */
  score: number;
  /** Grade: A (90+), B (75-89), C (60-74), D (40-59), F (<40) */
  grade: string;
  /** All detected issues */
  issues: DataQualityIssue[];
  /** Summary statistics */
  stats: {
    totalExamples: number;
    totalIntents: number;
    avgExamplesPerIntent: number;
    minExamplesPerIntent: number;
    maxExamplesPerIntent: number;
    duplicateCount: number;
    nearDuplicateCount: number;
    suspectedMislabels: number;
    shortExamples: number;
    vocabularySize: number;
  };
  /** Per-intent breakdown */
  perIntent: Array<{
    intent: string;
    exampleCount: number;
    uniqueTokens: number;
    avgLength: number;
    duplicates: number;
    suspectedMislabels: number;
  }>;
}

/**
 * Compute Jaccard similarity between two sets of words
 */
function jaccardSimilarity(a: string, b: string): number {
  const setA = new Set(a.toLowerCase().split(/\s+/));
  const setB = new Set(b.toLowerCase().split(/\s+/));
  let intersection = 0;
  for (const w of setA) if (setB.has(w)) intersection++;
  const union = setA.size + setB.size - intersection;
  return union > 0 ? intersection / union : 0;
}

/**
 * Check if a text is mostly stop words / low information
 */
function isLowInformation(text: string): boolean {
  const STOP_WORDS = new Set([
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "to", "of", "in", "for",
    "on", "with", "at", "by", "from", "as", "into", "through", "during",
    "i", "me", "my", "we", "our", "you", "your", "he", "she", "they",
    "it", "this", "that", "and", "or", "but", "so", "if", "not", "no",
  ]);

  const words = text.toLowerCase().split(/\s+/).filter((w) => w.length > 1);
  if (words.length === 0) return true;

  const contentWords = words.filter((w) => !STOP_WORDS.has(w));
  return contentWords.length === 0;
}

/**
 * Run comprehensive data quality validation
 *
 * @param data - Training data as array of {text, intent} pairs
 * @param model - Optional trained model for mislabel detection (cross-prediction)
 * @returns Detailed quality report with scores, issues, and suggestions
 */
export function validateDataQuality(
  data: Array<{ text: string; intent: string }>,
  model?: EnsembleModel,
): DataQualityReport {
  const issues: DataQualityIssue[] = [];

  if (data.length === 0) {
    return {
      score: 0,
      grade: "F",
      issues: [{ severity: "error", category: "empty_example", message: "No training data", intents: [], examples: [], suggestion: "Add training examples" }],
      stats: { totalExamples: 0, totalIntents: 0, avgExamplesPerIntent: 0, minExamplesPerIntent: 0, maxExamplesPerIntent: 0, duplicateCount: 0, nearDuplicateCount: 0, suspectedMislabels: 0, shortExamples: 0, vocabularySize: 0 },
      perIntent: [],
    };
  }

  // Group by intent
  const byIntent: Record<string, string[]> = {};
  for (const item of data) {
    if (!byIntent[item.intent]) byIntent[item.intent] = [];
    byIntent[item.intent].push(item.text);
  }

  const intents = Object.keys(byIntent);
  let duplicateCount = 0;
  let nearDuplicateCount = 0;
  let suspectedMislabels = 0;
  let shortExamples = 0;
  const globalVocab = new Set<string>();
  const perIntent: DataQualityReport["perIntent"] = [];

  // === Check 1: Per-intent analysis ===
  for (const [intent, examples] of Object.entries(byIntent)) {
    const intentVocab = new Set<string>();
    let intentDuplicates = 0;
    let intentMislabels = 0;

    // Exact duplicates
    const seen = new Set<string>();
    for (const ex of examples) {
      const normalized = ex.toLowerCase().trim();
      if (seen.has(normalized)) {
        intentDuplicates++;
        duplicateCount++;
        issues.push({
          severity: "warning",
          category: "duplicate",
          message: `Exact duplicate in "${intent}"`,
          intents: [intent],
          examples: [ex],
          suggestion: "Remove duplicate to prevent overfit on this example",
        });
      }
      seen.add(normalized);

      // Build vocabulary
      for (const word of normalized.split(/\s+/)) {
        intentVocab.add(word);
        globalVocab.add(word);
      }
    }

    // Near duplicates (Jaccard > 0.8)
    for (let i = 0; i < examples.length; i++) {
      for (let j = i + 1; j < examples.length; j++) {
        if (jaccardSimilarity(examples[i], examples[j]) > 0.85) {
          nearDuplicateCount++;
          if (nearDuplicateCount <= 20) { // cap issues reported
            issues.push({
              severity: "info",
              category: "near_duplicate",
              message: `Near-duplicate pair in "${intent}"`,
              intents: [intent],
              examples: [examples[i], examples[j]],
              suggestion: "Rephrase one example to increase diversity",
            });
          }
        }
      }
    }

    // Short examples
    for (const ex of examples) {
      if (ex.trim().length === 0) {
        shortExamples++;
        issues.push({
          severity: "error",
          category: "empty_example",
          message: `Empty example in "${intent}"`,
          intents: [intent],
          examples: [ex],
          suggestion: "Remove or replace empty examples",
        });
      } else if (ex.split(/\s+/).length < 2) {
        shortExamples++;
        issues.push({
          severity: "warning",
          category: "short_example",
          message: `Single-word example "${ex}" in "${intent}"`,
          intents: [intent],
          examples: [ex],
          suggestion: "Single-word examples may not generalize well. Consider adding context.",
        });
      }
    }

    // Low information examples
    for (const ex of examples) {
      if (isLowInformation(ex) && ex.trim().length > 0) {
        issues.push({
          severity: "info",
          category: "low_information",
          message: `Low-information example "${ex}" in "${intent}"`,
          intents: [intent],
          examples: [ex],
          suggestion: "This example contains mostly stop words. Add domain-specific vocabulary.",
        });
      }
    }

    // Mislabel detection via model
    if (model) {
      for (const ex of examples) {
        const prediction = predictEnsemble(ex, model);
        if (prediction.intent !== intent && prediction.confidence > 0.8) {
          intentMislabels++;
          suspectedMislabels++;
          issues.push({
            severity: "warning",
            category: "mislabel",
            message: `Suspected mislabel: "${ex}" labeled as "${intent}" but model predicts "${prediction.intent}" (${(prediction.confidence * 100).toFixed(0)}%)`,
            intents: [intent, prediction.intent],
            examples: [ex],
            suggestion: `Review: should this be "${prediction.intent}" instead of "${intent}"?`,
          });
        }
      }
    }

    // Avg length
    const avgLen = examples.reduce((s, e) => s + e.split(/\s+/).length, 0) / examples.length;

    perIntent.push({
      intent,
      exampleCount: examples.length,
      uniqueTokens: intentVocab.size,
      avgLength: Math.round(avgLen * 10) / 10,
      duplicates: intentDuplicates,
      suspectedMislabels: intentMislabels,
    });
  }

  // === Check 2: Class imbalance ===
  const exampleCounts = Object.values(byIntent).map((e) => e.length);
  const minCount = Math.min(...exampleCounts);
  const maxCount = Math.max(...exampleCounts);
  const avgCount = data.length / intents.length;

  if (maxCount / (minCount || 1) > 3) {
    const underrepresented = Object.entries(byIntent)
      .filter(([, e]) => e.length < avgCount * 0.5)
      .map(([i]) => i);

    if (underrepresented.length > 0) {
      issues.push({
        severity: "warning",
        category: "class_imbalance",
        message: `Class imbalance detected: ${underrepresented.length} intent(s) have <50% of average examples`,
        intents: underrepresented,
        examples: [],
        suggestion: `Add more examples for: ${underrepresented.join(", ")}`,
      });
    }
  }

  // Intents with very few examples
  for (const [intent, examples] of Object.entries(byIntent)) {
    if (examples.length < 5) {
      issues.push({
        severity: "error",
        category: "class_imbalance",
        message: `"${intent}" has only ${examples.length} examples (minimum recommended: 5)`,
        intents: [intent],
        examples: [],
        suggestion: `Add at least ${5 - examples.length} more examples for "${intent}"`,
      });
    } else if (examples.length < 10) {
      issues.push({
        severity: "warning",
        category: "class_imbalance",
        message: `"${intent}" has only ${examples.length} examples (recommended: 10+)`,
        intents: [intent],
        examples: [],
        suggestion: `Consider adding more examples for better generalization`,
      });
    }
  }

  // === Check 3: Low vocabulary per intent ===
  for (const pi of perIntent) {
    if (pi.uniqueTokens < 10 && pi.exampleCount >= 5) {
      issues.push({
        severity: "info",
        category: "low_vocabulary",
        message: `"${pi.intent}" has low vocabulary diversity (${pi.uniqueTokens} unique tokens)`,
        intents: [pi.intent],
        examples: [],
        suggestion: "Add examples with varied phrasing to improve generalization",
      });
    }
  }

  // === Check 4: High class overlap ===
  for (let i = 0; i < intents.length; i++) {
    for (let j = i + 1; j < intents.length; j++) {
      const vocabA = new Set(byIntent[intents[i]].flatMap((e) => e.toLowerCase().split(/\s+/)));
      const vocabB = new Set(byIntent[intents[j]].flatMap((e) => e.toLowerCase().split(/\s+/)));
      let overlap = 0;
      for (const w of vocabA) if (vocabB.has(w)) overlap++;
      const overlapRatio = overlap / Math.min(vocabA.size, vocabB.size);

      if (overlapRatio > 0.7) {
        issues.push({
          severity: "info",
          category: "class_overlap",
          message: `High vocabulary overlap (${(overlapRatio * 100).toFixed(0)}%) between "${intents[i]}" and "${intents[j]}"`,
          intents: [intents[i], intents[j]],
          examples: [],
          suggestion: "These intents may be confusable. Ensure examples have clear distinguishing features.",
        });
      }
    }
  }

  // === Compute overall score ===
  let score = 100;
  for (const issue of issues) {
    if (issue.severity === "error") score -= 10;
    else if (issue.severity === "warning") score -= 3;
    else score -= 1;
  }
  score = Math.max(0, Math.min(100, score));

  const grade = score >= 90 ? "A" : score >= 75 ? "B" : score >= 60 ? "C" : score >= 40 ? "D" : "F";

  return {
    score,
    grade,
    issues,
    stats: {
      totalExamples: data.length,
      totalIntents: intents.length,
      avgExamplesPerIntent: Math.round(avgCount * 10) / 10,
      minExamplesPerIntent: minCount,
      maxExamplesPerIntent: maxCount,
      duplicateCount,
      nearDuplicateCount,
      suspectedMislabels,
      shortExamples,
      vocabularySize: globalVocab.size,
    },
    perIntent,
  };
}
