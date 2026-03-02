const STOP_WORDS = new Set([
  "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
  "have", "has", "had", "do", "does", "did", "will", "would", "could",
  "should", "may", "might", "shall", "can", "need", "dare", "ought",
  "used", "to", "of", "in", "for", "on", "with", "at", "by", "from",
  "as", "into", "through", "during", "before", "after", "above", "below",
  "between", "out", "off", "over", "under", "again", "further", "then",
  "once", "here", "there", "when", "where", "why", "how", "all", "both",
  "each", "few", "more", "most", "other", "some", "such", "no", "nor",
  "not", "only", "own", "same", "so", "than", "too", "very", "just",
  "because", "but", "and", "or", "if", "while", "that", "this", "it",
  "its", "i", "me", "my", "we", "our", "you", "your", "he", "him",
  "his", "she", "her", "they", "them", "their", "what", "which", "who",
  "am", "about", "up", "down",
]);

/**
 * Light stemmer — only safe inflectional rules.
 * Removes: -ing, -ed, -s/-es, -ies→y
 * Skips dangerous rules that break root words (-er, -tion→te, etc.)
 */
function stem(word: string): string {
  if (word.length <= 4) return word;
  let w = word;
  if (w.endsWith("ing") && w.length > 6) w = w.slice(0, -3);
  else if (w.endsWith("ies") && w.length > 5) w = w.slice(0, -3) + "y";
  else if (w.endsWith("ed") && w.length > 5) w = w.slice(0, -2);
  else if (w.endsWith("es") && w.length > 5) w = w.slice(0, -2);
  else if (w.endsWith("s") && !w.endsWith("ss") && !w.endsWith("us") && !w.endsWith("is") && w.length > 4)
    w = w.slice(0, -1);
  // Reduce doubled trailing consonant: "shipp"→"ship", "cancell"→"cancel"
  if (w.length > 3 && w[w.length - 1] === w[w.length - 2] && /[bcdfghjklmnpqrstvwxyz]/.test(w[w.length - 1]))
    w = w.slice(0, -1);
  return w;
}

/** Tokenize text into cleaned, lightly stemmed tokens */
export function tokenize(text: string): string[] {
  return text
    .toLowerCase()
    .replace(/[^a-z0-9\s'-]/g, " ")
    .split(/\s+/)
    .filter((t) => t.length > 1 && !STOP_WORDS.has(t))
    .map(stem);
}

/** Generate n-grams from tokens */
export function ngrams(tokens: string[], n: number): string[] {
  const result: string[] = [];
  for (let i = 0; i <= tokens.length - n; i++) {
    result.push(tokens.slice(i, i + n).join("_"));
  }
  return result;
}

// Intent-signal patterns: aggregate keyword groups into strong meta-features.
// Each signal fires when ANY word in the group is present, giving concentrated
// discriminative power vs individual sparse unigrams.
const SIGNAL_PATTERNS: Array<{ signal: string; pattern: RegExp }> = [
  { signal: "__track", pattern: /\b(track|ship|deliver|arriv|dispatch|transit)\b/ },
  { signal: "__return", pattern: /\b(return|send back|exchange|swap)\b/ },
  { signal: "__refund", pattern: /\b(refund|money back|reimburse|credit back)\b/ },
  { signal: "__cancel", pattern: /\b(cancel|void|revoke|abort|withdrawal)\b/ },
  { signal: "__defect", pattern: /\b(defective|broken|damaged|terrible|awful|poor|worst|faulty|disappoint|unhappy|dissatisf|furious|upset|scam|fake|stale|crush)\b/ },
  { signal: "__pay", pattern: /\b(payment|billing|charged|card|declined|transaction|checkout|invoice|coupon|promo)\b/ },
  { signal: "__acct", pattern: /\b(account|password|login|log in|email|profile|username|sign up|verification|two factor|deactivat)\b/ },
  { signal: "__human", pattern: /\b(human|agent|person|manager|operator|representative|staff|supervisor|bot|automat|live chat|live support)\b/ },
  { signal: "__greet", pattern: /\b(hello|hi|hey|howdy|greetings|good morning|good afternoon|good evening|hiya|sup)\b/ },
  { signal: "__bye", pattern: /\b(bye|goodbye|farewell|leaving|see ya|take care|night|peace out|adios|toodle)\b/ },
  { signal: "__thank", pattern: /\b(thank|thanks|appreciate|grateful|helpful|splendid|brilliant)\b/ },
  { signal: "__product", pattern: /\b(stock|available|price|cost|size|color|specification|warranty|feature|dimension|resolution|compatible|bluetooth|waterproof)\b/ },
];

/** Tokenize with unigrams + bigrams + intent-signal features */
export function tokenizeWithNgrams(text: string): string[] {
  const unigrams = tokenize(text);
  const bigrams = ngrams(unigrams, 2);

  // Add intent-signal meta-features from raw text
  const lower = text.toLowerCase();
  const signals: string[] = [];
  for (const { signal, pattern } of SIGNAL_PATTERNS) {
    if (pattern.test(lower)) {
      signals.push(signal);
    }
  }

  return [...unigrams, ...bigrams, ...signals];
}

/** Tokenize without stemming (for entity extraction) */
export function tokenizeRaw(text: string): string[] {
  return text
    .toLowerCase()
    .replace(/[^a-z0-9\s'-]/g, " ")
    .split(/\s+/)
    .filter((t) => t.length > 0);
}

/** Build vocabulary from a collection of tokenized documents */
export function buildVocabulary(documents: string[][]): string[] {
  const vocab = new Set<string>();
  for (const doc of documents) {
    for (const token of doc) {
      vocab.add(token);
    }
  }
  return Array.from(vocab).sort();
}

/** Compute TF (term frequency) vector for a document */
export function computeTF(tokens: string[], vocabulary: string[]): number[] {
  const freq: Record<string, number> = {};
  for (const token of tokens) {
    freq[token] = (freq[token] || 0) + 1;
  }
  const maxFreq = Math.max(...Object.values(freq), 1);
  return vocabulary.map((word) => (freq[word] || 0) / maxFreq);
}

/** Compute IDF (inverse document frequency) weights */
export function computeIDF(
  documents: string[][],
  vocabulary: string[]
): Record<string, number> {
  const N = documents.length;
  const idf: Record<string, number> = {};
  for (const word of vocabulary) {
    const docsWithWord = documents.filter((doc) => doc.includes(word)).length;
    idf[word] = Math.log((N + 1) / (docsWithWord + 1)) + 1;
  }
  return idf;
}

/** Compute TF-IDF vector */
export function computeTFIDF(
  tokens: string[],
  vocabulary: string[],
  idf: Record<string, number>
): number[] {
  const tf = computeTF(tokens, vocabulary);
  return vocabulary.map((word, i) => tf[i] * (idf[word] || 1));
}

/** Cosine similarity between two vectors */
export function cosineSimilarity(a: number[], b: number[]): number {
  let dot = 0;
  let normA = 0;
  let normB = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }
  const denom = Math.sqrt(normA) * Math.sqrt(normB);
  return denom === 0 ? 0 : dot / denom;
}
