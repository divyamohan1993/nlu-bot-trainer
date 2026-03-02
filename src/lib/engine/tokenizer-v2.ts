/**
 * Advanced Tokenizer v2 - Multi-Strategy Feature Extraction
 *
 * Combines multiple feature extraction strategies for robust NLU:
 * 1. Word unigrams + bigrams (semantic features)
 * 2. Character n-grams (morphological features, typo resistance)
 * 3. Intent signal patterns (domain-specific boosting)
 * 4. Subword features (BPE-lite for OOV handling)
 * 5. Positional features (word position encoding)
 * 6. Syntactic features (question words, negation, sentence structure)
 *
 * Designed for: O(n) processing where n = text length
 * Memory: O(k) where k = number of unique features extracted
 */

// Extended stop words - tuned for customer support NLU
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
  "am", "about", "up", "down", "please", "also", "really", "right",
  "well", "like", "know", "think", "want", "get", "got", "go", "going",
  "make", "let", "way", "thing", "things", "still", "something",
]);

// Negation words - critical for intent discrimination
const NEGATION_WORDS = new Set([
  "not", "no", "never", "neither", "nor", "nobody", "nothing", "nowhere",
  "don't", "dont", "doesn't", "doesnt", "didn't", "didnt", "won't", "wont",
  "wouldn't", "wouldnt", "can't", "cant", "cannot", "couldn't", "couldnt",
  "shouldn't", "shouldnt", "isn't", "isnt", "aren't", "arent", "wasn't",
  "wasnt", "weren't", "werent", "haven't", "havent", "hasn't", "hasnt",
  "hadn't", "hadnt",
]);

// Question indicators
const QUESTION_WORDS = new Set([
  "what", "when", "where", "which", "who", "whom", "whose", "why", "how",
  "is", "are", "was", "were", "do", "does", "did", "can", "could", "will",
  "would", "should", "may", "might",
]);

// Intent signal patterns (domain-specific meta-features)
const SIGNAL_PATTERNS: Array<{ signal: string; pattern: RegExp }> = [
  { signal: "__sig_track", pattern: /\b(track|ship|deliver|arriv|dispatch|transit|courier|fedex|ups|dhl|post office)\b/ },
  { signal: "__sig_return", pattern: /\b(return|send back|exchange|swap|replace|replacement)\b/ },
  { signal: "__sig_refund", pattern: /\b(refund|money back|reimburse|credit back|chargeback)\b/ },
  { signal: "__sig_cancel", pattern: /\b(cancel|void|revoke|abort|withdrawal|undo|stop order)\b/ },
  { signal: "__sig_defect", pattern: /\b(defective|broken|damaged|terrible|awful|poor|worst|faulty|disappoint|unhappy|dissatisf|furious|upset|scam|fake|stale|crush|crack|scratch|leak|malfunction)\b/ },
  { signal: "__sig_pay", pattern: /\b(payment|billing|charged|card|declined|transaction|checkout|invoice|coupon|promo|receipt|price|cost|fee|tax)\b/ },
  { signal: "__sig_acct", pattern: /\b(account|password|login|log in|email|profile|username|sign up|verification|two factor|deactivat|register|auth)\b/ },
  { signal: "__sig_human", pattern: /\b(human|agent|person|manager|operator|representative|staff|supervisor|bot|automat|live chat|live support|speak|talk|call)\b/ },
  { signal: "__sig_greet", pattern: /\b(hello|hi|hey|howdy|greetings|good morning|good afternoon|good evening|hiya|sup|yo)\b/ },
  { signal: "__sig_bye", pattern: /\b(bye|goodbye|farewell|leaving|see ya|take care|night|peace out|adios|toodle|cya|ttyl)\b/ },
  { signal: "__sig_thank", pattern: /\b(thank|thanks|appreciate|grateful|helpful|splendid|brilliant|great help)\b/ },
  { signal: "__sig_product", pattern: /\b(stock|available|price|cost|size|color|specification|warranty|feature|dimension|resolution|compatible|bluetooth|waterproof|weight|material|brand)\b/ },
  { signal: "__sig_urgent", pattern: /\b(urgent|asap|immediately|right now|emergency|critical|hurry)\b/ },
  { signal: "__sig_negative", pattern: /\b(terrible|horrible|worst|awful|disgusting|pathetic|useless|ridiculous|unacceptable|outrageous|infuriating)\b/ },
];

// Contraction expansion for normalization
const CONTRACTIONS: Record<string, string> = {
  "i'm": "i am", "i've": "i have", "i'll": "i will", "i'd": "i would",
  "you're": "you are", "you've": "you have", "you'll": "you will", "you'd": "you would",
  "he's": "he is", "she's": "she is", "it's": "it is",
  "we're": "we are", "we've": "we have", "we'll": "we will",
  "they're": "they are", "they've": "they have", "they'll": "they will",
  "that's": "that is", "who's": "who is", "what's": "what is",
  "where's": "where is", "there's": "there is", "here's": "here is",
  "don't": "do not", "doesn't": "does not", "didn't": "did not",
  "won't": "will not", "wouldn't": "would not", "couldn't": "could not",
  "shouldn't": "should not", "can't": "cannot", "cannot": "can not",
  "isn't": "is not", "aren't": "are not", "wasn't": "was not",
  "weren't": "were not", "haven't": "have not", "hasn't": "has not",
  "hadn't": "had not", "ain't": "am not",
};

/**
 * Light stemmer - safe inflectional rules only
 * Aggressive enough for intent classification, conservative enough not to break words
 */
function stem(word: string): string {
  if (word.length <= 4) return word;
  let w = word;
  if (w.endsWith("ying")) return w; // keep "buying", "trying"
  if (w.endsWith("ing") && w.length > 6) w = w.slice(0, -3);
  else if (w.endsWith("ment") && w.length > 7) w = w.slice(0, -4);
  else if (w.endsWith("ness") && w.length > 7) w = w.slice(0, -4);
  else if (w.endsWith("tion") && w.length > 6) w = w.slice(0, -4) + "t";
  else if (w.endsWith("ies") && w.length > 5) w = w.slice(0, -3) + "y";
  else if (w.endsWith("ful") && w.length > 6) w = w.slice(0, -3);
  else if (w.endsWith("ally") && w.length > 6) w = w.slice(0, -4);
  else if (w.endsWith("ly") && w.length > 5) w = w.slice(0, -2);
  else if (w.endsWith("ed") && w.length > 5) w = w.slice(0, -2);
  else if (w.endsWith("es") && w.length > 5) w = w.slice(0, -2);
  else if (w.endsWith("s") && !w.endsWith("ss") && !w.endsWith("us") && !w.endsWith("is") && w.length > 4)
    w = w.slice(0, -1);
  // Reduce doubled trailing consonant
  if (w.length > 3 && w[w.length - 1] === w[w.length - 2] && /[bcdfghjklmnpqrstvwxyz]/.test(w[w.length - 1]))
    w = w.slice(0, -1);
  return w;
}

/**
 * Normalize text: lowercase, expand contractions, clean punctuation
 */
export function normalizeText(text: string): string {
  let normalized = text.toLowerCase().trim();
  // Expand contractions
  for (const [contraction, expansion] of Object.entries(CONTRACTIONS)) {
    normalized = normalized.replace(new RegExp(`\\b${contraction.replace("'", "'")}\\b`, "g"), expansion);
  }
  // Normalize unicode quotes and dashes
  normalized = normalized
    .replace(/[\u2018\u2019]/g, "'")
    .replace(/[\u201C\u201D]/g, '"')
    .replace(/[\u2013\u2014]/g, "-");
  return normalized;
}

/**
 * Extract word tokens with stemming
 */
function extractWords(text: string): string[] {
  return text
    .replace(/[^a-z0-9\s'-]/g, " ")
    .split(/\s+/)
    .filter((t) => t.length > 1 && !STOP_WORDS.has(t))
    .map(stem);
}

/**
 * Extract character n-grams from text (3-grams and 4-grams)
 * Provides typo resilience and morphological features
 */
function extractCharNgrams(text: string): string[] {
  const clean = text.replace(/\s+/g, " ").trim();
  const features: string[] = [];
  // Add word boundary markers
  const padded = `^${clean}$`;
  for (let n = 3; n <= 4; n++) {
    for (let i = 0; i <= padded.length - n; i++) {
      features.push(`_c${n}_${padded.slice(i, i + n)}`);
    }
  }
  return features;
}

/**
 * Extract word bigrams
 */
function extractBigrams(words: string[]): string[] {
  const bigrams: string[] = [];
  for (let i = 0; i < words.length - 1; i++) {
    bigrams.push(`${words[i]}_${words[i + 1]}`);
  }
  return bigrams;
}

/**
 * Extract syntactic features
 */
function extractSyntacticFeatures(text: string): string[] {
  const features: string[] = [];
  const words = text.split(/\s+/);
  const firstWord = words[0]?.toLowerCase() || "";

  // Question detection
  if (text.includes("?") || QUESTION_WORDS.has(firstWord)) {
    features.push("__syn_question");
  }

  // Negation detection with scope
  let hasNegation = false;
  for (const word of words) {
    if (NEGATION_WORDS.has(word.toLowerCase())) {
      hasNegation = true;
      features.push("__syn_negation");
      break;
    }
  }

  // Exclamation / emphasis
  if (text.includes("!") || text === text.toUpperCase() && text.length > 3) {
    features.push("__syn_emphasis");
  }

  // Sentence length buckets (short queries vs long descriptions)
  if (words.length <= 3) features.push("__syn_short");
  else if (words.length <= 7) features.push("__syn_medium");
  else features.push("__syn_long");

  // Negation + key concept combinations (extremely discriminative)
  if (hasNegation) {
    const concepts = ["work", "help", "receiv", "arriv", "ship", "want", "need"];
    for (const concept of concepts) {
      if (text.toLowerCase().includes(concept)) {
        features.push(`__syn_neg_${concept}`);
      }
    }
  }

  return features;
}

/**
 * Extract intent signal features from raw text
 */
function extractSignalFeatures(text: string): string[] {
  const lower = text.toLowerCase();
  const signals: string[] = [];
  for (const { signal, pattern } of SIGNAL_PATTERNS) {
    if (pattern.test(lower)) {
      signals.push(signal);
    }
  }
  return signals;
}

/**
 * Extract positional features - first and last word indicators
 */
function extractPositionalFeatures(words: string[]): string[] {
  const features: string[] = [];
  if (words.length > 0) {
    features.push(`__pos_first_${words[0]}`);
    features.push(`__pos_last_${words[words.length - 1]}`);
  }
  return features;
}

/**
 * Subword features - split long words into common subword units
 * Lightweight BPE-like approach using common English affixes
 */
const COMMON_PREFIXES = ["un", "re", "pre", "dis", "mis", "over", "out", "sub"];
const COMMON_SUFFIXES = ["ing", "tion", "ment", "ness", "able", "ible", "ful", "less", "ous", "ive", "ize", "ate"];

function extractSubwordFeatures(words: string[]): string[] {
  const features: string[] = [];
  for (const word of words) {
    if (word.length < 6) continue;
    for (const prefix of COMMON_PREFIXES) {
      if (word.startsWith(prefix) && word.length > prefix.length + 2) {
        features.push(`__sub_pfx_${prefix}`);
        features.push(`__sub_root_${word.slice(prefix.length)}`);
        break;
      }
    }
    for (const suffix of COMMON_SUFFIXES) {
      if (word.endsWith(suffix) && word.length > suffix.length + 2) {
        features.push(`__sub_sfx_${suffix}`);
        break;
      }
    }
  }
  return features;
}

export interface TokenizerConfig {
  useCharNgrams: boolean;
  useBigrams: boolean;
  useSyntactic: boolean;
  useSignals: boolean;
  usePositional: boolean;
  useSubword: boolean;
  maxCharNgramLen?: number;
}

export const DEFAULT_TOKENIZER_CONFIG: TokenizerConfig = {
  useCharNgrams: true,
  useBigrams: true,
  useSyntactic: true,
  useSignals: true,
  usePositional: true,
  useSubword: true,
};

export const FAST_TOKENIZER_CONFIG: TokenizerConfig = {
  useCharNgrams: false,
  useBigrams: true,
  useSyntactic: true,
  useSignals: true,
  usePositional: false,
  useSubword: false,
};

/**
 * Full-featured tokenizer - extracts all feature types
 * Returns array of feature strings ready for hashing
 */
export function tokenizeV2(
  text: string,
  config: TokenizerConfig = DEFAULT_TOKENIZER_CONFIG,
): string[] {
  const normalized = normalizeText(text);
  const words = extractWords(normalized);
  const features: string[] = [...words];

  if (config.useBigrams) {
    features.push(...extractBigrams(words));
  }

  if (config.useCharNgrams) {
    features.push(...extractCharNgrams(normalized));
  }

  if (config.useSyntactic) {
    features.push(...extractSyntacticFeatures(normalized));
  }

  if (config.useSignals) {
    features.push(...extractSignalFeatures(text));
  }

  if (config.usePositional) {
    features.push(...extractPositionalFeatures(words));
  }

  if (config.useSubword) {
    features.push(...extractSubwordFeatures(words));
  }

  return features;
}

/**
 * Raw tokenization without stemming (for entity extraction)
 */
export function tokenizeRawV2(text: string): string[] {
  return normalizeText(text)
    .replace(/[^a-z0-9\s'-]/g, " ")
    .split(/\s+/)
    .filter((t) => t.length > 0);
}
