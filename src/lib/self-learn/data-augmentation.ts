/**
 * Data Augmentation Engine for NLU
 *
 * Multiple augmentation strategies:
 * 1. EDA (Easy Data Augmentation) - Wei & Zou 2019
 *    - Synonym replacement
 *    - Random insertion
 *    - Random swap
 *    - Random deletion
 * 2. Template-based generation
 * 3. Compositional augmentation (entity slot filling)
 * 4. Paraphrase patterns (rule-based)
 * 5. Noise injection (typo simulation)
 */

// Lightweight synonym dictionary for customer support domain
const SYNONYMS: Record<string, string[]> = {
  order: ["purchase", "order", "buying", "transaction"],
  return: ["send back", "return", "bring back", "give back"],
  refund: ["money back", "refund", "reimbursement", "credit"],
  cancel: ["cancel", "void", "revoke", "abort"],
  track: ["track", "trace", "follow", "monitor", "locate"],
  deliver: ["deliver", "ship", "send", "dispatch"],
  broken: ["broken", "damaged", "defective", "faulty", "malfunctioning"],
  terrible: ["terrible", "awful", "horrible", "dreadful", "appalling"],
  help: ["help", "assist", "support", "aid"],
  account: ["account", "profile", "login", "credentials"],
  password: ["password", "passcode", "pin", "credentials"],
  payment: ["payment", "charge", "billing", "transaction"],
  product: ["product", "item", "article", "goods", "merchandise"],
  want: ["want", "need", "wish", "desire", "like"],
  change: ["change", "modify", "update", "alter", "edit"],
  receive: ["receive", "get", "obtain", "got"],
  problem: ["problem", "issue", "trouble", "difficulty"],
  fast: ["fast", "quick", "rapid", "swift", "speedy"],
  good: ["good", "great", "excellent", "amazing", "wonderful"],
  bad: ["bad", "poor", "terrible", "awful", "horrible"],
  speak: ["speak", "talk", "chat", "communicate"],
  person: ["person", "human", "agent", "representative", "someone"],
  hello: ["hello", "hi", "hey", "greetings", "howdy"],
  goodbye: ["goodbye", "bye", "farewell", "see you", "take care"],
  thanks: ["thanks", "thank you", "appreciate", "grateful"],
  where: ["where", "what location", "which place"],
  when: ["when", "what time", "how soon", "how long"],
  price: ["price", "cost", "amount", "fee", "charge"],
  available: ["available", "in stock", "on hand", "accessible"],
  size: ["size", "dimension", "measurement", "fit"],
  color: ["color", "shade", "hue", "tint"],
};

// Paraphrase templates for common NLU patterns
const PARAPHRASE_PATTERNS: Array<{ from: RegExp; to: string[] }> = [
  { from: /^i want to (.+)/, to: ["i'd like to $1", "can i $1", "i need to $1", "please $1", "help me $1"] },
  { from: /^how do i (.+)/, to: ["how can i $1", "what's the way to $1", "i need help with $1", "help me $1"] },
  { from: /^can i (.+)/, to: ["is it possible to $1", "am i able to $1", "i want to $1"] },
  { from: /^where is (.+)/, to: ["what happened to $1", "i'm looking for $1", "can you locate $1"] },
  { from: /^what is (.+)/, to: ["tell me about $1", "i need info on $1", "explain $1"] },
  { from: /^i need (.+)/, to: ["i require $1", "i want $1", "give me $1", "please provide $1"] },
  { from: /^please (.+)/, to: ["can you $1", "i need you to $1", "kindly $1", "$1 please"] },
];

// Template slots for domain-specific generation
const TEMPLATES: Record<string, string[]> = {
  order_status: [
    "where is my {product} order",
    "when will my {product} arrive",
    "track my {product} delivery",
    "has my {product} shipped yet",
    "what's the status of my {product}",
    "i ordered a {product} {time_ago} where is it",
    "my {product} order is {status}",
    "i haven't received my {product} yet",
    "estimated delivery for my {product}",
    "any update on my {product} order",
  ],
  return_product: [
    "i want to return my {product}",
    "how do i return the {product}",
    "the {product} doesn't fit can i return it",
    "return policy for {product}",
    "i need a return label for my {product}",
    "can i exchange my {product} for a different {attribute}",
    "the {product} is the wrong {attribute} i want to return it",
    "send back my {product} order",
  ],
  complaint: [
    "the {product} i received is {defect}",
    "my {product} arrived {defect}",
    "terrible quality {product}",
    "the {product} stopped working after {time_period}",
    "i'm very disappointed with the {product}",
    "the {product} looks nothing like the picture",
    "this {product} is completely {defect}",
  ],
  product_inquiry: [
    "do you have {product} in {color}",
    "what {attribute} does the {product} come in",
    "is the {product} {feature}",
    "tell me about the {product} specifications",
    "how much does the {product} cost",
    "is the {product} available in {size}",
    "does the {product} have {feature}",
  ],
};

const SLOT_FILLERS: Record<string, string[]> = {
  product: ["laptop", "phone", "tablet", "headphones", "charger", "keyboard", "mouse", "camera", "watch", "speaker", "monitor"],
  color: ["red", "blue", "black", "white", "green", "silver", "gold", "pink", "grey"],
  size: ["small", "medium", "large", "extra large", "XS", "XL", "XXL"],
  time_ago: ["yesterday", "last week", "3 days ago", "a week ago", "two weeks ago"],
  time_period: ["one day", "a week", "two days", "three hours", "a month"],
  status: ["still processing", "delayed", "stuck in transit", "not delivered"],
  defect: ["broken", "damaged", "scratched", "not working", "defective", "cracked", "faulty"],
  attribute: ["color", "size", "model", "version"],
  feature: ["waterproof", "wireless", "bluetooth", "rechargeable", "compatible with my phone"],
};

/**
 * Synonym replacement (SR) - Replace n random non-stop words with synonyms
 */
export function synonymReplacement(text: string, n: number = 1): string {
  const words = text.split(/\s+/);
  const result = [...words];

  for (let attempt = 0; attempt < n * 3 && n > 0; attempt++) {
    const idx = Math.floor(Math.random() * words.length);
    const word = words[idx].toLowerCase().replace(/[^a-z]/g, "");
    const syns = SYNONYMS[word];
    if (syns && syns.length > 1) {
      const alternatives = syns.filter((s) => s !== word);
      if (alternatives.length > 0) {
        result[idx] = alternatives[Math.floor(Math.random() * alternatives.length)];
        n--;
      }
    }
  }

  return result.join(" ");
}

/**
 * Random insertion (RI) - Insert n random synonyms of random words
 */
export function randomInsertion(text: string, n: number = 1): string {
  const words = text.split(/\s+/);
  const result = [...words];

  for (let i = 0; i < n; i++) {
    const idx = Math.floor(Math.random() * words.length);
    const word = words[idx].toLowerCase().replace(/[^a-z]/g, "");
    const syns = SYNONYMS[word];
    if (syns) {
      const syn = syns[Math.floor(Math.random() * syns.length)];
      const insertPos = Math.floor(Math.random() * (result.length + 1));
      result.splice(insertPos, 0, syn);
    }
  }

  return result.join(" ");
}

/**
 * Random swap (RS) - Swap n random pairs of words
 */
export function randomSwap(text: string, n: number = 1): string {
  const words = text.split(/\s+/);
  if (words.length < 2) return text;
  const result = [...words];

  for (let i = 0; i < n; i++) {
    const a = Math.floor(Math.random() * result.length);
    let b = Math.floor(Math.random() * result.length);
    while (b === a) b = Math.floor(Math.random() * result.length);
    [result[a], result[b]] = [result[b], result[a]];
  }

  return result.join(" ");
}

/**
 * Random deletion (RD) - Delete words with probability p
 */
export function randomDeletion(text: string, p: number = 0.1): string {
  const words = text.split(/\s+/);
  if (words.length <= 2) return text;
  const result = words.filter(() => Math.random() > p);
  return result.length > 0 ? result.join(" ") : words[0];
}

/**
 * Paraphrase via pattern matching
 */
export function paraphrase(text: string): string | null {
  const lower = text.toLowerCase();
  for (const { from, to } of PARAPHRASE_PATTERNS) {
    const match = lower.match(from);
    if (match) {
      const template = to[Math.floor(Math.random() * to.length)];
      return template.replace("$1", match[1]);
    }
  }
  return null;
}

/**
 * Generate from templates with slot filling
 */
export function generateFromTemplate(intentName: string): string | null {
  const templates = TEMPLATES[intentName];
  if (!templates) return null;

  const template = templates[Math.floor(Math.random() * templates.length)];
  return template.replace(/\{(\w+)\}/g, (_, slot) => {
    const fillers = SLOT_FILLERS[slot];
    if (!fillers) return slot;
    return fillers[Math.floor(Math.random() * fillers.length)];
  });
}

/**
 * Typo injection for robustness training
 * Simulates common keyboard errors
 */
const KEYBOARD_ADJACENT: Record<string, string> = {
  q: "wa", w: "qeas", e: "wrds", r: "etf", t: "ryg", y: "tuh", u: "yij", i: "uok", o: "iplk", p: "ol",
  a: "qws", s: "awde", d: "serf", f: "drtg", g: "ftyh", h: "guyj", j: "huik", k: "jiol", l: "kop",
  z: "asx", x: "zsdc", c: "xdfv", v: "cfgb", b: "vghn", n: "bhjm", m: "njk",
};

export function injectTypo(text: string): string {
  const words = text.split(/\s+/);
  if (words.length === 0) return text;

  const wordIdx = Math.floor(Math.random() * words.length);
  const word = words[wordIdx];
  if (word.length < 3) return text;

  const charIdx = Math.floor(Math.random() * word.length);
  const char = word[charIdx].toLowerCase();
  const typoType = Math.random();

  if (typoType < 0.4 && KEYBOARD_ADJACENT[char]) {
    // Adjacent key press
    const adjacent = KEYBOARD_ADJACENT[char];
    const replacement = adjacent[Math.floor(Math.random() * adjacent.length)];
    words[wordIdx] = word.slice(0, charIdx) + replacement + word.slice(charIdx + 1);
  } else if (typoType < 0.6 && word.length > 3) {
    // Character swap
    const swapIdx = Math.min(charIdx + 1, word.length - 1);
    const chars = word.split("");
    [chars[charIdx], chars[swapIdx]] = [chars[swapIdx], chars[charIdx]];
    words[wordIdx] = chars.join("");
  } else if (typoType < 0.8) {
    // Character deletion
    words[wordIdx] = word.slice(0, charIdx) + word.slice(charIdx + 1);
  } else {
    // Character duplication
    words[wordIdx] = word.slice(0, charIdx) + word[charIdx] + word.slice(charIdx);
  }

  return words.join(" ");
}

/**
 * Apply all augmentation strategies to a single example
 * Returns multiple augmented versions
 */
export function augmentExample(
  text: string,
  intentName: string,
  count: number = 4,
): string[] {
  const augmented: string[] = [];
  const strategies = [
    () => synonymReplacement(text, 1 + Math.floor(Math.random() * 2)),
    () => randomInsertion(text, 1),
    () => randomSwap(text, 1),
    () => randomDeletion(text, 0.15),
    () => paraphrase(text),
    () => generateFromTemplate(intentName),
    () => injectTypo(text),
  ];

  for (let i = 0; i < count; i++) {
    const strategy = strategies[Math.floor(Math.random() * strategies.length)];
    const result = strategy();
    if (result && result !== text && result.trim().length > 0) {
      augmented.push(result);
    }
  }

  return [...new Set(augmented)]; // deduplicate
}

/**
 * Batch augment entire training dataset
 * Returns new examples only (doesn't include originals)
 */
export function augmentDataset(
  data: Array<{ text: string; intent: string }>,
  augmentationsPerExample: number = 3,
): Array<{ text: string; intent: string }> {
  const newExamples: Array<{ text: string; intent: string }> = [];
  const existingTexts = new Set(data.map((d) => d.text.toLowerCase()));

  for (const item of data) {
    const augmented = augmentExample(item.text, item.intent, augmentationsPerExample);
    for (const text of augmented) {
      if (!existingTexts.has(text.toLowerCase())) {
        newExamples.push({ text, intent: item.intent });
        existingTexts.add(text.toLowerCase());
      }
    }
  }

  return newExamples;
}
