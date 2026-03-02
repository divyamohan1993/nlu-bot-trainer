import { NextRequest, NextResponse } from "next/server";

const GEMINI_ENDPOINT =
  "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent";

const GENERATION_PROMPT = `You are an expert NLU training data generator. Generate diverse, realistic training examples for intent classification in a customer service chatbot.

## Intent: {INTENT_NAME}
## Description: {INTENT_DESCRIPTION}

## Existing examples (for style reference):
{EXISTING_EXAMPLES}

## Instructions:
1. Generate {COUNT} NEW unique training examples for the "{INTENT_NAME}" intent
2. Vary the examples by:
   - Different phrasings and sentence structures
   - Formal and informal language
   - Short and long utterances (3-20 words)
   - Questions, statements, and commands
   - Include typos/casual language occasionally (10% of examples)
   - Different levels of politeness
   - Include context-rich examples ("I ordered 3 days ago and...")
3. Do NOT repeat or closely paraphrase the existing examples
4. Each example should clearly belong to the "{INTENT_NAME}" intent

## Output format (JSON array of strings, no other text):
["example 1", "example 2", ...]`;

export async function POST(request: NextRequest) {
  const apiKey = process.env.GOOGLE_API_KEY;
  if (!apiKey) {
    return NextResponse.json(
      { error: "GOOGLE_API_KEY not configured. Add it to .env.local" },
      { status: 500 }
    );
  }

  let body: {
    intentName: string;
    intentDescription: string;
    existingExamples: string[];
    count: number;
  };

  try {
    body = await request.json();
  } catch {
    return NextResponse.json({ error: "Invalid JSON body" }, { status: 400 });
  }

  const { intentName, intentDescription, existingExamples, count } = body;

  if (!intentName || !count || count < 1 || count > 50) {
    return NextResponse.json(
      { error: "intentName is required and count must be between 1 and 50" },
      { status: 400 }
    );
  }

  const sampleExamples = existingExamples
    .slice(0, 8)
    .map((e) => `  - ${e}`)
    .join("\n");

  const prompt = GENERATION_PROMPT.replace(/\{INTENT_NAME\}/g, intentName)
    .replace("{INTENT_DESCRIPTION}", intentDescription || intentName)
    .replace("{EXISTING_EXAMPLES}", sampleExamples || "  (none)")
    .replace("{COUNT}", String(Math.min(count, 50)));

  try {
    const res = await fetch(`${GEMINI_ENDPOINT}?key=${apiKey}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        contents: [{ parts: [{ text: prompt }] }],
        generationConfig: {
          temperature: 0.8,
          topP: 0.95,
          maxOutputTokens: 4096,
        },
      }),
    });

    if (!res.ok) {
      const errBody = await res.text();
      return NextResponse.json(
        { error: `Gemini API error (${res.status}): ${errBody.slice(0, 300)}` },
        { status: 502 }
      );
    }

    const data = await res.json();
    const rawText =
      data?.candidates?.[0]?.content?.parts?.[0]?.text ?? "";

    const examples = parseJsonArray(rawText);
    if (!examples) {
      return NextResponse.json(
        { error: "Failed to parse Gemini response", raw: rawText.slice(0, 500) },
        { status: 502 }
      );
    }

    // Deduplicate against existing
    const existingLower = new Set(
      existingExamples.map((e) => e.toLowerCase().trim())
    );
    const unique = examples.filter(
      (e: string) => !existingLower.has(e.toLowerCase().trim())
    );

    return NextResponse.json({ examples: unique });
  } catch (err) {
    return NextResponse.json(
      { error: `Network error: ${err instanceof Error ? err.message : String(err)}` },
      { status: 502 }
    );
  }
}

function parseJsonArray(text: string): string[] | null {
  let cleaned = text.trim();

  // Strip markdown code fences
  if (cleaned.startsWith("```")) {
    const lines = cleaned.split("\n");
    cleaned = lines
      .slice(1, lines[lines.length - 1].trim() === "```" ? -1 : undefined)
      .join("\n");
  }

  try {
    const parsed = JSON.parse(cleaned);
    if (Array.isArray(parsed)) return parsed.map(String);
  } catch {
    // Try extracting array from surrounding text
    const start = cleaned.indexOf("[");
    const end = cleaned.lastIndexOf("]");
    if (start !== -1 && end > start) {
      try {
        const parsed = JSON.parse(cleaned.slice(start, end + 1));
        if (Array.isArray(parsed)) return parsed.map(String);
      } catch {
        return null;
      }
    }
  }
  return null;
}
