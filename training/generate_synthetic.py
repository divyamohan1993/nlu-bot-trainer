#!/usr/bin/env python3
"""
Synthetic Training Data Generator using Google Vertex AI (Gemini)
=================================================================
Generates synthetic NLU training examples using Gemini models via
Vertex AI or the Gemini Developer API. Supports both batch and
online inference for cost optimization.

Usage:
  # Using Vertex AI (requires gcloud auth)
  python generate_synthetic.py --input training_data.json --output augmented.json --provider vertex

  # Using Gemini Developer API (requires GOOGLE_API_KEY)
  python generate_synthetic.py --input training_data.json --output augmented.json --provider gemini

  # Generate from scratch (no input, provide intent descriptions)
  python generate_synthetic.py --intents intents.json --output synthetic.json --count 50

Requirements:
  pip install google-cloud-aiplatform google-generativeai

Cost Estimates (March 2026):
  ---------------------------------------------------------------
  Model                  Input/1M tokens  Output/1M tokens  Batch
  ---------------------------------------------------------------
  Gemini 2.0 Flash       $0.15            $0.60             50%
  Gemini 2.5 Flash       $0.30            $2.50             50%
  Gemini 2.5 Pro         $1.25            $10.00            50%
  ---------------------------------------------------------------

  For 12 intents x 50 synthetic examples each (600 examples):
  - ~15K input tokens (prompts) + ~30K output tokens (responses)
  - Gemini 2.0 Flash: ~$0.02 online, ~$0.01 batch
  - Gemini 2.5 Flash: ~$0.08 online, ~$0.04 batch
  - Total pipeline (1000 augmented examples): $0.02 - $0.10
"""

import argparse
import json
import os
import random
import time
from pathlib import Path


# =============================================================================
# Prompts for synthetic data generation
# =============================================================================

GENERATION_PROMPT = """You are an expert NLU training data generator. Generate diverse, realistic training examples for intent classification in a customer service chatbot.

## Intent: {intent_name}
## Description: {intent_description}

## Existing examples (for style reference):
{existing_examples}

## Instructions:
1. Generate {count} NEW unique training examples for the "{intent_name}" intent
2. Vary the examples by:
   - Different phrasings and sentence structures
   - Formal and informal language
   - Short and long utterances (3-20 words)
   - Questions, statements, and commands
   - Include typos/casual language occasionally (10% of examples)
   - Different levels of politeness
   - Include context-rich examples ("I ordered 3 days ago and...")
3. Do NOT repeat or closely paraphrase the existing examples
4. Each example should clearly belong to the "{intent_name}" intent

## Output format (JSON array of strings, no other text):
["example 1", "example 2", ...]
"""

AUGMENTATION_PROMPT = """You are a text augmentation expert. Given the original text, generate {count} paraphrases that preserve the same intent and meaning but use different words and sentence structures.

## Original text: "{text}"
## Intent: {intent}

## Rules:
1. Preserve the original meaning and intent
2. Use different vocabulary and phrasing
3. Vary formality level (casual to formal)
4. One example should have a minor typo
5. One should be shorter, one should be longer

## Output format (JSON array of strings, no other text):
"""

MULTI_INTENT_PROMPT = """Generate a diverse set of customer service training examples. For each intent below, create {count_per_intent} unique, realistic examples.

## Intents:
{intent_list}

## Rules:
1. Make examples diverse: questions, commands, statements
2. Include casual and formal language
3. Keep examples between 3-25 words
4. Make sure each example clearly belongs to ONE intent only
5. Include some examples with typos or informal language

## Output as JSON object with intent names as keys and arrays of example strings as values:
"""


# =============================================================================
# Provider: Vertex AI
# =============================================================================

def generate_with_vertex(prompt: str, model: str = "gemini-2.0-flash") -> str:
    """Generate text using Vertex AI."""
    from google.cloud import aiplatform
    from vertexai.generative_models import GenerativeModel

    # Initialize Vertex AI (uses default credentials from gcloud auth)
    project = os.environ.get("GCP_PROJECT_ID", os.environ.get("GOOGLE_CLOUD_PROJECT", ""))
    location = os.environ.get("GCP_REGION", "us-central1")

    if project:
        aiplatform.init(project=project, location=location)

    model_instance = GenerativeModel(model)
    response = model_instance.generate_content(
        prompt,
        generation_config={
            "temperature": 0.8,
            "top_p": 0.95,
            "max_output_tokens": 4096,
        },
    )
    return response.text


def batch_generate_vertex(prompts: list[str], model: str = "gemini-2.0-flash") -> list[str]:
    """
    Batch generation via Vertex AI (50% cost reduction).
    Uses BigQuery or GCS for input/output.
    """
    from google.cloud import aiplatform
    from vertexai.generative_models import GenerativeModel
    import vertexai

    project = os.environ.get("GCP_PROJECT_ID", "")
    location = os.environ.get("GCP_REGION", "us-central1")

    if project:
        vertexai.init(project=project, location=location)

    # For batch predictions, write prompts to a JSONL file in GCS
    bucket = os.environ.get("GCS_BUCKET", "")
    if not bucket:
        log("GCS_BUCKET not set. Falling back to online generation.")
        return [generate_with_vertex(p, model) for p in prompts]

    from google.cloud import storage

    client = storage.Client()
    bucket_obj = client.bucket(bucket)

    # Upload input JSONL
    timestamp = int(time.time())
    input_path = f"batch-input/nlu-gen-{timestamp}.jsonl"
    output_path = f"batch-output/nlu-gen-{timestamp}/"

    input_data = ""
    for i, prompt in enumerate(prompts):
        input_data += json.dumps({
            "request": {
                "contents": [{"role": "user", "parts": [{"text": prompt}]}],
                "generation_config": {
                    "temperature": 0.8,
                    "top_p": 0.95,
                    "max_output_tokens": 4096,
                },
            },
        }) + "\n"

    blob = bucket_obj.blob(input_path)
    blob.upload_from_string(input_data)
    log(f"Uploaded batch input to gs://{bucket}/{input_path}")

    # Submit batch job
    model_instance = GenerativeModel(model)

    batch_job = model_instance.batch_predict(
        source_uri=f"gs://{bucket}/{input_path}",
        destination_uri_prefix=f"gs://{bucket}/{output_path}",
    )

    log(f"Batch job submitted: {batch_job.resource_name}")
    log("Waiting for batch completion...")

    # Poll for completion
    while batch_job.state.name not in ["JOB_STATE_SUCCEEDED", "JOB_STATE_FAILED"]:
        time.sleep(30)
        batch_job.refresh()
        log(f"Batch state: {batch_job.state.name}")

    if batch_job.state.name == "JOB_STATE_FAILED":
        raise RuntimeError(f"Batch job failed: {batch_job.error}")

    # Download results
    results = []
    blobs = list(bucket_obj.list_blobs(prefix=output_path))
    for blob in blobs:
        if blob.name.endswith(".jsonl"):
            content = blob.download_as_text()
            for line in content.strip().split("\n"):
                result = json.loads(line)
                text = result.get("response", {}).get("candidates", [{}])[0].get(
                    "content", {}
                ).get("parts", [{}])[0].get("text", "")
                results.append(text)

    log(f"Batch complete. Got {len(results)} responses.")
    return results


# =============================================================================
# Provider: Gemini Developer API (simpler, no GCP project needed)
# =============================================================================

def generate_with_gemini_api(prompt: str, model: str = "gemini-2.0-flash") -> str:
    """Generate text using Gemini Developer API."""
    import google.generativeai as genai

    api_key = os.environ.get("GOOGLE_API_KEY", "")
    if not api_key:
        raise RuntimeError(
            "Set GOOGLE_API_KEY env var. Get one at https://aistudio.google.com/apikey"
        )

    genai.configure(api_key=api_key)
    model_instance = genai.GenerativeModel(model)

    response = model_instance.generate_content(
        prompt,
        generation_config=genai.GenerationConfig(
            temperature=0.8,
            top_p=0.95,
            max_output_tokens=4096,
        ),
    )
    return response.text


# =============================================================================
# Core generation logic
# =============================================================================

def parse_json_response(text: str) -> list | dict:
    """Parse JSON from model response (handles markdown code blocks)."""
    text = text.strip()

    # Remove markdown code blocks
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Try to extract JSON array or object from the text
        for start_char, end_char in [("[", "]"), ("{", "}")]:
            start = text.find(start_char)
            end = text.rfind(end_char)
            if start != -1 and end != -1 and end > start:
                try:
                    return json.loads(text[start : end + 1])
                except json.JSONDecodeError:
                    continue

        raise ValueError(f"Could not parse JSON from response: {text[:200]}...")


def generate_for_intent(
    intent_name: str,
    intent_description: str,
    existing_examples: list[str],
    count: int,
    generate_fn,
) -> list[str]:
    """Generate synthetic examples for a single intent."""
    # Show a few existing examples for context
    sample = random.sample(existing_examples, min(5, len(existing_examples)))
    examples_str = "\n".join(f"  - {e}" for e in sample)

    prompt = GENERATION_PROMPT.format(
        intent_name=intent_name,
        intent_description=intent_description,
        existing_examples=examples_str,
        count=count,
    )

    response = generate_fn(prompt)
    parsed = parse_json_response(response)

    if isinstance(parsed, list):
        # Filter out duplicates and near-duplicates of existing examples
        existing_lower = {e.lower().strip() for e in existing_examples}
        new_examples = []
        for ex in parsed:
            if isinstance(ex, str) and ex.lower().strip() not in existing_lower:
                new_examples.append(ex.strip())
                existing_lower.add(ex.lower().strip())
        return new_examples
    else:
        log(f"Unexpected response format for {intent_name}")
        return []


def augment_examples(
    text: str,
    intent: str,
    count: int,
    generate_fn,
) -> list[str]:
    """Generate paraphrases of a specific example."""
    prompt = AUGMENTATION_PROMPT.format(text=text, intent=intent, count=count)
    response = generate_fn(prompt)
    parsed = parse_json_response(response)
    return [str(ex).strip() for ex in parsed] if isinstance(parsed, list) else []


def generate_multi_intent(
    intents: dict[str, str],  # name -> description
    count_per_intent: int,
    generate_fn,
) -> dict[str, list[str]]:
    """Generate examples for multiple intents in a single call (cheaper)."""
    intent_list = "\n".join(
        f"- {name}: {desc}" for name, desc in intents.items()
    )

    prompt = MULTI_INTENT_PROMPT.format(
        intent_list=intent_list,
        count_per_intent=count_per_intent,
    )

    response = generate_fn(prompt)
    parsed = parse_json_response(response)

    if isinstance(parsed, dict):
        return {k: [str(v) for v in vs] for k, vs in parsed.items() if isinstance(vs, list)}
    else:
        log("Multi-intent generation returned non-dict format")
        return {}


# =============================================================================
# Main pipeline
# =============================================================================

def run_generation(
    input_path: str,
    output_path: str,
    provider: str = "gemini",
    model: str = "gemini-2.0-flash",
    count_per_intent: int = 50,
    augment: bool = True,
    batch: bool = False,
):
    """Run the full synthetic data generation pipeline."""
    log("=" * 60)
    log("Synthetic Data Generation Pipeline")
    log("=" * 60)
    log(f"Provider: {provider}")
    log(f"Model: {model}")
    log(f"Count per intent: {count_per_intent}")

    # Select generation function
    if provider == "vertex":
        if batch:
            generate_fn = lambda p: batch_generate_vertex([p], model)[0]
        else:
            generate_fn = lambda p: generate_with_vertex(p, model)
    elif provider == "gemini":
        generate_fn = lambda p: generate_with_gemini_api(p, model)
    else:
        raise ValueError(f"Unknown provider: {provider}")

    # Load existing training data
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    intents = data.get("intents", [])
    total_generated = 0

    for intent_obj in intents:
        intent_name = intent_obj["name"]
        intent_desc = intent_obj.get("description", intent_name)
        existing = [
            ex["text"] if isinstance(ex, dict) else str(ex)
            for ex in intent_obj.get("examples", [])
        ]

        log(f"\nGenerating for: {intent_name} ({len(existing)} existing)")

        try:
            new_examples = generate_for_intent(
                intent_name, intent_desc, existing, count_per_intent, generate_fn
            )
            log(f"  Generated {len(new_examples)} new examples")

            # Add to intent
            for text in new_examples:
                intent_obj["examples"].append({
                    "text": text,
                    "entities": [],
                    "_synthetic": True,
                })
                total_generated += 1

            # Rate limiting (be nice to the API)
            time.sleep(1)

        except Exception as e:
            log(f"  Failed: {e}")
            continue

    # Update metadata
    if "metadata" in data:
        data["metadata"]["totalExamples"] = sum(
            len(i.get("examples", [])) for i in data["intents"]
        )
        data["metadata"]["syntheticCount"] = total_generated

    # Save
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    log(f"\nGenerated {total_generated} total synthetic examples")
    log(f"Output saved to: {output_path}")

    # Cost estimate
    est_input_tokens = total_generated * 200  # ~200 tokens per prompt
    est_output_tokens = total_generated * 50  # ~50 tokens per response
    costs = {
        "gemini-2.0-flash": (0.15, 0.60),
        "gemini-2.5-flash": (0.30, 2.50),
        "gemini-2.5-pro": (1.25, 10.00),
    }
    if model in costs:
        input_cost, output_cost = costs[model]
        total_cost = (est_input_tokens * input_cost + est_output_tokens * output_cost) / 1_000_000
        if batch:
            total_cost *= 0.5
        log(f"\nEstimated cost: ${total_cost:.4f}")
        log(f"  Input tokens:  ~{est_input_tokens:,}")
        log(f"  Output tokens: ~{est_output_tokens:,}")


def log(msg: str):
    print(f"[SYNTH] {msg}")


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic NLU training data using Gemini",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Environment variables:
  GOOGLE_API_KEY      - Gemini Developer API key (for --provider gemini)
  GCP_PROJECT_ID      - GCP project ID (for --provider vertex)
  GCP_REGION          - GCP region (default: us-central1)
  GCS_BUCKET          - GCS bucket for batch predictions

Examples:
  # Quick generation with Gemini API
  export GOOGLE_API_KEY=your-key-here
  python generate_synthetic.py -i training_data.json -o augmented.json -c 30

  # Vertex AI with batch (50% cheaper)
  python generate_synthetic.py -i training_data.json -o augmented.json --provider vertex --batch

Cost estimates for 12 intents x 50 examples:
  gemini-2.0-flash:  ~$0.02 (online) / ~$0.01 (batch)
  gemini-2.5-flash:  ~$0.08 (online) / ~$0.04 (batch)
  gemini-2.5-pro:    ~$0.35 (online) / ~$0.18 (batch)
        """,
    )
    parser.add_argument("--input", "-i", required=True, help="Input training data JSON")
    parser.add_argument("--output", "-o", default="augmented_data.json", help="Output file")
    parser.add_argument("--provider", "-p", choices=["gemini", "vertex"], default="gemini",
                        help="API provider (default: gemini)")
    parser.add_argument("--model", "-m", default="gemini-2.0-flash",
                        help="Model name (default: gemini-2.0-flash)")
    parser.add_argument("--count", "-c", type=int, default=50,
                        help="Synthetic examples per intent (default: 50)")
    parser.add_argument("--batch", action="store_true",
                        help="Use batch prediction (50%% cheaper, requires GCS_BUCKET)")

    args = parser.parse_args()

    run_generation(
        input_path=args.input,
        output_path=args.output,
        provider=args.provider,
        model=args.model,
        count_per_intent=args.count,
        batch=args.batch,
    )


if __name__ == "__main__":
    main()
