import { NextResponse } from "next/server";
import type {
  ParsedDependency,
  DependencyResult,
  DependencyVulnerability,
  Severity,
} from "@/lib/vuln/types";

// ---------------------------------------------------------------------------
// CVSS helpers
// ---------------------------------------------------------------------------

/** Extract the numeric base score from a CVSS v3 vector string. */
function parseCvssScore(vector: string): number | null {
  // CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H  — no embedded numeric
  // score, but the OSV `score` field *is* the vector string itself.
  // Some feeds embed a trailing "/score:N.N" or the field is just a float.
  const floatMatch = vector.match(/^(\d+(?:\.\d+)?)$/);
  if (floatMatch) return parseFloat(floatMatch[1]);

  // Try to pull a trailing score that some providers append.
  const trailingMatch = vector.match(/\/(\d+(?:\.\d+))$/);
  if (trailingMatch) return parseFloat(trailingMatch[1]);

  return null;
}

function scoreToseverity(score: number): Severity {
  if (score >= 9.0) return "Critical";
  if (score >= 7.0) return "High";
  if (score >= 4.0) return "Medium";
  return "Low";
}

function extractSeverity(vuln: Record<string, unknown>): Severity {
  // 1. vuln.severity array → parse CVSS score
  const sevArr = vuln.severity as { type?: string; score?: string }[] | undefined;
  if (Array.isArray(sevArr) && sevArr.length > 0) {
    const raw = sevArr[0].score ?? "";
    const score = parseCvssScore(raw);
    if (score !== null) return scoreToseverity(score);
  }

  // 2. database_specific.severity (string like "HIGH")
  const dbSpec = vuln.database_specific as Record<string, unknown> | undefined;
  if (dbSpec?.severity) {
    const s = (dbSpec.severity as string).toLowerCase();
    if (s === "critical") return "Critical";
    if (s === "high") return "High";
    if (s === "low") return "Low";
    return "Medium";
  }

  return "Medium";
}

// ---------------------------------------------------------------------------
// Route handler
// ---------------------------------------------------------------------------

const OSV_URL = "https://api.osv.dev/v1/query";
const BATCH_SIZE = 10;
const VALID_ECOSYSTEMS = new Set(["npm", "PyPI", "Maven"]);

export async function POST(request: Request) {
  // --- Parse & validate ------------------------------------------------
  let body: { dependencies?: unknown };
  try {
    body = await request.json();
  } catch {
    return NextResponse.json({ error: "Invalid JSON body" }, { status: 400 });
  }

  const deps = body.dependencies;
  if (!Array.isArray(deps) || deps.length === 0) {
    return NextResponse.json({ error: "dependencies must be a non-empty array" }, { status: 400 });
  }
  if (deps.length > 100) {
    return NextResponse.json({ error: "Maximum 100 dependencies per request" }, { status: 400 });
  }

  for (const d of deps) {
    if (!d?.name || !d?.version || !VALID_ECOSYSTEMS.has(d?.ecosystem)) {
      return NextResponse.json(
        { error: `Invalid dependency: each must have name, version, and ecosystem (npm|PyPI|Maven)` },
        { status: 400 },
      );
    }
  }

  const dependencies = deps as ParsedDependency[];

  // --- Query OSV in batches of 10 -------------------------------------
  const results: DependencyResult[] = [];

  try {
    for (let i = 0; i < dependencies.length; i += BATCH_SIZE) {
      const batch = dependencies.slice(i, i + BATCH_SIZE);

      const batchResults = await Promise.all(
        batch.map(async (dep): Promise<DependencyResult> => {
          const res = await fetch(OSV_URL, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
              package: { name: dep.name, ecosystem: dep.ecosystem },
              version: dep.version,
            }),
          });

          if (!res.ok) {
            // Treat individual failures as zero vulns rather than aborting.
            return { name: dep.name, version: dep.version, ecosystem: dep.ecosystem, vulnerabilities: [] };
          }

          const data = (await res.json()) as { vulns?: Record<string, unknown>[] };
          const vulns: DependencyVulnerability[] = (data.vulns ?? []).map((v) => {
            // Fixed version: look through affected[].ranges[].events[]
            let fixedVersion: string | null = null;
            const affected = v.affected as { ranges?: { events?: Record<string, string>[] }[] }[] | undefined;
            if (Array.isArray(affected)) {
              for (const a of affected) {
                for (const r of a.ranges ?? []) {
                  const fixed = (r.events ?? []).find((e) => e.fixed);
                  if (fixed) { fixedVersion = fixed.fixed; break; }
                }
                if (fixedVersion) break;
              }
            }

            const dbSpec = v.database_specific as Record<string, unknown> | undefined;

            return {
              id: v.id as string,
              summary:
                (v.summary as string) ||
                ((v.details as string) ?? "").split("\n")[0] ||
                "No description available",
              severity: extractSeverity(v),
              fixedVersion,
              publishedDate: (v.published as string) || (v.modified as string) || "",
              cwes: (dbSpec?.cwe_ids as string[]) ?? [],
            };
          });

          return { name: dep.name, version: dep.version, ecosystem: dep.ecosystem, vulnerabilities: vulns };
        }),
      );

      results.push(...batchResults);
    }
  } catch (err) {
    const message = err instanceof Error ? err.message : "Unknown error";
    return NextResponse.json({ error: `OSV API request failed: ${message}` }, { status: 500 });
  }

  return NextResponse.json(results);
}
