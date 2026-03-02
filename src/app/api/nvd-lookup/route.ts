import { NextResponse } from "next/server";
import type { NVDCVEResponse, AffectedProduct } from "@/lib/vuln/types";

const CVE_RE = /^CVE-\d{4}-\d{4,}$/;

export async function GET(request: Request) {
  const { searchParams } = new URL(request.url);
  const cveId = searchParams.get("cveId")?.trim() ?? "";

  if (!CVE_RE.test(cveId)) {
    return NextResponse.json(
      { error: "Invalid CVE ID. Expected format: CVE-YYYY-NNNNN+" },
      { status: 400 },
    );
  }

  const url = `https://services.nvd.nist.gov/rest/json/cves/2.0?cveId=${cveId}`;
  const headers: HeadersInit = { Accept: "application/json" };
  if (process.env.NVD_API_KEY) {
    headers.apiKey = process.env.NVD_API_KEY;
  }

  const ctrl = new AbortController();
  const timer = setTimeout(() => ctrl.abort(), 10_000);

  try {
    const res = await fetch(url, { headers, signal: ctrl.signal });
    clearTimeout(timer);

    if (res.status === 404 || res.status === 403) {
      return NextResponse.json({ error: `CVE ${cveId} not found` }, { status: 404 });
    }
    if (res.status === 429) {
      return NextResponse.json(
        { error: "NVD rate limit exceeded. Try again shortly." },
        { status: 429 },
      );
    }
    if (!res.ok) {
      return NextResponse.json(
        { error: `NVD returned ${res.status}` },
        { status: 502 },
      );
    }

    /* eslint-disable @typescript-eslint/no-explicit-any */
    const data = await res.json();
    const vulns = data?.vulnerabilities;
    if (!Array.isArray(vulns) || vulns.length === 0) {
      return NextResponse.json({ error: `CVE ${cveId} not found` }, { status: 404 });
    }

    const cve = vulns[0].cve;

    // Description (English)
    const description =
      cve.descriptions?.find((d: any) => d.lang === "en")?.value ?? "";

    // CVSS v3.1
    const v31 = cve.metrics?.cvssMetricV31?.[0]?.cvssData;
    const cvssV3Score: number | null = v31?.baseScore ?? null;
    const cvssV3Vector: string | null = v31?.vectorString ?? null;
    const cvssV3Severity: string | null = v31?.baseSeverity ?? null;

    // CVSS v2
    const cvssV2Score: number | null =
      cve.metrics?.cvssMetricV2?.[0]?.cvssData?.baseScore ?? null;

    // CWEs
    const cwes: string[] = [];
    const IGNORE = new Set(["NVD-CWE-Other", "NVD-CWE-noinfo"]);
    for (const w of cve.weaknesses ?? []) {
      for (const d of w.description ?? []) {
        if (d.value && !IGNORE.has(d.value)) cwes.push(d.value);
      }
    }

    // Affected products from CPE 2.3 URIs
    const affectedProducts: AffectedProduct[] = [];
    const seen = new Set<string>();
    for (const cfg of cve.configurations ?? []) {
      for (const node of cfg.nodes ?? []) {
        for (const m of node.cpeMatch ?? []) {
          const parts = (m.criteria as string)?.split(":");
          if (!parts || parts.length < 6) continue;
          const vendor = parts[3];
          const product = parts[4];
          const version = parts[5] === "*" ? "all" : parts[5];
          const key = `${vendor}:${product}:${version}`;
          if (seen.has(key)) continue;
          seen.add(key);
          affectedProducts.push({ vendor, product, versions: version });
        }
      }
    }

    // References
    const references: { url: string; source: string }[] = (
      cve.references ?? []
    ).map((r: any) => ({ url: r.url, source: r.source ?? "" }));

    const nvdResponse: NVDCVEResponse = {
      cveId,
      description,
      publishedDate: cve.published ?? "",
      lastModifiedDate: cve.lastModified ?? "",
      cvssV3Score,
      cvssV3Vector,
      cvssV3Severity,
      cvssV2Score,
      cwes,
      affectedProducts,
      references,
    };

    return NextResponse.json(nvdResponse);
    /* eslint-enable @typescript-eslint/no-explicit-any */
  } catch (err: unknown) {
    clearTimeout(timer);
    if (err instanceof DOMException && err.name === "AbortError") {
      return NextResponse.json(
        { error: "NVD request timed out (10 s)" },
        { status: 504 },
      );
    }
    const msg = err instanceof Error ? err.message : "Unknown error";
    return NextResponse.json({ error: msg }, { status: 500 });
  }
}
