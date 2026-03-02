// Shared types for the vulnerability triage feature

export type Severity = "Critical" | "High" | "Medium" | "Low";

// ─── Existing (moved from page.tsx) ─────────────────────────────────────────

export interface Prediction {
  cwe: string;
  score: number;
  name: string;
  description: string;
  severity: Severity;
  owasp: string;
  remediation: string[];
  category: string;
}

export interface ClassifyResult {
  predictions: Prediction[];
  inferenceMs: number;
  modelInfo: {
    parameters: string;
    classes: number;
    architecture: string;
  };
}

// ─── Mode 1: CVE ID Lookup (NVD) ───────────────────────────────────────────

export interface AffectedProduct {
  vendor: string;
  product: string;
  versions: string;
}

export interface NVDCVEResponse {
  cveId: string;
  description: string;
  publishedDate: string;
  lastModifiedDate: string;
  cvssV3Score: number | null;
  cvssV3Vector: string | null;
  cvssV3Severity: string | null;
  cvssV2Score: number | null;
  cwes: string[];
  affectedProducts: AffectedProduct[];
  references: { url: string; source: string }[];
}

export interface CVELookupResult {
  nvd: NVDCVEResponse;
  classification: ClassifyResult;
  cweMatch: {
    nvdCwes: string[];
    predictedCwe: string;
    agreement: boolean;
  };
}

// ─── Mode 2: Code Scanner ──────────────────────────────────────────────────

export type ScanLanguage =
  | "javascript"
  | "typescript"
  | "python"
  | "java"
  | "csharp"
  | "cpp"
  | "go"
  | "php"
  | "ruby"
  | "unknown";

export interface CodeFinding {
  line: number;
  column: number;
  snippet: string;
  patternId: string;
  cwe: string;
  severity: Severity;
  title: string;
  explanation: string;
  fix: string;
}

export interface CodeScanResult {
  language: ScanLanguage;
  findings: CodeFinding[];
  scannedLines: number;
  scanTimeMs: number;
}

// ─── Mode 3: Dependency Scanner ─────────────────────────────────────────────

export type DependencyFileType = "package.json" | "requirements.txt" | "pom.xml";

export interface ParsedDependency {
  name: string;
  version: string;
  ecosystem: "npm" | "PyPI" | "Maven";
}

export interface DependencyVulnerability {
  id: string;
  summary: string;
  severity: Severity;
  fixedVersion: string | null;
  publishedDate: string;
  cwes: string[];
}

export interface DependencyResult {
  name: string;
  version: string;
  ecosystem: "npm" | "PyPI" | "Maven";
  vulnerabilities: DependencyVulnerability[];
}

export interface DependencyScanResult {
  fileType: DependencyFileType;
  dependencies: DependencyResult[];
  totalDeps: number;
  vulnerableDeps: number;
  totalVulns: number;
  scanTimeMs: number;
}
