// Code vulnerability pattern scanner — runs entirely client-side
// Detects common vulnerability patterns in pasted code using regex matching
// NOTE: This file contains regex patterns that MATCH dangerous constructs for DETECTION purposes only.

import { getCWEInfo } from "./cwe-database";
import type { CodeFinding, CodeScanResult, ScanLanguage, Severity } from "./types";

interface VulnPattern {
  id: string;
  cwe: string;
  title: string;
  regex: RegExp;
  severity: Severity;
  explanation: string;
  fix: string;
  languages: ScanLanguage[] | "all";
}

// ─── Language Detection ─────────────────────────────────────────────────────

export function detectLanguage(code: string): ScanLanguage {
  const lines = code.slice(0, 2000);
  if (/^<\?php/m.test(lines)) return "php";
  if (/^#include\s+[<"]/m.test(lines) || /\bint\s+main\s*\(/m.test(lines) || /\bstd::/m.test(lines)) return "cpp";
  if (/^package\s+main\b/m.test(lines) || /\bfunc\s+\w+\s*\(/m.test(lines) || /\bfmt\.\w+/m.test(lines)) return "go";
  if (/^using\s+System/m.test(lines) || /\bConsole\.Write/m.test(lines)) return "csharp";
  if (/\bpublic\s+class\b/m.test(lines) || /\bSystem\.out\./m.test(lines)) return "java";
  if (/^import\s+\w+/m.test(lines) && /\bdef\s+\w+\s*\(/m.test(lines)) return "python";
  if (/\bdef\s+\w+\s*\(/m.test(lines) || /^from\s+\w+\s+import/m.test(lines) || /\bprint\s*\(/m.test(lines)) return "python";
  if (/\binterface\s+\w+/m.test(lines) || /:\s*(string|number|boolean|any)\b/m.test(lines)) return "typescript";
  if (/\bconst\s+\w+\s*=/m.test(lines) || /\b(?:require|import)\s*\(/m.test(lines) || /=>\s*\{/m.test(lines)) return "javascript";
  return "unknown";
}

// ─── Pattern Registry ───────────────────────────────────────────────────────

const PATTERNS: VulnPattern[] = [
  // ── SQL Injection (CWE-89) ──
  {
    id: "sqli-string-concat",
    cwe: "CWE-89",
    title: "SQL Injection via String Concatenation",
    regex: /["'`](?:SELECT|INSERT|UPDATE|DELETE|DROP|ALTER|CREATE)\s.+?["'`]\s*\+/i,
    severity: "Critical",
    explanation: "Building SQL queries by concatenating user input allows attackers to inject arbitrary SQL commands.",
    fix: "Use parameterized queries or prepared statements. Never concatenate user input into SQL strings.",
    languages: "all",
  },
  {
    id: "sqli-template-literal",
    cwe: "CWE-89",
    title: "SQL Injection via Template Literal",
    regex: /`(?:SELECT|INSERT|UPDATE|DELETE|DROP|ALTER|CREATE)\s[^`]*\$\{/i,
    severity: "Critical",
    explanation: "Template literals with interpolated variables in SQL queries are vulnerable to injection.",
    fix: "Use parameterized queries: db.query('SELECT * FROM users WHERE id = ?', [userId])",
    languages: ["javascript", "typescript"],
  },
  {
    id: "sqli-fstring",
    cwe: "CWE-89",
    title: "SQL Injection via f-string",
    regex: /f["'](?:SELECT|INSERT|UPDATE|DELETE|DROP|ALTER|CREATE)\s.*?\{/i,
    severity: "Critical",
    explanation: "Python f-strings in SQL queries allow injection of arbitrary SQL.",
    fix: "Use parameterized queries: cursor.execute('SELECT * FROM users WHERE id = %s', (user_id,))",
    languages: ["python"],
  },
  {
    id: "sqli-format",
    cwe: "CWE-89",
    title: "SQL Injection via .format()",
    regex: /["'](?:SELECT|INSERT|UPDATE|DELETE)\s.+?["']\.format\s*\(/i,
    severity: "Critical",
    explanation: "Using .format() to build SQL queries injects unsanitized input directly.",
    fix: "Use parameterized queries with the database driver instead of string formatting.",
    languages: ["python"],
  },

  // ── XSS (CWE-79) ──
  {
    id: "xss-innerhtml",
    cwe: "CWE-79",
    title: "XSS via innerHTML Assignment",
    regex: /\.innerHTML\s*=(?!=)/,
    severity: "High",
    explanation: "Setting innerHTML with user-controlled data allows script injection in the browser.",
    fix: "Use textContent or innerText instead. If HTML is needed, sanitize with DOMPurify.",
    languages: ["javascript", "typescript"],
  },
  {
    id: "xss-document-write",
    cwe: "CWE-79",
    title: "XSS via document.write",
    regex: /document\.write(?:ln)?\s*\(/,
    severity: "High",
    explanation: "document.write() injects arbitrary HTML/JS into the page.",
    fix: "Use DOM APIs (createElement, textContent) instead.",
    languages: ["javascript", "typescript"],
  },
  {
    id: "xss-react-unsafe-html",
    cwe: "CWE-79",
    title: "XSS via Unsafe HTML Rendering (React)",
    regex: /dangerouslySetInner/,
    severity: "High",
    explanation: "React unsafe HTML rendering bypasses XSS protection when used with unsanitized data.",
    fix: "Sanitize with DOMPurify before rendering, or avoid raw HTML rendering.",
    languages: ["javascript", "typescript"],
  },
  {
    id: "xss-vue-html",
    cwe: "CWE-79",
    title: "XSS via v-html Directive",
    regex: /v-html\s*=/,
    severity: "High",
    explanation: "Vue v-html renders raw HTML. User data flowing into it enables XSS.",
    fix: "Use v-text for text content, or sanitize before using v-html.",
    languages: ["javascript", "typescript"],
  },

  // ── Command Injection (CWE-78) ──
  {
    id: "cmdi-shell-js",
    cwe: "CWE-78",
    title: "Command Injection via Shell (Node.js)",
    regex: /child_process\s*[\.\s]+(exec|execSync)\s*\(/,
    severity: "Critical",
    explanation: "Shell execution with user input allows arbitrary command injection on the server.",
    fix: "Use execFile() or spawn() with argument arrays instead.",
    languages: ["javascript", "typescript"],
  },
  {
    id: "cmdi-os-system-py",
    cwe: "CWE-78",
    title: "Command Injection via os.system/subprocess",
    regex: /(?:os\.system|os\.popen|subprocess\.call|subprocess\.run|subprocess\.Popen)\s*\(/,
    severity: "Critical",
    explanation: "Shell commands with user input allow arbitrary command injection.",
    fix: "Use subprocess.run() with shell=False and pass arguments as a list.",
    languages: ["python"],
  },
  {
    id: "cmdi-runtime-java",
    cwe: "CWE-78",
    title: "Command Injection via Runtime (Java)",
    regex: /Runtime\.getRuntime\(\)\s*\.\s*exec\s*\(/,
    severity: "Critical",
    explanation: "Java Runtime can run arbitrary OS commands if input is not validated.",
    fix: "Use ProcessBuilder with an explicit argument list.",
    languages: ["java"],
  },
  {
    id: "cmdi-system-c",
    cwe: "CWE-78",
    title: "Command Injection via system() (C/C++)",
    regex: /\bsystem\s*\(\s*[^"')\s]/,
    severity: "Critical",
    explanation: "The C system() function passes strings directly to the shell.",
    fix: "Use execvp() or posix_spawn() with explicit argument arrays.",
    languages: ["cpp"],
  },

  // ── Path Traversal (CWE-22) ──
  {
    id: "path-traversal-dotdot",
    cwe: "CWE-22",
    title: "Potential Path Traversal (../ sequence)",
    regex: /\.\.[/\\]/,
    severity: "Medium",
    explanation: "Relative path sequences can traverse outside intended directories.",
    fix: "Resolve paths canonically, then verify the result stays within the allowed base directory.",
    languages: "all",
  },
  {
    id: "path-traversal-user-input",
    cwe: "CWE-22",
    title: "Unsanitized File Path with User Input",
    regex: /(?:path\.join|path\.resolve|os\.path\.join)\s*\([^)]*(?:req\.|request\.|params|query|body|args)/,
    severity: "High",
    explanation: "Joining file paths with user-controlled input allows directory traversal.",
    fix: "Normalize the path, then check it starts with the allowed base directory.",
    languages: "all",
  },

  // ── Hardcoded Secrets (CWE-798) ──
  {
    id: "secret-aws-key",
    cwe: "CWE-798",
    title: "Hardcoded AWS Access Key",
    regex: /AKIA[0-9A-Z]{16}/,
    severity: "Critical",
    explanation: "AWS access keys in source code grant cloud resource access to anyone who finds them.",
    fix: "Use environment variables or IAM roles. Rotate this key immediately.",
    languages: "all",
  },
  {
    id: "secret-generic-key",
    cwe: "CWE-798",
    title: "Hardcoded API Key or Secret",
    regex: /(?:api[_-]?key|api[_-]?secret|auth[_-]?token|access[_-]?token|secret[_-]?key)\s*[:=]\s*["'][a-zA-Z0-9_\-/+=]{20,}["']/i,
    severity: "High",
    explanation: "API keys in source code can be extracted from version control history.",
    fix: "Store secrets in environment variables or a secrets manager.",
    languages: "all",
  },
  {
    id: "secret-private-key",
    cwe: "CWE-798",
    title: "Hardcoded Private Key",
    regex: /-----BEGIN (?:RSA |EC |DSA |OPENSSH )?PRIVATE KEY-----/,
    severity: "Critical",
    explanation: "Private keys in source code compromise all encrypted communications.",
    fix: "Store private keys in a secure key management system.",
    languages: "all",
  },
  {
    id: "secret-db-url",
    cwe: "CWE-798",
    title: "Hardcoded Database Connection String",
    regex: /(?:mongodb|postgres|mysql|redis|amqp):\/\/[^\s"']+:[^\s"']+@/i,
    severity: "Critical",
    explanation: "Database URLs with embedded credentials expose the database if code is shared.",
    fix: "Use environment variables for connection strings.",
    languages: "all",
  },
  {
    id: "secret-github-token",
    cwe: "CWE-798",
    title: "Hardcoded GitHub Token",
    regex: /ghp_[0-9a-zA-Z]{36}/,
    severity: "Critical",
    explanation: "GitHub tokens in code grant repository access to anyone who finds them.",
    fix: "Use environment variables. Revoke and regenerate the token.",
    languages: "all",
  },

  // ── Insecure Crypto (CWE-327/328/338) ──
  {
    id: "crypto-md5",
    cwe: "CWE-328",
    title: "Weak Hash Algorithm (MD5)",
    regex: /(?:createHash\s*\(\s*['"]md5['"]|hashlib\.md5|MessageDigest\.getInstance\s*\(\s*['"]MD5['"])/i,
    severity: "Medium",
    explanation: "MD5 is cryptographically broken. Collision attacks are practical.",
    fix: "Use SHA-256 for integrity, bcrypt/Argon2 for passwords.",
    languages: "all",
  },
  {
    id: "crypto-sha1",
    cwe: "CWE-328",
    title: "Weak Hash Algorithm (SHA-1)",
    regex: /(?:createHash\s*\(\s*['"]sha1['"]|hashlib\.sha1|MessageDigest\.getInstance\s*\(\s*['"]SHA-?1['"])/i,
    severity: "Medium",
    explanation: "SHA-1 has known collision attacks. Not suitable for security.",
    fix: "Upgrade to SHA-256 or SHA-3.",
    languages: "all",
  },
  {
    id: "crypto-math-random",
    cwe: "CWE-338",
    title: "Insecure Random Number Generator",
    regex: /Math\.random\s*\(\)/,
    severity: "Medium",
    explanation: "Math.random() is not cryptographically secure. Output is predictable.",
    fix: "Use crypto.getRandomValues() (browser) or crypto.randomBytes() (Node.js).",
    languages: ["javascript", "typescript"],
  },

  // ── Insecure Deserialization (CWE-502) ──
  {
    id: "deser-pickle",
    cwe: "CWE-502",
    title: "Insecure Deserialization (pickle)",
    regex: /pickle\.loads?\s*\(/,
    severity: "Critical",
    explanation: "Python pickle runs arbitrary code during deserialization.",
    fix: "Use JSON or a safe format. Never unpickle untrusted data.",
    languages: ["python"],
  },
  {
    id: "deser-yaml-unsafe",
    cwe: "CWE-502",
    title: "Insecure YAML Deserialization",
    regex: /yaml\.load\s*\([^)]*\)\s*(?!.*SafeLoader)/,
    severity: "High",
    explanation: "yaml.load() without SafeLoader can instantiate arbitrary Python objects.",
    fix: "Use yaml.safe_load() or pass Loader=yaml.SafeLoader.",
    languages: ["python"],
  },
  {
    id: "deser-code-eval",
    cwe: "CWE-94",
    title: "Dynamic Code Evaluation",
    regex: /\beval\s*\(\s*[^)'"]/,
    severity: "Critical",
    explanation: "Dynamic code evaluation runs arbitrary code. Dangerous with user input.",
    fix: "Use JSON.parse() for data. Use domain-specific parsers instead.",
    languages: "all",
  },

  // ── SSRF (CWE-918) ──
  {
    id: "ssrf-user-url",
    cwe: "CWE-918",
    title: "Potential SSRF via User-Controlled URL",
    regex: /(?:fetch|axios\.get|axios\.post|requests\.get|requests\.post|urllib\.request\.urlopen|http\.get)\s*\(\s*(?:req\.|request\.|params|query|body|args)/,
    severity: "High",
    explanation: "HTTP requests to user-controlled URLs can reach internal services.",
    fix: "Validate URLs against an allowlist. Block private IP ranges.",
    languages: "all",
  },

  // ── Buffer Overflow (CWE-120) ──
  {
    id: "bufover-strcpy",
    cwe: "CWE-120",
    title: "Buffer Overflow via strcpy()",
    regex: /\bstrcpy\s*\(/,
    severity: "Critical",
    explanation: "strcpy() copies without bounds checking, overwriting adjacent memory.",
    fix: "Use strncpy() or strlcpy() with explicit buffer size limits.",
    languages: ["cpp"],
  },
  {
    id: "bufover-gets",
    cwe: "CWE-120",
    title: "Buffer Overflow via gets()",
    regex: /\bgets\s*\(/,
    severity: "Critical",
    explanation: "gets() reads input with no buffer limit. Removed from C11 standard.",
    fix: "Use fgets(buffer, size, stdin) with explicit size.",
    languages: ["cpp"],
  },
  {
    id: "bufover-sprintf",
    cwe: "CWE-120",
    title: "Buffer Overflow via sprintf()",
    regex: /\bsprintf\s*\(/,
    severity: "High",
    explanation: "sprintf() writes without bounds checking.",
    fix: "Use snprintf(buffer, size, format, ...) with explicit size.",
    languages: ["cpp"],
  },
];

// ─── Scanner Entry Point ────────────────────────────────────────────────────

export function scanCode(code: string): CodeScanResult {
  const t0 = performance.now();
  const language = detectLanguage(code);
  const lines = code.split("\n");
  const findings: CodeFinding[] = [];
  const seen = new Set<string>();

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];
    const trimmed = line.trim();
    if (!trimmed || trimmed.startsWith("//") || trimmed.startsWith("#") || trimmed.startsWith("*") || trimmed.startsWith("/*")) {
      continue;
    }

    for (const pattern of PATTERNS) {
      if (pattern.languages !== "all" && !pattern.languages.includes(language)) continue;
      if (pattern.regex.test(line)) {
        const key = `${pattern.id}:${i}`;
        if (seen.has(key)) continue;
        seen.add(key);

        const match = line.match(pattern.regex);
        findings.push({
          line: i + 1,
          column: match?.index ?? 0,
          snippet: trimmed,
          patternId: pattern.id,
          cwe: pattern.cwe,
          severity: pattern.severity,
          title: pattern.title,
          explanation: pattern.explanation,
          fix: pattern.fix,
        });
      }
    }
  }

  const severityOrder: Record<string, number> = { Critical: 0, High: 1, Medium: 2, Low: 3 };
  findings.sort((a, b) => {
    const diff = (severityOrder[a.severity] ?? 9) - (severityOrder[b.severity] ?? 9);
    return diff !== 0 ? diff : a.line - b.line;
  });

  return { language, findings, scannedLines: lines.length, scanTimeMs: Math.round((performance.now() - t0) * 100) / 100 };
}

// ─── Sample Code for UI ─────────────────────────────────────────────────────

export const CODE_SAMPLES = [
  {
    label: "SQL Injection",
    language: "javascript",
    code: "const app = require('express')();\nconst db = require('./db');\n\napp.get('/user', (req, res) => {\n  const userId = req.query.id;\n  const query = \"SELECT * FROM users WHERE id = \" + userId;\n  db.query(query, (err, results) => {\n    res.json(results);\n  });\n});",
  },
  {
    label: "XSS + Secrets",
    language: "javascript",
    code: "const API_KEY = \"AKIAIOSFODNN7EXAMPLE1234\";\n\nfunction renderComment(comment) {\n  const div = document.createElement('div');\n  div.innerHTML = comment.text;\n  document.getElementById('comments').appendChild(div);\n}\n\nfunction search(query) {\n  document.write('<h1>Results for: ' + query + '</h1>');\n}",
  },
  {
    label: "Python Vulns",
    language: "python",
    code: "import os\nimport pickle\nimport subprocess\nimport yaml\n\ndef run_command(user_input):\n    os.system(\"ping \" + user_input)\n\ndef load_data(raw_bytes):\n    return pickle.loads(raw_bytes)\n\ndef run_task(cmd):\n    subprocess.call(cmd, shell=True)\n\ndef parse_config(text):\n    return yaml.load(text)\n\ndef get_user(name):\n    query = f\"SELECT * FROM users WHERE name = '{name}'\"\n    cursor.execute(query)",
  },
  {
    label: "C/C++ Buffer",
    language: "cpp",
    code: "#include <stdio.h>\n#include <string.h>\n\nvoid process_input(char *input) {\n    char buffer[64];\n    strcpy(buffer, input);\n    printf(\"Processed: %s\\n\", buffer);\n}\n\nint main() {\n    char name[256];\n    printf(\"Enter name: \");\n    gets(name);\n    char msg[100];\n    sprintf(msg, \"Hello, %s! Welcome.\", name);\n    printf(\"%s\\n\", msg);\n    return 0;\n}",
  },
];
