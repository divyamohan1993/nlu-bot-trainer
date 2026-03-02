/**
 * CWE Enrichment Database
 *
 * Static lookup for all 349 CWE classes supported by the vulnerability classifier.
 * Provides human-readable names, descriptions, severity, OWASP mapping, and remediation.
 *
 * Data sourced from MITRE CWE (https://cwe.mitre.org) and OWASP Top 10 2021.
 */

export interface CWEEntry {
  id: string;
  name: string;
  description: string;
  severity: "Critical" | "High" | "Medium" | "Low";
  owasp: string;
  remediation: string[];
  category: string;
}

// ─── Curated entries for the most impactful CWEs ──────────────────────────────

const CURATED: Record<string, Omit<CWEEntry, "id">> = {
  "CWE-79": {
    name: "Cross-site Scripting (XSS)",
    description: "The application includes untrusted data in web page output without proper validation or escaping, allowing attackers to execute scripts in a victim's browser. This can lead to session hijacking, defacement, or redirection to malicious sites.",
    severity: "High",
    owasp: "A03:2021 Injection",
    remediation: [
      "Encode all output using context-aware escaping (HTML, JS, URL, CSS)",
      "Implement a strict Content Security Policy (CSP) header",
      "Use frameworks that auto-escape by default (React, Angular)",
      "Validate and sanitize all user input on the server side",
    ],
    category: "Injection",
  },
  "CWE-89": {
    name: "SQL Injection",
    description: "The application constructs SQL queries using unsanitized user input, allowing attackers to execute arbitrary SQL commands. This can expose, modify, or delete database contents and potentially compromise the entire server.",
    severity: "Critical",
    owasp: "A03:2021 Injection",
    remediation: [
      "Use parameterized queries or prepared statements exclusively",
      "Apply the principle of least privilege to database accounts",
      "Validate and sanitize all user input before use in queries",
      "Deploy a Web Application Firewall (WAF) as defense-in-depth",
    ],
    category: "Injection",
  },
  "CWE-78": {
    name: "OS Command Injection",
    description: "The application passes unsanitized user input to a system shell command, allowing attackers to execute arbitrary OS commands. This typically leads to full server compromise.",
    severity: "Critical",
    owasp: "A03:2021 Injection",
    remediation: [
      "Avoid calling OS commands from application code entirely",
      "Use language-native libraries instead of shell commands",
      "If unavoidable, use strict allowlists for permitted characters",
      "Run the application with minimal OS privileges",
    ],
    category: "Injection",
  },
  "CWE-77": {
    name: "Command Injection",
    description: "The application constructs commands using externally-influenced input without proper neutralization. Attackers can inject additional commands that execute with the application's privileges.",
    severity: "Critical",
    owasp: "A03:2021 Injection",
    remediation: [
      "Use parameterized APIs instead of constructing command strings",
      "Validate input against a strict allowlist of expected values",
      "Escape special characters appropriate to the command interpreter",
      "Run processes with least-privilege permissions",
    ],
    category: "Injection",
  },
  "CWE-74": {
    name: "Injection",
    description: "The application constructs part of a command, data structure, or query using externally-influenced input without neutralizing special elements. This is the parent category for all injection vulnerabilities.",
    severity: "High",
    owasp: "A03:2021 Injection",
    remediation: [
      "Use structured mechanisms that separate data from commands",
      "Apply context-specific output encoding",
      "Validate all input against expected formats",
      "Implement defense-in-depth with WAF rules",
    ],
    category: "Injection",
  },
  "CWE-94": {
    name: "Code Injection",
    description: "The application allows user input to be interpreted as code, enabling attackers to inject and execute arbitrary code on the server or client. This can lead to complete system compromise.",
    severity: "Critical",
    owasp: "A03:2021 Injection",
    remediation: [
      "Never pass user input to eval(), exec(), or similar dynamic execution functions",
      "Use sandboxed execution environments if dynamic code is required",
      "Validate input against strict allowlists",
      "Apply Content Security Policy to prevent client-side code injection",
    ],
    category: "Injection",
  },
  "CWE-90": {
    name: "LDAP Injection",
    description: "The application constructs LDAP queries using unsanitized user input, allowing attackers to modify query logic. This can bypass authentication or expose directory information.",
    severity: "High",
    owasp: "A03:2021 Injection",
    remediation: [
      "Use parameterized LDAP queries or LDAP frameworks with built-in escaping",
      "Validate input against strict allowlists of expected characters",
      "Escape special LDAP characters (*, (, ), \\, NUL) in user input",
    ],
    category: "Injection",
  },
  "CWE-611": {
    name: "XML External Entity (XXE) Injection",
    description: "The application processes XML input containing references to external entities, which can be exploited to access local files, perform SSRF attacks, or cause denial of service.",
    severity: "High",
    owasp: "A05:2021 Security Misconfiguration",
    remediation: [
      "Disable external entity and DTD processing in all XML parsers",
      "Use simpler data formats like JSON where possible",
      "Validate and sanitize XML input on the server side",
      "Apply least privilege to file system and network access",
    ],
    category: "Injection",
  },
  "CWE-917": {
    name: "Expression Language Injection",
    description: "The application evaluates expression language statements constructed from user input, allowing attackers to execute arbitrary code or access sensitive data through the expression engine.",
    severity: "Critical",
    owasp: "A03:2021 Injection",
    remediation: [
      "Never include user input in expression language statements",
      "Use parameterized templates instead of string concatenation",
      "Sandbox the expression evaluation context",
    ],
    category: "Injection",
  },
  "CWE-918": {
    name: "Server-Side Request Forgery (SSRF)",
    description: "The application fetches remote resources using a URL provided by the user without proper validation, allowing attackers to scan internal networks, access internal services, or exfiltrate data.",
    severity: "High",
    owasp: "A10:2021 Server-Side Request Forgery",
    remediation: [
      "Validate and sanitize all user-supplied URLs against an allowlist",
      "Block requests to internal/private IP ranges (10.x, 172.16.x, 192.168.x, 127.x)",
      "Use a dedicated egress proxy for outbound requests",
      "Disable unnecessary URL schemes (file://, gopher://, etc.)",
    ],
    category: "Input Validation",
  },
  "CWE-22": {
    name: "Path Traversal",
    description: "The application uses user input to construct file paths without proper sanitization, allowing attackers to access files outside the intended directory using sequences like '../'. This can expose source code, configuration, or sensitive data.",
    severity: "High",
    owasp: "A01:2021 Broken Access Control",
    remediation: [
      "Canonicalize file paths and verify they remain within the intended directory",
      "Use an allowlist of permitted filenames or a file ID mapping",
      "Strip or reject path traversal sequences (../, ..\\ and encoded variants)",
      "Run the application with minimal file system permissions",
    ],
    category: "Input Validation",
  },
  "CWE-20": {
    name: "Improper Input Validation",
    description: "The application does not validate or insufficiently validates input, which can lead to unexpected behavior, data corruption, or security vulnerabilities when the data is processed.",
    severity: "High",
    owasp: "A03:2021 Injection",
    remediation: [
      "Validate all input against expected types, ranges, lengths, and formats",
      "Use allowlists over denylists for input validation",
      "Validate on the server side even if client-side validation exists",
      "Reject unexpected input rather than attempting to sanitize it",
    ],
    category: "Input Validation",
  },
  "CWE-119": {
    name: "Buffer Overflow",
    description: "The application performs operations on a buffer without proper bounds checking, allowing attackers to write beyond the buffer's boundaries. This can corrupt memory, crash the application, or enable arbitrary code execution.",
    severity: "Critical",
    owasp: "A06:2021 Vulnerable and Outdated Components",
    remediation: [
      "Use memory-safe languages (Rust, Go, Java) or safe buffer APIs",
      "Enable compiler protections: ASLR, stack canaries, DEP/NX",
      "Validate buffer sizes before all read/write operations",
      "Use static analysis tools to detect buffer overflow patterns",
    ],
    category: "Memory Safety",
  },
  "CWE-125": {
    name: "Out-of-bounds Read",
    description: "The application reads data past the end or before the beginning of a buffer. This can expose sensitive information from memory or cause crashes.",
    severity: "High",
    owasp: "A06:2021 Vulnerable and Outdated Components",
    remediation: [
      "Validate array indices and buffer offsets before every read",
      "Use bounds-checked containers and iterators",
      "Enable memory safety tools (ASan, MSan) during testing",
    ],
    category: "Memory Safety",
  },
  "CWE-787": {
    name: "Out-of-bounds Write",
    description: "The application writes data past the end or before the beginning of a buffer. This can corrupt adjacent memory, crash the application, or enable arbitrary code execution.",
    severity: "Critical",
    owasp: "A06:2021 Vulnerable and Outdated Components",
    remediation: [
      "Validate buffer bounds before all write operations",
      "Use memory-safe languages or bounds-checked APIs",
      "Enable compiler hardening: stack protectors, ASLR, DEP",
      "Employ fuzzing and static analysis in CI/CD",
    ],
    category: "Memory Safety",
  },
  "CWE-416": {
    name: "Use After Free",
    description: "The application continues to reference memory after it has been freed, leading to undefined behavior. Attackers can exploit this to execute arbitrary code by manipulating the freed memory region.",
    severity: "Critical",
    owasp: "A06:2021 Vulnerable and Outdated Components",
    remediation: [
      "Set pointers to NULL immediately after freeing memory",
      "Use smart pointers (unique_ptr, shared_ptr) in C++",
      "Use memory-safe languages (Rust, Go) that prevent this class of bugs",
      "Run memory sanitizers (ASan, Valgrind) during testing",
    ],
    category: "Memory Safety",
  },
  "CWE-476": {
    name: "NULL Pointer Dereference",
    description: "The application dereferences a pointer that is expected to be valid but is NULL, causing a crash. While primarily a reliability issue, it can sometimes be exploited for code execution.",
    severity: "Medium",
    owasp: "A06:2021 Vulnerable and Outdated Components",
    remediation: [
      "Check pointers for NULL before dereferencing",
      "Use optional/nullable types that enforce checking at compile time",
      "Enable static analysis tools that detect null dereference patterns",
    ],
    category: "Memory Safety",
  },
  "CWE-190": {
    name: "Integer Overflow",
    description: "An arithmetic operation produces a result that exceeds the maximum value for the integer type, wrapping around to an unexpected value. This can lead to buffer overflows, incorrect calculations, or bypassed security checks.",
    severity: "High",
    owasp: "A06:2021 Vulnerable and Outdated Components",
    remediation: [
      "Use safe integer arithmetic libraries that detect overflow",
      "Validate input ranges before arithmetic operations",
      "Use larger integer types when overflow is possible",
      "Enable compiler overflow detection flags",
    ],
    category: "Memory Safety",
  },
  "CWE-287": {
    name: "Improper Authentication",
    description: "The application does not sufficiently verify that a user is who they claim to be. This allows attackers to gain access to resources or functionality without proper credentials.",
    severity: "Critical",
    owasp: "A07:2021 Identification and Authentication Failures",
    remediation: [
      "Use established authentication frameworks (OAuth 2.0, OpenID Connect)",
      "Implement multi-factor authentication for sensitive operations",
      "Enforce strong password policies and secure credential storage (bcrypt, Argon2)",
      "Use constant-time comparison for authentication tokens",
    ],
    category: "Authentication",
  },
  "CWE-306": {
    name: "Missing Authentication for Critical Function",
    description: "The application does not require authentication before allowing access to a critical function or resource, enabling unauthorized users to perform privileged operations.",
    severity: "Critical",
    owasp: "A07:2021 Identification and Authentication Failures",
    remediation: [
      "Require authentication for all sensitive endpoints and operations",
      "Implement an authentication middleware that covers all routes by default",
      "Use the deny-by-default principle: explicitly whitelist public endpoints",
    ],
    category: "Authentication",
  },
  "CWE-798": {
    name: "Hard-coded Credentials",
    description: "The application contains hard-coded passwords, API keys, or cryptographic keys in source code. Anyone with access to the code or binary can extract these credentials.",
    severity: "Critical",
    owasp: "A07:2021 Identification and Authentication Failures",
    remediation: [
      "Store credentials in environment variables or a secrets manager",
      "Use credential scanning tools (git-secrets, truffleHog) in CI/CD",
      "Rotate all credentials found in source code immediately",
      "Never commit .env files or credentials to version control",
    ],
    category: "Authentication",
  },
  "CWE-862": {
    name: "Missing Authorization",
    description: "The application does not verify that a user is authorized to access a resource or perform an action, allowing users to access data or functionality beyond their intended permissions.",
    severity: "High",
    owasp: "A01:2021 Broken Access Control",
    remediation: [
      "Implement role-based or attribute-based access control (RBAC/ABAC)",
      "Check authorization on every request, not just the UI layer",
      "Deny access by default and explicitly grant permissions",
      "Log authorization failures for security monitoring",
    ],
    category: "Authorization",
  },
  "CWE-200": {
    name: "Information Exposure",
    description: "The application exposes sensitive information to unauthorized actors, whether through error messages, logs, API responses, or other channels. This can reveal system internals that aid further attacks.",
    severity: "Medium",
    owasp: "A01:2021 Broken Access Control",
    remediation: [
      "Return generic error messages to users; log details server-side only",
      "Remove stack traces, debug information, and version numbers from production",
      "Audit API responses to ensure no sensitive data leaks",
      "Classify data by sensitivity and enforce access controls accordingly",
    ],
    category: "Information Disclosure",
  },
  "CWE-352": {
    name: "Cross-Site Request Forgery (CSRF)",
    description: "The application does not verify that a state-changing request was intentionally submitted by the authenticated user, allowing attackers to trick users into performing unintended actions.",
    severity: "High",
    owasp: "A01:2021 Broken Access Control",
    remediation: [
      "Use anti-CSRF tokens on all state-changing requests",
      "Implement SameSite cookie attribute (Strict or Lax)",
      "Verify the Origin and Referer headers on sensitive endpoints",
      "Require re-authentication for high-impact operations",
    ],
    category: "Session Management",
  },
  "CWE-327": {
    name: "Use of Broken Cryptographic Algorithm",
    description: "The application uses a cryptographic algorithm known to be weak (MD5, SHA1, DES, RC4), providing a false sense of security. Attackers may be able to break the cryptography and access protected data.",
    severity: "High",
    owasp: "A02:2021 Cryptographic Failures",
    remediation: [
      "Use modern algorithms: AES-256-GCM for encryption, SHA-256+ for hashing, Argon2 for passwords",
      "Disable deprecated algorithms in all TLS and crypto configurations",
      "Follow NIST or OWASP cryptographic guidelines",
      "Plan for cryptographic agility to enable future algorithm transitions",
    ],
    category: "Cryptography",
  },
  "CWE-311": {
    name: "Missing Encryption of Sensitive Data",
    description: "The application transmits or stores sensitive data in cleartext, making it accessible to anyone who can intercept the communication or access the storage.",
    severity: "High",
    owasp: "A02:2021 Cryptographic Failures",
    remediation: [
      "Encrypt all sensitive data in transit (TLS 1.2+) and at rest (AES-256)",
      "Classify data by sensitivity and apply encryption accordingly",
      "Use HTTPS for all communications, including internal services",
      "Encrypt database columns containing PII, credentials, or financial data",
    ],
    category: "Cryptography",
  },
  "CWE-502": {
    name: "Deserialization of Untrusted Data",
    description: "The application deserializes data from untrusted sources without validation, which can lead to remote code execution, denial of service, or other attacks through crafted serialized objects.",
    severity: "Critical",
    owasp: "A08:2021 Software and Data Integrity Failures",
    remediation: [
      "Avoid deserializing data from untrusted sources entirely",
      "Use safe data formats (JSON) instead of language-native serialization",
      "If deserialization is necessary, validate and integrity-check data beforehand",
      "Implement allowlists for permitted classes during deserialization",
    ],
    category: "Data Integrity",
  },
  "CWE-434": {
    name: "Unrestricted File Upload",
    description: "The application allows file uploads without validating file type, size, or content, enabling attackers to upload malicious files (web shells, malware) that can be executed on the server.",
    severity: "Critical",
    owasp: "A04:2021 Insecure Design",
    remediation: [
      "Validate file type by content inspection (magic bytes), not just extension",
      "Store uploaded files outside the web root in a non-executable location",
      "Generate random filenames and strip metadata from uploads",
      "Enforce strict file size limits and scan uploads with antivirus",
    ],
    category: "Input Validation",
  },
  "CWE-522": {
    name: "Insufficiently Protected Credentials",
    description: "The application stores, transmits, or manages credentials in a way that allows them to be captured or stolen, such as using plain text storage, weak hashing, or unencrypted transmission.",
    severity: "High",
    owasp: "A07:2021 Identification and Authentication Failures",
    remediation: [
      "Hash passwords with Argon2id, bcrypt, or scrypt with appropriate work factors",
      "Never store credentials in plaintext, source code, or configuration files",
      "Transmit credentials only over encrypted channels (TLS 1.2+)",
      "Use a dedicated secrets management system for API keys and tokens",
    ],
    category: "Authentication",
  },
  "CWE-400": {
    name: "Resource Exhaustion (DoS)",
    description: "The application does not properly limit resource consumption, allowing attackers to exhaust CPU, memory, disk, or network resources and deny service to legitimate users.",
    severity: "Medium",
    owasp: "A05:2021 Security Misconfiguration",
    remediation: [
      "Implement rate limiting on all public-facing endpoints",
      "Set timeouts and size limits for all resource-consuming operations",
      "Use circuit breakers and bulkheads to isolate resource pools",
      "Monitor resource usage and alert on anomalous patterns",
    ],
    category: "Availability",
  },
  "CWE-362": {
    name: "Race Condition",
    description: "The application accesses a shared resource without proper synchronization, creating a time-of-check to time-of-use (TOCTOU) window that attackers can exploit to bypass security checks or corrupt data.",
    severity: "Medium",
    owasp: "A04:2021 Insecure Design",
    remediation: [
      "Use mutexes, semaphores, or atomic operations for shared resource access",
      "Implement database-level locking for critical transactions",
      "Design operations to be idempotent where possible",
      "Use file locking or atomic file operations for filesystem resources",
    ],
    category: "Concurrency",
  },
  "CWE-601": {
    name: "Open Redirect",
    description: "The application redirects users to a URL specified by user input without validation, enabling phishing attacks that appear to originate from the trusted application domain.",
    severity: "Medium",
    owasp: "A01:2021 Broken Access Control",
    remediation: [
      "Validate redirect URLs against an allowlist of permitted domains",
      "Use relative URLs for internal redirects instead of absolute URLs",
      "Avoid passing redirect targets as user-controllable parameters",
      "Display a warning page before redirecting to external domains",
    ],
    category: "Input Validation",
  },
  "CWE-1321": {
    name: "Prototype Pollution",
    description: "The application allows modification of JavaScript object prototypes through user input, enabling attackers to inject properties that affect all objects and potentially bypass security checks or achieve code execution.",
    severity: "High",
    owasp: "A03:2021 Injection",
    remediation: [
      "Freeze object prototypes (Object.freeze(Object.prototype))",
      "Use Map instead of plain objects for user-controlled key-value stores",
      "Validate and sanitize keys to block __proto__ and constructor",
      "Use Object.create(null) for dictionary-like objects",
    ],
    category: "Injection",
  },
};

// ─── Category-based fallback remediation ──────────────────────────────────────

const CATEGORY_REMEDIATION: Record<string, string[]> = {
  "Injection": [
    "Validate and sanitize all user input before processing",
    "Use parameterized queries and structured APIs instead of string concatenation",
    "Apply context-specific output encoding",
    "Implement defense-in-depth with WAF rules",
  ],
  "Memory Safety": [
    "Use memory-safe languages or bounds-checked APIs",
    "Enable compiler hardening flags (ASLR, stack canaries, DEP)",
    "Run memory sanitizers (ASan, MSan, Valgrind) during testing",
    "Validate buffer sizes and array indices before all operations",
  ],
  "Authentication": [
    "Use established authentication frameworks and multi-factor authentication",
    "Store credentials securely (bcrypt, Argon2) and never in plaintext",
    "Enforce strong session management with secure cookie attributes",
    "Log and monitor authentication failures",
  ],
  "Authorization": [
    "Implement role-based or attribute-based access control",
    "Apply the principle of least privilege throughout the application",
    "Check authorization on every request at the server side",
    "Deny by default and explicitly grant required permissions",
  ],
  "Cryptography": [
    "Use modern, well-reviewed cryptographic algorithms (AES-256, SHA-256+)",
    "Follow NIST or OWASP cryptographic guidelines",
    "Encrypt sensitive data in transit (TLS 1.2+) and at rest",
    "Use cryptographically secure random number generators",
  ],
  "Information Disclosure": [
    "Return generic error messages to users; log details server-side only",
    "Remove debug information and stack traces from production responses",
    "Classify data by sensitivity and enforce access controls accordingly",
    "Audit API responses and logs to prevent unintended data exposure",
  ],
  "Input Validation": [
    "Validate all input against expected types, ranges, lengths, and formats",
    "Use allowlists over denylists for input validation",
    "Validate on the server side even if client-side validation exists",
    "Reject unexpected input rather than attempting to sanitize it",
  ],
  "Session Management": [
    "Regenerate session IDs after authentication",
    "Set secure cookie attributes: HttpOnly, Secure, SameSite",
    "Implement appropriate session timeouts",
    "Invalidate sessions server-side on logout",
  ],
  "Data Integrity": [
    "Validate and integrity-check data from untrusted sources",
    "Use digital signatures or MACs to detect tampering",
    "Avoid deserializing data from untrusted sources",
    "Implement integrity verification for software updates and dependencies",
  ],
  "Availability": [
    "Implement rate limiting and request size validation",
    "Set timeouts and resource limits for all operations",
    "Use circuit breakers and bulkheads for resource isolation",
    "Monitor resource usage and alert on anomalous patterns",
  ],
  "Concurrency": [
    "Use proper synchronization primitives for shared resource access",
    "Design operations to be idempotent where possible",
    "Implement database-level locking for critical transactions",
    "Test for race conditions with concurrent stress testing",
  ],
  "Configuration": [
    "Follow security hardening guides for all frameworks and platforms",
    "Disable unnecessary features, default accounts, and debug modes in production",
    "Automate security configuration checks in CI/CD pipelines",
    "Regularly review and update security configurations",
  ],
};

const CATEGORY_OWASP: Record<string, string> = {
  "Injection": "A03:2021 Injection",
  "Memory Safety": "A06:2021 Vulnerable and Outdated Components",
  "Authentication": "A07:2021 Identification and Authentication Failures",
  "Authorization": "A01:2021 Broken Access Control",
  "Cryptography": "A02:2021 Cryptographic Failures",
  "Information Disclosure": "A01:2021 Broken Access Control",
  "Input Validation": "A03:2021 Injection",
  "Session Management": "A07:2021 Identification and Authentication Failures",
  "Data Integrity": "A08:2021 Software and Data Integrity Failures",
  "Availability": "A05:2021 Security Misconfiguration",
  "Concurrency": "A04:2021 Insecure Design",
  "Configuration": "A05:2021 Security Misconfiguration",
};

// ─── CWE ID to category mapping ──────────────────────────────────────────────

function classifyCategory(cweId: string): string {
  const num = parseInt(cweId.replace("CWE-", ""), 10);

  const injection = [74, 75, 77, 78, 79, 80, 84, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 98, 99, 113, 114, 116, 117, 134, 140, 150, 155, 158, 170, 172, 611, 776, 917, 943, 1321, 1336];
  if (injection.includes(num)) return "Injection";

  const memory = [118, 119, 120, 121, 122, 123, 124, 125, 126, 129, 130, 131, 190, 191, 193, 195, 197, 415, 416, 457, 476, 680, 681, 682, 763, 787, 788, 789, 805, 822, 823, 824, 825, 908];
  if (memory.includes(num)) return "Memory Safety";

  const auth = [255, 256, 257, 259, 260, 261, 287, 288, 289, 290, 294, 295, 297, 300, 302, 303, 304, 305, 306, 307, 384, 521, 522, 523, 524, 525, 613, 620, 640, 798, 916, 1390, 1391, 1392, 1393];
  if (auth.includes(num)) return "Authentication";

  const authz = [250, 264, 266, 267, 268, 269, 270, 272, 273, 274, 275, 276, 277, 279, 280, 281, 282, 283, 284, 285, 286, 425, 639, 648, 732, 862, 863];
  if (authz.includes(num)) return "Authorization";

  const crypto = [310, 311, 312, 313, 316, 319, 320, 321, 323, 324, 325, 326, 327, 328, 330, 331, 334, 335, 338, 347, 757, 922];
  if (crypto.includes(num)) return "Cryptography";

  const info = [200, 201, 202, 203, 204, 208, 209, 212, 213, 214, 215, 226, 497, 532, 538, 540, 548, 549, 552, 598];
  if (info.includes(num)) return "Information Disclosure";

  const input = [15, 20, 22, 23, 24, 26, 27, 29, 35, 36, 41, 59, 61, 73, 183, 184, 185, 228, 233, 241, 434, 601, 918, 1284, 1285, 1286, 1287, 1288, 1295];
  if (input.includes(num)) return "Input Validation";

  const session = [345, 346, 348, 349, 352, 353, 565, 614, 1004, 1021];
  if (session.includes(num)) return "Session Management";

  const integrity = [354, 356, 357, 358, 470, 471, 472, 494, 501, 502, 506, 829, 830, 912, 913, 915, 924, 940, 942];
  if (integrity.includes(num)) return "Data Integrity";

  const availability = [400, 401, 404, 405, 406, 407, 409, 410, 770, 772, 779, 834, 835, 1333];
  if (availability.includes(num)) return "Availability";

  const concurrency = [362, 366, 367, 377, 378, 379, 662, 667, 833, 911];
  if (concurrency.includes(num)) return "Concurrency";

  return "Configuration";
}

// ─── CWE name lookup (all 349 from labels.json) ──────────────────────────────

const CWE_NAMES: Record<string, string> = {
  "CWE-1004": "Sensitive Cookie Without HttpOnly Flag",
  "CWE-1021": "Clickjacking",
  "CWE-113": "HTTP Response Splitting",
  "CWE-114": "Process Control",
  "CWE-115": "Misinterpretation of Input",
  "CWE-116": "Improper Encoding or Escaping of Output",
  "CWE-117": "Improper Output Neutralization for Logs",
  "CWE-118": "Incorrect Access of Indexable Resource",
  "CWE-1188": "Hard-Coded Network Resource Configuration",
  "CWE-119": "Buffer Overflow",
  "CWE-120": "Buffer Copy without Checking Size",
  "CWE-121": "Stack-based Buffer Overflow",
  "CWE-122": "Heap-based Buffer Overflow",
  "CWE-1220": "Insufficient Granularity of Access Control",
  "CWE-123": "Write-what-where Condition",
  "CWE-1230": "Exposure of Sensitive Information Through Metadata",
  "CWE-1236": "CSV Formula Injection",
  "CWE-124": "Buffer Underwrite",
  "CWE-125": "Out-of-bounds Read",
  "CWE-126": "Buffer Over-read",
  "CWE-1284": "Improper Validation of Specified Quantity in Input",
  "CWE-1285": "Improper Validation of Specified Index",
  "CWE-1286": "Improper Validation of Syntactic Correctness",
  "CWE-1287": "Improper Validation of Specified Type of Input",
  "CWE-1288": "Improper Validation of Consistency within Input",
  "CWE-129": "Improper Validation of Array Index",
  "CWE-1295": "Debug Messages Revealing Unnecessary Information",
  "CWE-130": "Improper Handling of Length Parameter Inconsistency",
  "CWE-131": "Incorrect Calculation of Buffer Size",
  "CWE-1321": "Prototype Pollution",
  "CWE-1333": "ReDoS (Regular Expression Denial of Service)",
  "CWE-1336": "Template Engine Injection",
  "CWE-134": "Externally-Controlled Format String",
  "CWE-1385": "Missing Origin Validation in WebSockets",
  "CWE-1386": "Insecure Windows Junction / Mount Point",
  "CWE-1390": "Weak Authentication",
  "CWE-1391": "Use of Weak Credentials",
  "CWE-1392": "Use of Default Credentials",
  "CWE-1393": "Use of Default Password",
  "CWE-140": "Improper Neutralization of Delimiters",
  "CWE-15": "External Control of System or Configuration Setting",
  "CWE-150": "Improper Neutralization of Escape Sequences",
  "CWE-155": "Improper Neutralization of Wildcards",
  "CWE-158": "Improper Neutralization of Null Byte",
  "CWE-16": "Configuration",
  "CWE-17": "Code",
  "CWE-170": "Improper Null Termination",
  "CWE-172": "Encoding Error",
  "CWE-178": "Improper Handling of Case Sensitivity",
  "CWE-183": "Permissive List of Allowed Inputs",
  "CWE-184": "Incomplete List of Disallowed Inputs",
  "CWE-185": "Incorrect Regular Expression",
  "CWE-189": "Numeric Errors",
  "CWE-19": "Data Processing Errors",
  "CWE-190": "Integer Overflow or Wraparound",
  "CWE-191": "Integer Underflow",
  "CWE-193": "Off-by-one Error",
  "CWE-195": "Signed to Unsigned Conversion Error",
  "CWE-197": "Numeric Truncation Error",
  "CWE-20": "Improper Input Validation",
  "CWE-200": "Exposure of Sensitive Information",
  "CWE-201": "Insertion of Sensitive Information Into Sent Data",
  "CWE-202": "Exposure of Sensitive Information Through Data Queries",
  "CWE-203": "Observable Discrepancy (Side Channel)",
  "CWE-204": "Observable Response Discrepancy",
  "CWE-208": "Observable Timing Discrepancy",
  "CWE-209": "Information Exposure Through Error Messages",
  "CWE-212": "Improper Removal of Sensitive Information",
  "CWE-213": "Exposure Due to Incompatible Policies",
  "CWE-214": "Invocation of Process Using Visible Sensitive Information",
  "CWE-215": "Sensitive Information in Debugging Code",
  "CWE-22": "Path Traversal",
  "CWE-226": "Sensitive Information in Resource Not Removed Before Reuse",
  "CWE-228": "Improper Handling of Syntactically Invalid Structure",
  "CWE-23": "Relative Path Traversal",
  "CWE-233": "Improper Handling of Parameters",
  "CWE-24": "Path Traversal: '../filedir'",
  "CWE-241": "Improper Handling of Unexpected Data Type",
  "CWE-248": "Uncaught Exception",
  "CWE-250": "Execution with Unnecessary Privileges",
  "CWE-252": "Unchecked Return Value",
  "CWE-253": "Incorrect Check of Function Return Value",
  "CWE-254": "Security Features",
  "CWE-255": "Credentials Management Errors",
  "CWE-256": "Plaintext Storage of a Password",
  "CWE-257": "Storing Passwords in a Recoverable Format",
  "CWE-259": "Use of Hard-coded Password",
  "CWE-26": "Path Traversal: '/dir/../filename'",
  "CWE-260": "Password in Configuration File",
  "CWE-261": "Weak Encoding for Password",
  "CWE-264": "Permissions, Privileges, and Access Controls",
  "CWE-266": "Incorrect Privilege Assignment",
  "CWE-267": "Privilege Defined With Unsafe Actions",
  "CWE-268": "Privilege Chaining",
  "CWE-269": "Improper Privilege Management",
  "CWE-27": "Path Traversal: 'dir/../../filename'",
  "CWE-270": "Privilege Context Switching Error",
  "CWE-272": "Least Privilege Violation",
  "CWE-273": "Improper Check for Dropped Privileges",
  "CWE-274": "Improper Handling of Insufficient Privileges",
  "CWE-275": "Permission Issues",
  "CWE-276": "Incorrect Default Permissions",
  "CWE-277": "Insecure Inherited Permissions",
  "CWE-279": "Incorrect Execution-Assigned Permissions",
  "CWE-280": "Improper Handling of Insufficient Permissions",
  "CWE-281": "Improper Preservation of Permissions",
  "CWE-282": "Improper Ownership Management",
  "CWE-283": "Unverified Ownership",
  "CWE-284": "Improper Access Control",
  "CWE-285": "Improper Authorization",
  "CWE-286": "Incorrect User Management",
  "CWE-287": "Improper Authentication",
  "CWE-288": "Authentication Bypass Using Alternate Path",
  "CWE-289": "Authentication Bypass by Alternate Name",
  "CWE-29": "Path Traversal: '\\..\\filename'",
  "CWE-290": "Authentication Bypass by Spoofing",
  "CWE-294": "Authentication Bypass by Capture-replay",
  "CWE-295": "Improper Certificate Validation",
  "CWE-297": "Improper Validation of Certificate with Host Mismatch",
  "CWE-300": "Channel Accessible by Non-Endpoint",
  "CWE-302": "Authentication Bypass by Assumed-Immutable Data",
  "CWE-303": "Incorrect Implementation of Authentication Algorithm",
  "CWE-304": "Missing Critical Step in Authentication",
  "CWE-305": "Authentication Bypass by Primary Weakness",
  "CWE-306": "Missing Authentication for Critical Function",
  "CWE-307": "Improper Restriction of Excessive Authentication Attempts",
  "CWE-310": "Cryptographic Issues",
  "CWE-311": "Missing Encryption of Sensitive Data",
  "CWE-312": "Cleartext Storage of Sensitive Information",
  "CWE-313": "Cleartext Storage in a File or on Disk",
  "CWE-316": "Cleartext Storage in Memory",
  "CWE-319": "Cleartext Transmission of Sensitive Information",
  "CWE-320": "Key Management Errors",
  "CWE-321": "Use of Hard-coded Cryptographic Key",
  "CWE-323": "Reusing a Nonce or Key Pair in Encryption",
  "CWE-324": "Use of a Key Past its Expiration Date",
  "CWE-325": "Missing Cryptographic Step",
  "CWE-326": "Inadequate Encryption Strength",
  "CWE-327": "Use of Broken or Risky Cryptographic Algorithm",
  "CWE-328": "Use of Weak Hash",
  "CWE-330": "Use of Insufficiently Random Values",
  "CWE-331": "Insufficient Entropy",
  "CWE-334": "Small Space of Random Values",
  "CWE-335": "Incorrect Usage of Seeds in PRNG",
  "CWE-338": "Use of Cryptographically Weak PRNG",
  "CWE-345": "Insufficient Verification of Data Authenticity",
  "CWE-346": "Origin Validation Error",
  "CWE-347": "Improper Verification of Cryptographic Signature",
  "CWE-348": "Use of Less Trusted Source",
  "CWE-349": "Acceptance of Extraneous Untrusted Data",
  "CWE-35": "Path Traversal: '.../...//'",
  "CWE-350": "Reliance on Reverse DNS Resolution",
  "CWE-352": "Cross-Site Request Forgery (CSRF)",
  "CWE-353": "Missing Support for Integrity Check",
  "CWE-354": "Improper Validation of Integrity Check Value",
  "CWE-356": "UI Does Not Warn User of Unsafe Actions",
  "CWE-357": "Insufficient UI Warning of Dangerous Operations",
  "CWE-358": "Improperly Implemented Security Check",
  "CWE-359": "Exposure of Private Personal Information",
  "CWE-36": "Absolute Path Traversal",
  "CWE-362": "Race Condition",
  "CWE-366": "Race Condition within a Thread",
  "CWE-367": "TOCTOU Race Condition",
  "CWE-369": "Divide By Zero",
  "CWE-377": "Insecure Temporary File",
  "CWE-378": "Temporary File With Insecure Permissions",
  "CWE-379": "Temporary File in Directory with Insecure Permissions",
  "CWE-384": "Session Fixation",
  "CWE-385": "Covert Timing Channel",
  "CWE-388": "Error Handling",
  "CWE-390": "Detection of Error Without Action",
  "CWE-391": "Unchecked Error Condition",
  "CWE-395": "NullPointerException Catch for NULL Detection",
  "CWE-399": "Resource Management Errors",
  "CWE-400": "Uncontrolled Resource Consumption",
  "CWE-401": "Memory Leak",
  "CWE-402": "Transmission of Private Resources into New Sphere",
  "CWE-404": "Improper Resource Shutdown or Release",
  "CWE-405": "Asymmetric Resource Consumption (Amplification)",
  "CWE-406": "Insufficient Control of Network Message Volume",
  "CWE-407": "Inefficient Algorithmic Complexity",
  "CWE-409": "Improper Handling of Highly Compressed Data",
  "CWE-41": "Improper Resolution of Path Equivalence",
  "CWE-410": "Insufficient Resource Pool",
  "CWE-415": "Double Free",
  "CWE-416": "Use After Free",
  "CWE-417": "Communication Channel Errors",
  "CWE-420": "Unprotected Alternate Channel",
  "CWE-424": "Improper Protection of Alternate Path",
  "CWE-425": "Direct Request (Forced Browsing)",
  "CWE-426": "Untrusted Search Path",
  "CWE-427": "Uncontrolled Search Path Element",
  "CWE-428": "Unquoted Search Path or Element",
  "CWE-434": "Unrestricted Upload of File with Dangerous Type",
  "CWE-436": "Interpretation Conflict",
  "CWE-440": "Expected Behavior Violation",
  "CWE-441": "Confused Deputy",
  "CWE-444": "HTTP Request/Response Smuggling",
  "CWE-449": "UI Performs Wrong Action",
  "CWE-451": "UI Misrepresentation of Critical Information",
  "CWE-453": "Insecure Default Variable Initialization",
  "CWE-457": "Use of Uninitialized Variable",
  "CWE-459": "Incomplete Cleanup",
  "CWE-460": "Improper Cleanup on Thrown Exception",
  "CWE-470": "Unsafe Reflection",
  "CWE-471": "Modification of Assumed-Immutable Data",
  "CWE-472": "External Control of Web Parameter",
  "CWE-475": "Undefined Behavior for Input to API",
  "CWE-476": "NULL Pointer Dereference",
  "CWE-488": "Data Exposure to Wrong Session",
  "CWE-489": "Active Debug Code",
  "CWE-494": "Download of Code Without Integrity Check",
  "CWE-497": "Exposure of Sensitive System Information",
  "CWE-501": "Trust Boundary Violation",
  "CWE-502": "Deserialization of Untrusted Data",
  "CWE-506": "Embedded Malicious Code",
  "CWE-521": "Weak Password Requirements",
  "CWE-522": "Insufficiently Protected Credentials",
  "CWE-523": "Unprotected Transport of Credentials",
  "CWE-524": "Cache Containing Sensitive Information",
  "CWE-525": "Browser Cache Containing Sensitive Information",
  "CWE-532": "Sensitive Information in Log File",
  "CWE-538": "Sensitive Information in Externally-Accessible File",
  "CWE-540": "Sensitive Information in Source Code",
  "CWE-548": "Information Exposure Through Directory Listing",
  "CWE-549": "Missing Password Field Masking",
  "CWE-552": "Files Accessible to External Parties",
  "CWE-565": "Reliance on Cookies without Integrity Checking",
  "CWE-59": "Improper Link Resolution (Link Following)",
  "CWE-591": "Sensitive Data in Improperly Locked Memory",
  "CWE-592": "Authentication Bypass Issues",
  "CWE-598": "GET Request with Sensitive Query Strings",
  "CWE-601": "Open Redirect",
  "CWE-602": "Client-Side Enforcement of Server-Side Security",
  "CWE-603": "Use of Client-Side Authentication",
  "CWE-606": "Unchecked Input for Loop Condition",
  "CWE-61": "UNIX Symlink Following",
  "CWE-610": "Externally Controlled Reference to Another Sphere",
  "CWE-611": "XML External Entity (XXE)",
  "CWE-613": "Insufficient Session Expiration",
  "CWE-614": "Sensitive Cookie Without 'Secure' Attribute",
  "CWE-617": "Reachable Assertion",
  "CWE-620": "Unverified Password Change",
  "CWE-639": "IDOR (Insecure Direct Object Reference)",
  "CWE-640": "Weak Password Recovery Mechanism",
  "CWE-642": "External Control of Critical State Data",
  "CWE-644": "Improper Neutralization of HTTP Headers",
  "CWE-648": "Incorrect Use of Privileged APIs",
  "CWE-653": "Improper Isolation or Compartmentalization",
  "CWE-657": "Violation of Secure Design Principles",
  "CWE-662": "Improper Synchronization",
  "CWE-664": "Improper Control of Resource Lifetime",
  "CWE-665": "Improper Initialization",
  "CWE-667": "Improper Locking",
  "CWE-668": "Exposure of Resource to Wrong Sphere",
  "CWE-669": "Incorrect Resource Transfer Between Spheres",
  "CWE-670": "Always-Incorrect Control Flow",
  "CWE-672": "Operation on Resource after Expiration or Release",
  "CWE-674": "Uncontrolled Recursion",
  "CWE-680": "Integer Overflow to Buffer Overflow",
  "CWE-681": "Incorrect Conversion between Numeric Types",
  "CWE-682": "Incorrect Calculation",
  "CWE-684": "Incorrect Provision of Specified Functionality",
  "CWE-690": "Unchecked Return Value to NULL Pointer Dereference",
  "CWE-691": "Insufficient Control Flow Management",
  "CWE-693": "Protection Mechanism Failure",
  "CWE-696": "Incorrect Behavior Order",
  "CWE-697": "Incorrect Comparison",
  "CWE-701": "Weaknesses Introduced During Design",
  "CWE-703": "Improper Check of Exceptional Conditions",
  "CWE-704": "Incorrect Type Conversion or Cast",
  "CWE-706": "Use of Incorrectly-Resolved Name or Reference",
  "CWE-707": "Improper Neutralization",
  "CWE-708": "Incorrect Ownership Assignment",
  "CWE-73": "External Control of File Name or Path",
  "CWE-732": "Incorrect Permission Assignment for Critical Resource",
  "CWE-74": "Injection",
  "CWE-749": "Exposed Dangerous Method or Function",
  "CWE-75": "Failure to Sanitize Special Elements",
  "CWE-754": "Improper Check for Unusual Conditions",
  "CWE-755": "Improper Handling of Exceptional Conditions",
  "CWE-757": "Algorithm Downgrade",
  "CWE-763": "Release of Invalid Pointer or Reference",
  "CWE-77": "Command Injection",
  "CWE-770": "Allocation Without Limits or Throttling",
  "CWE-772": "Missing Release of Resource",
  "CWE-776": "XML Entity Expansion (Billion Laughs)",
  "CWE-778": "Insufficient Logging",
  "CWE-779": "Logging of Excessive Data",
  "CWE-78": "OS Command Injection",
  "CWE-782": "Exposed IOCTL with Insufficient Access Control",
  "CWE-783": "Operator Precedence Logic Error",
  "CWE-787": "Out-of-bounds Write",
  "CWE-788": "Access of Memory After End of Buffer",
  "CWE-789": "Memory Allocation with Excessive Size",
  "CWE-79": "Cross-site Scripting (XSS)",
  "CWE-790": "Improper Filtering of Special Elements",
  "CWE-791": "Incomplete Filtering of Special Elements",
  "CWE-798": "Hard-coded Credentials",
  "CWE-799": "Improper Control of Interaction Frequency",
  "CWE-80": "Basic XSS",
  "CWE-805": "Buffer Access with Incorrect Length",
  "CWE-807": "Reliance on Untrusted Inputs in Security Decision",
  "CWE-821": "Incorrect Synchronization",
  "CWE-822": "Untrusted Pointer Dereference",
  "CWE-823": "Out-of-range Pointer Offset",
  "CWE-824": "Access of Uninitialized Pointer",
  "CWE-825": "Expired Pointer Dereference",
  "CWE-829": "Inclusion from Untrusted Control Sphere",
  "CWE-830": "Inclusion of Web Functionality from Untrusted Source",
  "CWE-833": "Deadlock",
  "CWE-834": "Excessive Iteration",
  "CWE-835": "Infinite Loop",
  "CWE-838": "Inappropriate Encoding for Output",
  "CWE-84": "Improper Neutralization of Encoded URI Schemes",
  "CWE-840": "Business Logic Errors",
  "CWE-841": "Improper Enforcement of Behavioral Workflow",
  "CWE-843": "Type Confusion",
  "CWE-862": "Missing Authorization",
  "CWE-863": "Incorrect Authorization",
  "CWE-87": "Alternate XSS Syntax",
  "CWE-88": "Improper Neutralization of Argument Delimiters",
  "CWE-89": "SQL Injection",
  "CWE-90": "LDAP Injection",
  "CWE-908": "Use of Uninitialized Resource",
  "CWE-909": "Missing Initialization of Resource",
  "CWE-91": "XML Injection",
  "CWE-911": "Improper Update of Reference Count",
  "CWE-912": "Hidden Functionality",
  "CWE-913": "Improper Control of Dynamically-Managed Code Resources",
  "CWE-915": "Improperly Controlled Modification of Object Attributes",
  "CWE-916": "Password Hash With Insufficient Computational Effort",
  "CWE-917": "Expression Language Injection",
  "CWE-918": "Server-Side Request Forgery (SSRF)",
  "CWE-92": "Improper Sanitization of Custom Special Characters",
  "CWE-922": "Insecure Storage of Sensitive Information",
  "CWE-923": "Improper Restriction of Communication Channel",
  "CWE-924": "Improper Enforcement of Message Integrity",
  "CWE-926": "Improper Export of Android Application Components",
  "CWE-927": "Use of Implicit Intent for Sensitive Communication",
  "CWE-93": "CRLF Injection",
  "CWE-94": "Code Injection",
  "CWE-940": "Improper Verification of Source of Communication Channel",
  "CWE-942": "Permissive Cross-domain Policy",
  "CWE-943": "Improper Neutralization in Data Query Logic",
  "CWE-95": "Eval Injection",
  "CWE-96": "Static Code Injection",
  "CWE-98": "PHP Remote File Inclusion",
  "CWE-99": "Resource Injection",
};

// ─── Severity heuristic ───────────────────────────────────────────────────────

function inferSeverity(cweId: string, category: string): "Critical" | "High" | "Medium" | "Low" {
  const num = parseInt(cweId.replace("CWE-", ""), 10);

  // Critical: RCE-capable, injection, auth bypass, deserialization
  const critical = [77, 78, 89, 94, 95, 96, 119, 120, 121, 122, 123, 287, 306, 416, 434, 502, 787, 798, 917];
  if (critical.includes(num)) return "Critical";

  // High: most injection/memory/auth/crypto
  if (["Injection", "Memory Safety", "Authentication", "Cryptography"].includes(category)) return "High";

  // Medium: info disclosure, session, availability, config
  return "Medium";
}

// ─── Public API ───────────────────────────────────────────────────────────────

export function getCWEInfo(cweId: string): CWEEntry {
  // Check curated entries first
  const curated = CURATED[cweId];
  if (curated) {
    return { id: cweId, ...curated };
  }

  // Auto-generate from lookup tables
  const category = classifyCategory(cweId);
  const name = CWE_NAMES[cweId] || cweId;
  const severity = inferSeverity(cweId, category);
  const owasp = CATEGORY_OWASP[category] || "A05:2021 Security Misconfiguration";
  const remediation = CATEGORY_REMEDIATION[category] || CATEGORY_REMEDIATION["Configuration"];

  const description = `This weakness involves ${name.toLowerCase()}. It can allow attackers to compromise the ${category.toLowerCase()} properties of the affected system, potentially leading to unauthorized access, data exposure, or system instability.`;

  return { id: cweId, name, description, severity, owasp, remediation, category };
}

/** Severity color for UI rendering */
export function getSeverityColor(severity: string): string {
  switch (severity) {
    case "Critical": return "#ef4444";
    case "High": return "#f97316";
    case "Medium": return "#eab308";
    case "Low": return "#22c55e";
    default: return "#6b7280";
  }
}

/** Severity badge CSS classes for UI rendering */
export function getSeverityBg(severity: string): string {
  switch (severity) {
    case "Critical": return "bg-red-500/20 text-red-400 border-red-500/30";
    case "High": return "bg-orange-500/20 text-orange-400 border-orange-500/30";
    case "Medium": return "bg-yellow-500/20 text-yellow-400 border-yellow-500/30";
    case "Low": return "bg-green-500/20 text-green-400 border-green-500/30";
    default: return "bg-gray-500/20 text-gray-400 border-gray-500/30";
  }
}
