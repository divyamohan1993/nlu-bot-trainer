# Security Policy

## Supported Versions

| Version | Supported |
|---------|-----------|
| 1.x     | Yes       |

## Architecture Security Notes

Sentio runs **entirely client-side**. All model training, inference, and data storage happen in the browser's `localStorage`. No data leaves the user's machine during normal operation.

**What this means:**

- No server-side attack surface for the core NLU engine
- No API keys, tokens, or credentials stored or transmitted
- Training data lives in `localStorage` — clearing browser data removes it
- The optional GCloud training pipeline (`training/`) operates on user-provisioned infrastructure

## Reporting a Vulnerability

If you discover a security vulnerability, please report it responsibly.

**Do NOT open a public GitHub issue for security vulnerabilities.**

### How to Report

1. **Email:** Send details to [divyamohan1993@gmail.com](mailto:divyamohan1993@gmail.com)
2. **Subject line:** `[SECURITY] nlu-bot-trainer — <brief description>`
3. **Include:**
   - Description of the vulnerability
   - Steps to reproduce
   - Potential impact assessment
   - Suggested fix (if you have one)

### What to Expect

- **Acknowledgment** within 48 hours
- **Assessment** within 7 days
- **Fix timeline** communicated after assessment — critical issues patched within 72 hours

### Scope

The following are in scope:

| Area | Examples |
|------|----------|
| **Client-side XSS** | Injection through training data labels, intent names, or test inputs that execute in the UI |
| **localStorage poisoning** | Crafted data that corrupts model state or causes code execution on load |
| **Export injection** | Malicious content in exported files (Rasa YAML, Dialogflow JSON, etc.) that could exploit downstream systems |
| **GCloud pipeline** | Command injection in `training/*.sh` scripts, credential exposure |
| **Dependency vulnerabilities** | CVEs in Next.js, React, or other dependencies |

The following are **out of scope**:

- Vulnerabilities requiring physical access to the user's machine
- Issues in browsers themselves (report to browser vendors)
- Social engineering attacks
- Denial of service against the local dev server

## Security Best Practices for Users

1. **Training data**: Do not paste sensitive customer data into the trainer. Use anonymized or synthetic examples.
2. **Exports**: Review exported model files before uploading to third-party platforms (Dialogflow, Lex, etc.).
3. **GCloud pipeline**: Use dedicated service accounts with minimal permissions. Spot VMs are ephemeral by design — no persistent attack surface.
4. **Dependencies**: Run `npm audit` regularly. The project enforces strict dependency policies.
