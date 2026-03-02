import type { DependencyFileType, ParsedDependency } from "./types";

// ─── File type detection ────────────────────────────────────────────────────

export function detectFileType(content: string): DependencyFileType | null {
  // package.json: valid JSON with dependency keys
  try {
    const json = JSON.parse(content);
    if (json.dependencies || json.devDependencies) return "package.json";
  } catch {
    /* not JSON */
  }

  // pom.xml: contains <dependencies> tag
  if (/<dependencies>/.test(content)) return "pom.xml";

  // requirements.txt: lines matching name==version or name>=version
  const reqLine = /^[a-zA-Z0-9_-]+\s*(==|>=|~=|<=)\s*[\d.]+/m;
  if (reqLine.test(content)) return "requirements.txt";

  return null;
}

// ─── Dependency parsing ─────────────────────────────────────────────────────

function stripPrefix(version: string): string {
  return version.replace(/^[\^~>=<]+\s*/, "").trim();
}

function parsePackageJson(content: string): ParsedDependency[] {
  const json = JSON.parse(content);
  const results: ParsedDependency[] = [];
  for (const section of ["dependencies", "devDependencies"] as const) {
    const deps = json[section] as Record<string, string> | undefined;
    if (!deps) continue;
    for (const [name, raw] of Object.entries(deps)) {
      results.push({ name, version: stripPrefix(raw), ecosystem: "npm" });
    }
  }
  return results;
}

function parseRequirementsTxt(content: string): ParsedDependency[] {
  const results: ParsedDependency[] = [];
  const lineRe = /^([a-zA-Z0-9_.-]+)\s*(==|>=|~=|<=)\s*([\d.*]+)/;
  for (const raw of content.split("\n")) {
    const line = raw.trim();
    if (!line || line.startsWith("#") || line.startsWith("-r") || line.startsWith("-e")) continue;
    const m = line.match(lineRe);
    if (m) results.push({ name: m[1], version: m[3], ecosystem: "PyPI" });
  }
  return results;
}

function parsePomXml(content: string): ParsedDependency[] {
  const results: ParsedDependency[] = [];
  const depRe = /<dependency>[\s\S]*?<groupId>([^<]+)<\/groupId>[\s\S]*?<artifactId>([^<]+)<\/artifactId>[\s\S]*?<version>([^<]+)<\/version>/g;
  let m: RegExpExecArray | null;
  while ((m = depRe.exec(content)) !== null) {
    results.push({ name: `${m[1]}:${m[2]}`, version: m[3], ecosystem: "Maven" });
  }
  return results;
}

export function parseDependencies(content: string, fileType: DependencyFileType): ParsedDependency[] {
  switch (fileType) {
    case "package.json":
      return parsePackageJson(content);
    case "requirements.txt":
      return parseRequirementsTxt(content);
    case "pom.xml":
      return parsePomXml(content);
  }
}

// ─── Sample dependency files ────────────────────────────────────────────────

export const DEPENDENCY_SAMPLES: Record<string, { label: string; content: string }> = {
  npm: {
    label: "package.json (Node.js)",
    content: JSON.stringify(
      {
        name: "demo-app",
        version: "1.0.0",
        dependencies: { lodash: "^4.17.20", express: "^4.17.1", minimist: "^1.2.5", axios: "^0.21.1" },
        devDependencies: { "node-fetch": "^2.6.1" },
      },
      null,
      2,
    ),
  },
  pypi: {
    label: "requirements.txt (Python)",
    content: [
      "# Python dependencies",
      "django==3.1",
      "requests==2.25.0",
      "flask==1.1.2",
      "urllib3==1.26.4",
      "jinja2==2.11.3",
    ].join("\n"),
  },
  maven: {
    label: "pom.xml (Java)",
    content: [
      "<dependencies>",
      "  <dependency>",
      "    <groupId>org.apache.logging.log4j</groupId>",
      "    <artifactId>log4j-core</artifactId>",
      "    <version>2.14.1</version>",
      "  </dependency>",
      "  <dependency>",
      "    <groupId>org.springframework</groupId>",
      "    <artifactId>spring-core</artifactId>",
      "    <version>5.3.0</version>",
      "  </dependency>",
      "  <dependency>",
      "    <groupId>commons-collections</groupId>",
      "    <artifactId>commons-collections</artifactId>",
      "    <version>3.2.1</version>",
      "  </dependency>",
      "</dependencies>",
    ].join("\n"),
  },
};
