import { describe, it, expect } from "vitest";
import { validateDataQuality } from "@/lib/engine/data-quality";

describe("Data Quality Validator", () => {
  it("detects empty dataset", () => {
    const report = validateDataQuality([]);
    expect(report.score).toBe(0);
    expect(report.grade).toBe("F");
  });

  it("scores a good dataset highly", () => {
    const data = [];
    for (let i = 0; i < 20; i++) {
      data.push({ text: `example ${i} for greeting`, intent: "greet" });
      data.push({ text: `example ${i} for farewell`, intent: "farewell" });
    }
    const report = validateDataQuality(data);
    expect(report.score).toBeGreaterThan(50);
    expect(report.stats.totalExamples).toBe(40);
    expect(report.stats.totalIntents).toBe(2);
  });

  it("detects exact duplicates", () => {
    const data = [
      { text: "hello there", intent: "greet" },
      { text: "hello there", intent: "greet" },
      { text: "goodbye", intent: "farewell" },
      { text: "goodbye", intent: "farewell" },
      { text: "see ya", intent: "farewell" },
    ];
    const report = validateDataQuality(data);
    expect(report.stats.duplicateCount).toBe(2);
    expect(report.issues.some((i) => i.category === "duplicate")).toBe(true);
  });

  it("detects class imbalance", () => {
    const data = [
      ...Array.from({ length: 30 }, (_, i) => ({ text: `greet example ${i}`, intent: "greet" })),
      { text: "bye", intent: "farewell" },
      { text: "goodbye", intent: "farewell" },
    ];
    const report = validateDataQuality(data);
    expect(report.issues.some((i) => i.category === "class_imbalance")).toBe(true);
  });

  it("detects short examples", () => {
    const data = [
      { text: "hi", intent: "greet" },
      { text: "hello there friend", intent: "greet" },
      { text: "x", intent: "farewell" },
      { text: "goodbye see you later", intent: "farewell" },
      { text: "bye for now", intent: "farewell" },
    ];
    const report = validateDataQuality(data);
    expect(report.stats.shortExamples).toBeGreaterThan(0);
  });

  it("detects empty examples", () => {
    const data = [
      { text: "", intent: "greet" },
      { text: "hello", intent: "greet" },
      { text: "hi there", intent: "greet" },
      { text: "bye", intent: "farewell" },
      { text: "goodbye", intent: "farewell" },
    ];
    const report = validateDataQuality(data);
    expect(report.issues.some((i) => i.category === "empty_example")).toBe(true);
  });

  it("provides per-intent breakdown", () => {
    const data = [
      ...Array.from({ length: 10 }, (_, i) => ({ text: `greeting ${i}`, intent: "greet" })),
      ...Array.from({ length: 10 }, (_, i) => ({ text: `farewell ${i}`, intent: "farewell" })),
    ];
    const report = validateDataQuality(data);
    expect(report.perIntent).toHaveLength(2);
    expect(report.perIntent[0].exampleCount).toBe(10);
  });

  it("assigns letter grade", () => {
    const data = Array.from({ length: 40 }, (_, i) => ({
      text: `unique example number ${i} with some words`,
      intent: i < 20 ? "greet" : "farewell",
    }));
    const report = validateDataQuality(data);
    expect(["A", "B", "C", "D", "F"]).toContain(report.grade);
  });
});
