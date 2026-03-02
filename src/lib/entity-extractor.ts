import type { Entity, EntityAnnotation } from "@/types";

/** Extract entities from text using dictionary matching */
export function extractEntities(
  text: string,
  entityDefinitions: Entity[]
): EntityAnnotation[] {
  const annotations: EntityAnnotation[] = [];
  const lowerText = text.toLowerCase();

  for (const entityDef of entityDefinitions) {
    for (const value of entityDef.values) {
      const lowerValue = value.toLowerCase();
      let searchFrom = 0;

      while (searchFrom < lowerText.length) {
        const idx = lowerText.indexOf(lowerValue, searchFrom);
        if (idx === -1) break;

        // Check word boundaries
        const before = idx === 0 || /\W/.test(lowerText[idx - 1]);
        const after =
          idx + lowerValue.length >= lowerText.length ||
          /\W/.test(lowerText[idx + lowerValue.length]);

        if (before && after) {
          // Check for overlapping annotations
          const overlaps = annotations.some(
            (a) => idx < a.end && idx + lowerValue.length > a.start
          );

          if (!overlaps) {
            annotations.push({
              start: idx,
              end: idx + lowerValue.length,
              value: text.slice(idx, idx + lowerValue.length),
              entity: entityDef.name,
            });
          }
        }

        searchFrom = idx + 1;
      }
    }
  }

  // Sort by start position
  annotations.sort((a, b) => a.start - b.start);
  return annotations;
}
