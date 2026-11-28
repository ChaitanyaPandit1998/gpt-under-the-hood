---
name: compose-vectorcraft-visual
description: Translate a teaching visual brief into a bounded scene for the application's VectorCraft adapter. Use for composition with supported text, rectangle and path primitives, not for arbitrary SVG, renderer commands or raster image generation.
metadata:
  version: "1"
---

# Compose a VectorCraft Teaching Visual

Translate the supplied brief into the requested scene schema, preserving its
teaching goal, evidence roles, exact relationships and qualifications. The
application supplies the canvas, item limits and supported schema. Return only
scene data; do not execute renderer commands, output code, add file paths or
claim to have rendered or inspected the result.

## Design the Picture, Not a Text Report

Use one dominant visual relationship and a clear reading order. For an
illustration, build recognizable objects or an explanatory scene from layered
silhouettes, polygons and short paths. For a comparison, align corresponding
elements and emphasize the supported difference. For a quantitative figure,
preserve supplied scales, units and domains. Use a flow only when order or
branching is actually part of the claim; do not default to stacked text boxes.

Use rectangles for panels or fills and path points for lines, arrows, curved
contours approximated with points, or closed polygons. Work within the schema:
do not request images, embedded fonts, SVG filters, arbitrary gradients or
unsupported primitives. When the composition exceeds that vocabulary, simplify
the artwork while retaining its meaning rather than fabricating new fields.

Keep decorative shapes distinct from evidence. Do not invent values, plotted
positions, observations or causal arrows. A narrative illustration depicts an
analogy, not a literal technical event. Preserve required visible qualifiers
and the boundary between analogy and fact. If the brief's composition conflicts
with its factual guardrails, prioritize accuracy and avoid drawing the conflict
as an established fact.

For a character scene, make the action clear through the pose and the prop
contact. Use one connected hand silhouette per visible wrist, aligned with
the sleeve cuff. Layer the thumb and fingers around the prop as needed without
creating a second-looking hand or hiding a teaching label. Keep the wrist and
hand proportionate to the pose; use a simple contour and a short crease
instead of drawing every finger. A brief that describes an approved face is
not an exact vector asset. Do not claim exact character reuse when the host
has not supplied a way to place that asset.

## Layout and Labels

Reserve most of the canvas for the visual itself. Use intentional spacing,
contrast and a restrained palette; pair color with shape, position or a label
when color encodes meaning. Keep backgrounds behind geometry and foreground
text after the geometry so shapes cannot paint over essential labels.

Reserve separate empty areas for complete label widths, not just their anchor
points. For narrow regions, use short leaders from a dedicated margin without
crossing labels. Arrows must connect the intended objects and have clear
direction. Do not treat successful schema validation as proof of correct
geometry, legibility or factual alignment.

The application displays the caption below the PNG. Never copy the caption,
alt text, narrative paragraphs or long explanations into the drawing, even if
the composition asks for them there. Use at most 60 characters per label and
40 words total for short teaching labels and essential qualifiers. Do not split
paragraphs into many short labels to evade the limit. Keep exact required
notation and make room for it instead of shrinking it until unreadable.

Compact scene encodings are available only when the host enables their schema.
In that mode, the host also supplies [compact guidance](references/compact-scenes.md).
Otherwise return ordinary items only. An invalid plan may receive one structural
repair request; simplify within the same factual constraints, not by dropping
important qualifications or inventing replacement evidence.
