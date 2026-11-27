# Compact Scene Encoding

Use compact items only when the selected response schema supports them.

- A repeat contains one rectangle or path in item, count copies total (2 to 24),
  and dx, dy between copies. Copy zero uses the original geometry. Use for evenly
  spaced cells, grid lines or ticks; never nest repeats.
- A text-run places literal labels in order at (x + index*dx, y + index*dy),
  sharing size, color and align. Use ordinary items for irregular positions.
- Compact items expand in place, retaining back-to-front ordering. Keep the
  expanded result at or under 96 ordinary items and all geometry in bounds.
- Repetition is an encoding convenience, not evidence of regularity. Do not
  introduce evenly spaced values when the supplied material does not support
  them. Retain the required labels and qualifications when simplifying.
