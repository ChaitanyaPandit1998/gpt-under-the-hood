# Reusable Story characters

Only use a reusable head for a fictional human who fits the brief. Do not add
people to make use of the assets, silently replace requested appearances, or add
heads to a factual diagram. For recurring characters, pick the same asset ID
when the cast description specifies the same appearance.

Use a single item per head: `{kind:"story-head",asset:"short-hair",x:250,y:210,scale:1.5,mirror:false}`.
The anchor is near the middle of the face: use x and y to align the neck above
the shoulders. The two head assets face inward by default: short-hair faces right,
long-hair faces left. Use mirror only to reverse the direction.

| Asset | Design | Hand skin | Relative bounds at scale 1 (x, y) |
| --- | --- | --- | --- |
| short-hair | Short dark hair, warm medium skin | #C88565 | -32 to 29, -50 to 41 |
| long-hair | Long dark hair, light warm skin | #F1AB8B | -25 to 49, -48 to 52 |

Plan the torso, hands, and props separately. Draw the torso first; place the
head so the skin of its neck overlaps the collar, then place hands, foreground
props and necessary labels as the scene demands. Inspect the scaled footprint
and keep the head away from labels and unrelated shapes. Do not draw a second
face, head, or neck on top of the asset. Scale is uniform from 0.75 to 3; at
most four heads may be placed, costing 9 or 10 paths each toward the same
96-item budget. No assets may be supplied by file or URL.
