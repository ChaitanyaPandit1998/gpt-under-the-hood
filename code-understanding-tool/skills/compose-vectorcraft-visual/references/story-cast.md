# Reusable Story characters

Only use a reusable head for a fictional human who fits the brief. Do not add
people to make use of the assets, silently replace requested appearances, or add
heads to a factual diagram. For recurring characters, pick the same asset ID
when the cast description specifies the same appearance.

Draw each torso as a solid rectangle or a closed filled path. Then place one
head with `{kind:"story-head",asset:"short-hair",torsoIndex:0,collarX:250,scale:1.5,mirror:false}`.
`torsoIndex` is the zero-based index of the earlier torso in the original
items array, before any expansion. Count all items, including backgrounds
and labels. `collarX` is the horizontal center of the shirt opening, well
inside both shoulders. The renderer calculates the vertical position from
the top of that torso and connects the neck to the shirt. Do not supply an
x or y for the face or reference a backdrop, sleeve, or prop as the torso.
Short-hair faces right; long-hair faces left. Mirror only to reverse direction.
When the request includes locked cast data, the allowed asset IDs are the
listed slots instead of the face IDs below. Use those slots when depicting
their corresponding roles. An absent role does not need a placement.

| Asset | Design | Hand skin | Approximate reach above the collar at scale 1 |
| --- | --- | --- | --- |
| short-hair | Short dark hair, warm medium skin | #C88565 | 85 px |
| long-hair | Long dark hair, light warm skin | #F1AB8B | 80 px |

Plan the torso, hands, and props separately. Keep the top edge around the
collar solid and wide enough to support the neck. Place hands, foreground
props and necessary labels as the scene demands. Allow for the scaled footprint
and keep the head away from labels and unrelated shapes. Do not draw a second
face, head, or neck on top of the asset. Scale is uniform from 0.75 to 3; at
most four heads may be placed, costing 9 or 10 paths each toward the same
96-item budget. No assets may be supplied by file or URL.
