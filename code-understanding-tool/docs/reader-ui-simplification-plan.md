# Reader UI Simplification Plan

Status: Proposed; documentation only, not implemented.

Cross-plan delivery order: [Prioritized Roadmap](prioritized-roadmap.md).
Implement first with lightweight checks; defer full end-to-end acceptance
execution until the integrated testing phase defined in that roadmap.

## 1. Reduce Clutter and Verbosity

- Make the chapter explanation and accepted inline illustration the primary
  content. Keep a compact chapter toolbar for Refine, Illustrate, Read aloud,
  and Export; show only actions available for the current scope.
- Open one tool panel at a time without clearing another tool's draft or
  cancelling active work silently. Keep generation status visible when a
  running tool is collapsed.
- Replace the long expanded visual-builder stack with progressive steps:
  What to teach, Look and feel, then Generate. Keep sensible defaults and a
  direct generation path for users who do not need customization.
- Keep prompt editing, additional context, source inclusion, full brief,
  drawing diagnostics, and model review behind clearly labelled expandable
  controls. Do not nest every optional panel inside an always-expanded form.
- Replace repeated instructional paragraphs with short helper text. Retain
  essential privacy disclosures, explicit review consent, uncertainty labels,
  and factual warnings at the relevant action or image.
- Show a concise review status with expandable findings instead of displaying
  lengthy review commentary above the chapter by default.
- Reuse existing themed buttons, cards, spacing, and typography. Preserve
  visible labels, keyboard access, focus management, and mobile layouts.

## 2. Chapters as the Only Reading View

- Remove the Chapters / Full explanation switch and its redundant introductory
  toolbar. Chapters become the default and only on-screen reading layout;
  do not replace the switch with a single unnecessary tab button.
- Retain the sticky chapter outline, chapter title and position, previous/next
  navigation, and saved reading position. Use a compact chapter picker on mobile.
- Keep Direct / Story as chapter presentation choices; they are not separate
  full-document reading modes.
- Keep every explanation section accessible through the chapter structure.
  Check short explanations, code, documentation, and PR results for missing or
  duplicated content before removing the full-view rendering branch.
- Preserve full-document export and read-aloud scope as tool capabilities,
  not reader tabs. Clearly label chapter versus entire-study actions.
- Continue accepting older saved/imported studies with a full-view preference;
  restore them into chapters without losing variants, stories, or images.
  Do not break the portable format just to remove an obsolete UI preference.
- Audit whole-explanation images and saved state so assets previously shown
  only in full view remain accessible in the chapter-only experience without
  falsely assigning them to an unrelated chapter.

## Implementation Order

1. Audit chapter completeness, saved-state compatibility, and full-view assets.
2. Remove the reading-mode switch and normalize restored reading state.
3. Simplify the chapter toolbar and introduce progressive tool disclosure.
4. Shorten helper copy and consolidate visual generation steps without changing
   generation contracts or discarding existing drafts.
5. Update component/browser tests and DEVLOG.md with actual validation results.

## Acceptance Checks

- Only chapter navigation is shown; no Full explanation reader toggle remains.
- The initial chapter view prioritizes readable content rather than open forms.
- Existing stories, images, refinements, exports, and read-aloud remain usable.
- Older saved and portable studies open without content or asset loss.
- Tool drafts survive switching panels; cancel/retry and active status remain clear.
- Privacy and review consent are preserved; no new model call happens merely
  because a panel is opened or closed.
- Desktop/mobile and keyboard tests cover chapter navigation, tools, restore,
  image generation, and export. Use the external pilot for manual review only;
  keep production logic and committed fixtures topic-independent.

Next step: implement chapter-only navigation and compatibility handling first,
then simplify the visual-builder presentation.
