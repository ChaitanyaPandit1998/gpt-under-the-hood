# PR Chapter Latency Benchmark

Append new batches here. Record the PR, chapter scope, input size, settings,
run order, stage timings, and review outcome so runs can be compared.

## 2026-11-26: Text-only Overview baseline

Six live requests to the local explanation API through Codex CLI. Settings
were Overview depth, beginner audience, plain tone, automatic focus, no
extra learner question, and diagrams disabled. Requests ran sequentially;
the second round reversed the input order. GitHub import time is excluded.

| Input | Scope | Source characters |
| --- | --- | ---: |
| Small | [Vite #16471](https://github.com/vitejs/vite/pull/16471), full-reload handler and helper excerpt | 2,405 |
| Medium | [FastAPI #12199](https://github.com/fastapi/fastapi/pull/12199), Implementation: fastapi - part 1 | 11,672 |
| Large | [Vite #16471](https://github.com/vitejs/vite/pull/16471), Implementation: packages - part 12 | 37,965 |

Times are seconds. Stage durations are estimates derived from polling the
progress endpoint every 500 ms. The final review column includes time from
the reported final review stage to receipt of the response. Total also
includes the initial overhead before drafting was observed.

| Round | Input | Draft | Initial review | Repair | Final review to response | Total | Review outcome |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | Small | 17.7 | 12.6 | 18.7 | 7.2 | 56.7 | Passed after repair |
| 1 | Medium | 19.7 | 18.6 | 19.7 | 7.1 | 65.6 | Passed after repair |
| 1 | Large | 28.2 | 13.1 | 34.8 | 10.3 | 86.9 | Passed after repair |
| 2 | Large | 30.8 | 18.1 | 28.2 | 11.9 | 89.5 | Incomplete |
| 2 | Medium | 19.7 | 15.6 | 21.7 | 9.8 | 67.3 | Passed after repair |
| 2 | Small | 18.7 | 13.1 | 18.1 | 6.3 | 56.7 | Passed after repair |

**Result:** Median response time was 66.5 seconds, with a range of 56.7-89.5
seconds. All six returned HTTP 200 with zero diagrams and no observed
timeouts. All six needed prose repair; five passed final review. The sixth
returned a result marked review incomplete, which is not a timeout.

**Limitation:** This batch covers three inputs from two PRs. It is a baseline,
not an estimate of production latency or timeout rate.

## 2026-11-27: Draft-first browser pilot

One live browser run on the same medium FastAPI chapter at Text-only Overview
settings. Loaded the PR, generated the draft, and immediately requested review
once the unreviewed draft label appeared. Import time and any user pause
between the two actions are excluded.

| Time to readable draft | Optional review | Combined | Review outcome |
| ---: | ---: | ---: | --- |
| 17.5s | 37.2s | 54.7s | Passed after repair |

The browser sent one draft request and one review request; both returned
zero diagrams. This single run verifies the flow, not a typical speedup.

## 2026-11-27: Large-chapter browser follow-up

Used the same Text-only Overview settings on two implementation chapters
from [Vite #16471](https://github.com/vitejs/vite/pull/16471). Both review
requests reused the selected chapter's evidence; no diagrams were returned.

| Chapter | Source characters | Readable draft | Optional review | Combined | Review outcome |
| --- | ---: | ---: | ---: | ---: | --- |
| Implementation: packages - part 2 | 35,852 | 28.8s | 43.5s | 72.3s | Passed after repair |
| Implementation: packages - part 12 | 37,965 | 29.4s | Canceled during repair | Not measured | No result |
| Implementation: packages - part 12, rerun | 37,965 | 22.9s | 48.3s | 71.2s | Passed after repair |

The browser waited for a response on the interrupted attempt, but server
logs recorded cancellation during repair, not a model timeout. The two
completed reviews corrected a transport cache-flag distinction and an
omitted source TODO, respectively. Draft-first reduced time to readable
text in these runs; it did not remove the cost of obtaining a reviewed result.

**Next benchmark:** Repeat on another large PR and track which review findings
could be prevented during drafting.
