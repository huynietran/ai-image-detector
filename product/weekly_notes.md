# Product weekly notes

One entry per week from the Product lead: what the API and demo can do, what was tested, and what is not done.
(Evaluation's results table lives in `evaluation/results/weekly_notes.md`.)

## Week 4: demo test on the real step-2 checkpoint (Oct 29)

- Setup: the Gradio demo served `step2.pt` (SHA-256 `9132a8901dc453a7400b1830c0662567604464364fc0b76fd8cd617811b971e2`, matches Evaluation's copy) with `AI_THRESHOLD` = 0.9838 from `threshold.json`.
- Consistency check: for one held-out Flux image (`ai_b7c9f908404b.jpg`) the demo and Model's Colab run give the same score (0.199) and the same five crop scores, so Product's loading and scoring path agrees with Model's.
- Demo tests (the real photo's filename was assumed, not confirmed):

| Image | Truth | Score | Result at 0.9838 | Fair test? |
|---|---|---|---|---|
| `C.jpg` (signed C2PA sample) | AI, per its metadata | n/a | Metadata says AI-generated | Metadata path, no model |
| `real_322a1824d426.jpg` | Real, held out | 0.04 | Not flagged | Yes |
| `ai_49704cef2e01.jpg` (red bus) | Flux, held out | 0.85 | Missed (inconclusive, below cut-off) | Yes |
| `ai_b7c9f908404b.jpg` (cookie) | Flux, held out | 0.20 | Missed (not flagged) | Yes |
| `ai_cd34ef72264a.jpg` (surfers) | Stable Diffusion, trained on | 1.00 | Flagged | No: seen in training |

- Reading: on unseen Flux, 0 of 2 AI images were flagged and the 1 real photo was cleared. That fits the measured ~10% TPR at 1% FPR. The surfers result only shows the model remembers its training images.
- Bug found and fixed (PR #5): the demo headline ignored `AI_THRESHOLD` and showed whichever bar was larger (an effective 0.5 cut-off). It now has three outcomes (flagged / inconclusive / not flagged), shows the score instead of a percentage, warns when the five crops disagree by more than 0.5, and shows metadata verdicts without a percentage. "Not flagged" is worded as not proof of real.
- Not done yet: user sessions with 3-5 testers; testing with more than five images.
