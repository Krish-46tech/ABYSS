# Phase 5 Fast-Mode Verification (2026-09-12)

The user finalized the deployed baseline-640/v4 checkpoint. No bounded-experiment candidate was adopted. The frontend was verified against the unchanged backend on port 8001, with Vite on port 5173. The temporary backend was stopped for the failure-state check and then restored. The separate pre-existing backend on port 8000 was not touched.

## Live one-image run

Uploaded the real held-out test image `abyss/data/processed/detection/test/images/side_scan_sonar_ship-097_png.rf.d1a00bc101cd3648e35a1e936ab2a487.jpg` in the browser and clicked Detect. The browser displayed one native Ship detection, a box overlay, all three distinct confidence values, one ranked priority row with component breakdown, and a visibly labeled simulated coordinate. [Screenshot](live_detection.png).

The [saved real `/detect` response](detect_response.json) for the same image confirms processed image shape `[640, 640, 3]`, class `Ship`, box `[323.24176025390625, 57.1595458984375, 598.7230224609375, 546.7343139648438]`, raw detector confidence `0.8917642831802368`, logistic-fused probability `0.8227330724324302`, and final temperature-calibrated probability `0.8332024856709906`. The UI rounded these separately to `0.8918`, `0.8227`, and `0.8332`. The browser's real `/priority` result showed rank 1, priority score `0.6033`, and confidence/hazard/size/proximity components `0.8332/0.6500/0.3293/0.0000`. The coordinate was displayed as `18.520481, 73.857368` with the `Simulated` tag; it is **not** a measured position.

## Failure-state check

Stopped only the temporary backend on port 8001. The open frontend changed to `Backend unavailable`. Clicking Detect again produced a visible `Failed to fetch` error, removed the stale boxes/scores/priority row, and retained the selected image. [Screenshot](backend_down.png). Restored the backend and verified the same image again with the three probabilities and priority row present.

## Code and console checks

- `npm run build` passed (`tsc -b` and Vite production build).
- `rg -n -i 'mock|fixture|sample.?json|fake|hard.?coded|example.?data|static.?detection' frontend/src frontend/scripts` found no matches.
- Browser console warning/error audit after the live and backend-down runs returned an empty list.
- The overlay draws each rectangle from `bbox_xyxy` using the API `image_shape`; it reproduces the backend letterbox geometry. No box coordinates or confidence/priority values are embedded in frontend sample data.
- The 3D view and its unused dependencies were removed. No automated Playwright/Cypress suite, polished redesign, or multiple-image test was performed, per fast-mode scope.

Phase 5 is complete at the requested scope. Phase 6 and 7 have not started.
