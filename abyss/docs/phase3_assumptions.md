# Phase 3 Assumptions

- Shadow direction is assumed to be left-to-right in processed image coordinates because no real sonar heading/slant-range metadata is present in the datasets.
- Shadow consistency is a heuristic computed from the region immediately to the right of the detection box. It uses relative shadow length, darkness versus nearby seabed, and column continuity.
- Image quality is a heuristic computed from contrast, local signal-to-noise ratio, and Laplacian sharpness. It is not a learned quality estimator.
- Correctness is a class-aware, one-to-one IoU >= 0.5 match against validation labels.
- Logistic regression uses standardized detector confidence, shadow consistency, and image quality. Temperature is fitted to validation-set logistic logits by minimizing NLL; neither fit uses test data.
- Validation ECE is an in-sample diagnostic. The improved-detector v4 artifact was fitted on baseline-640 validation predictions and frozen before held-out test evaluation; test data did not fit either model stage. Validation ECE is sensitive to 5 versus 10 bins; see `docs/METRICS.md`.
- Improved_v2 checkpoint training, calibration, inference, and evaluation use baseline preprocessing at 640 pixels. The v3 enhanced-768 artifact and its test results are superseded because they did not match this checkpoint's training configuration.
