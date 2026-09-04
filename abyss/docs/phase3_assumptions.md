# Phase 3 Assumptions

- Shadow direction is assumed to be left-to-right in processed image coordinates because no real sonar heading/slant-range metadata is present in the datasets.
- Shadow consistency is a heuristic computed from the region immediately to the right of the detection box. It uses relative shadow length, darkness versus nearby seabed, and column continuity.
- Image quality is a heuristic computed from contrast, local signal-to-noise ratio, and Laplacian sharpness. It is not a learned quality estimator.
- Composite confidence uses a transparent weighted average: detector confidence 0.60, image quality 0.20, shadow consistency 0.20.
- The Phase 3 confidence values are meant for triage and presentation honesty, not calibrated probabilities.
