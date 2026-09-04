# Phase 1 Assumptions

## Survey/Site Grouping

- The shipwreck dataset has natural site names in filenames, such as `Barge_No_1_13.png`; the site prefix before the final numeric suffix is used as the group.
- The side-scan YOLO dataset does not include navigation, survey, or site metadata. Filenames follow patterns such as `ship-154_png.rf.<hash>.jpg`; ABYSS uses the object prefix plus sequential 25-image numeric block as a proxy survey pass, for example `side_scan_ship_block_006`.
- Splits are assigned by whole group only. The code asserts that no group appears in more than one split.

## Detection Labels

- Side-scan YOLO labels are preserved as bounding boxes after letterbox resizing.
- Shipwreck segmentation masks are converted to one YOLO bounding box around the non-zero mask region. Empty masks are kept as background images with empty YOLO label files.
- The processed detection class list is `Plane`, `Ship`, and `Shipwreck`.

## Preprocessing

- Each image is converted to grayscale, denoised with a 3x3 median filter for speckle-like noise, normalized to 0-255 intensity, converted back to 3-channel BGR, and letterbox-resized to 640x640.
- Letterboxing preserves aspect ratio for tall sonar strips and adjusts labels into the resized coordinate frame.
