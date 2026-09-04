# ABYSS Backend

Run from the repository root:

```bash
source .venv/bin/activate
uvicorn abyss.backend.app.main:app --host 127.0.0.1 --port 8000
```

Swagger UI is available at:

```text
http://127.0.0.1:8000/docs
```

## Example Requests

Detect objects:

```bash
curl -s -X POST "http://127.0.0.1:8000/detect?conf=0.25" \
  -F "file=@abyss/data/processed/detection/test/images/side_scan_sonar_ship-080_png.rf.626d1c7098ccf0ef08f978419e5e0256.jpg"
```

Geolocate with provided survey metadata:

```bash
curl -s -X POST "http://127.0.0.1:8000/geolocate" \
  -H "Content-Type: application/json" \
  -d '{"pixel_x":320,"pixel_y":300,"image_width":640,"image_height":640,"origin_lat":18.5204,"origin_lon":73.8567,"heading_degrees":45,"meters_per_pixel":0.5}'
```

Geolocate with simulated fallback track:

```bash
curl -s -X POST "http://127.0.0.1:8000/geolocate" \
  -H "Content-Type: application/json" \
  -d '{"pixel_x":320,"pixel_y":300,"image_width":640,"image_height":640,"heading_degrees":45,"meters_per_pixel":0.5}'
```

Rank detections by transparent priority formula:

```bash
curl -s -X POST "http://127.0.0.1:8000/priority" \
  -H "Content-Type: application/json" \
  -d '{"detections":[{"detection_id":"a","class_name":"Ship","bbox_xyxy":[28,15,552,545],"composite_confidence":0.59,"distance_to_sensitive_zone_m":1200},{"detection_id":"b","class_name":"Shipwreck","bbox_xyxy":[176,342,276,368],"composite_confidence":0.52,"distance_to_sensitive_zone_m":4000}]}'
```

