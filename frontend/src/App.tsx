import { Canvas } from "@react-three/fiber";
import {
  AlertTriangle,
  CheckCircle2,
  Crosshair,
  Loader2,
  MapPin,
  Radar,
  ServerCrash,
  Upload,
} from "lucide-react";
import { ChangeEvent, FormEvent, useEffect, useMemo, useRef, useState } from "react";

const API_BASE = import.meta.env.VITE_ABYSS_API_BASE ?? "http://127.0.0.1:8000";
const DETECTION_SPACE_SIZE = 640;

type Detection = {
  detection_id: number;
  class_id: number;
  class_name: string;
  bbox_xyxy: [number, number, number, number];
  detector_confidence: number;
  image_quality_score: number;
  shadow_consistency_score: number;
  composite_confidence: number;
};

type DetectResponse = {
  filename: string;
  image_shape: number[];
  processing_latency_ms: number;
  detections: Detection[];
};

type GeolocateResponse = {
  lat: number;
  lon: number;
  geolocation_source: "provided_metadata" | "simulated";
  east_offset_m: number;
  north_offset_m: number;
  input_pixel: number[];
};

type PriorityItem = {
  rank: number;
  detection_id: string;
  priority_score: number;
  score_breakdown: Record<string, number>;
  class_name: string;
  composite_confidence: number;
};

type SessionDetection = Detection & {
  geo?: GeolocateResponse;
};

function logEvent(event: string, payload: Record<string, unknown>) {
  console.info(JSON.stringify({ event, timestamp: new Date().toISOString(), ...payload }));
}

async function fetchJson<T>(url: string, init: RequestInit, timeoutMs = 120000): Promise<T> {
  const controller = new AbortController();
  const timeout = window.setTimeout(() => controller.abort(), timeoutMs);
  const started = performance.now();
  try {
    const response = await fetch(url, { ...init, signal: controller.signal });
    const latencyMs = performance.now() - started;
    if (latencyMs > 3000) {
      logEvent("slow_api_response", { url, latency_ms: Number(latencyMs.toFixed(2)) });
    }
    if (!response.ok) {
      const detail = await response.text();
      throw new Error(`${response.status} ${response.statusText}: ${detail}`);
    }
    return (await response.json()) as T;
  } catch (error) {
    logEvent("api_call_failed", {
      url,
      message: error instanceof Error ? error.message : String(error),
    });
    throw error;
  } finally {
    window.clearTimeout(timeout);
  }
}

function scoreColor(value: number) {
  if (value >= 0.7) return "bg-emerald-400";
  if (value >= 0.45) return "bg-amber-300";
  return "bg-rose-400";
}

function formatScore(value: number) {
  return value.toFixed(4);
}

function bboxCenter([x1, y1, x2, y2]: [number, number, number, number]) {
  return { x: (x1 + x2) / 2, y: (y1 + y2) / 2 };
}

function SonarCanvas({
  imageUrl,
  detections,
  selectedId,
  onSelect,
}: {
  imageUrl: string | null;
  detections: SessionDetection[];
  selectedId: number | null;
  onSelect: (id: number) => void;
}) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.fillStyle = "#111820";
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    if (!imageUrl) {
      ctx.fillStyle = "#94a3b8";
      ctx.font = "18px sans-serif";
      ctx.textAlign = "center";
      ctx.fillText("Upload a sonar tile to begin", canvas.width / 2, canvas.height / 2);
      return;
    }

    const image = new Image();
    image.onload = () => {
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.fillStyle = "#05080b";
      ctx.fillRect(0, 0, canvas.width, canvas.height);

      const scale = Math.min(canvas.width / image.width, canvas.height / image.height);
      const drawWidth = image.width * scale;
      const drawHeight = image.height * scale;
      const offsetX = (canvas.width - drawWidth) / 2;
      const offsetY = (canvas.height - drawHeight) / 2;
      ctx.drawImage(image, offsetX, offsetY, drawWidth, drawHeight);

      detections.forEach((detection) => {
        const [x1, y1, x2, y2] = detection.bbox_xyxy;
        const sx = canvas.width / DETECTION_SPACE_SIZE;
        const sy = canvas.height / DETECTION_SPACE_SIZE;
        const isSelected = detection.detection_id === selectedId;
        const isUnknown = detection.class_name === "UNKNOWN_ANOMALY";
        ctx.strokeStyle = isUnknown ? "#fb7185" : isSelected ? "#facc15" : "#22d3ee";
        ctx.lineWidth = isSelected ? 4 : 2;
        ctx.strokeRect(x1 * sx, y1 * sy, (x2 - x1) * sx, (y2 - y1) * sy);
        ctx.fillStyle = isUnknown ? "#9f1239" : isSelected ? "#854d0e" : "#155e75";
        const label = `${detection.class_name} ${formatScore(detection.composite_confidence)}`;
        ctx.font = "14px sans-serif";
        const labelWidth = Math.min(ctx.measureText(label).width + 12, canvas.width - x1 * sx - 4);
        ctx.fillRect(x1 * sx, Math.max(0, y1 * sy - 24), labelWidth, 22);
        ctx.fillStyle = "#ffffff";
        ctx.fillText(label, x1 * sx + 6, Math.max(16, y1 * sy - 8));
      });
    };
    image.onerror = () => {
      ctx.fillStyle = "#fca5a5";
      ctx.fillText("The selected image could not be rendered.", canvas.width / 2, canvas.height / 2);
    };
    image.src = imageUrl;
  }, [detections, imageUrl, selectedId]);

  function handleCanvasClick(event: React.MouseEvent<HTMLCanvasElement>) {
    const canvas = canvasRef.current;
    if (!canvas || detections.length === 0) return;
    const rect = canvas.getBoundingClientRect();
    const x = ((event.clientX - rect.left) / rect.width) * DETECTION_SPACE_SIZE;
    const y = ((event.clientY - rect.top) / rect.height) * DETECTION_SPACE_SIZE;
    const hit = detections.find((detection) => {
      const [x1, y1, x2, y2] = detection.bbox_xyxy;
      return x >= x1 && x <= x2 && y >= y1 && y <= y2;
    });
    if (hit) onSelect(hit.detection_id);
  }

  return (
    <canvas
      ref={canvasRef}
      width={DETECTION_SPACE_SIZE}
      height={DETECTION_SPACE_SIZE}
      onClick={handleCanvasClick}
      className="aspect-square w-full cursor-crosshair border border-slate-700 bg-slate-950"
      aria-label="Sonar detection canvas"
    />
  );
}

function ConfidenceBreakdown({ detection }: { detection: SessionDetection | undefined }) {
  if (!detection) {
    return (
      <section className="border border-slate-800 bg-slate-950 p-4">
        <h2 className="text-sm font-semibold uppercase tracking-wide text-slate-300">Confidence Breakdown</h2>
        <p className="mt-4 text-sm text-slate-400">Select a detection to inspect its four backend scores.</p>
      </section>
    );
  }

  const rows = [
    ["detector_confidence", detection.detector_confidence],
    ["image_quality_score", detection.image_quality_score],
    ["shadow_consistency_score", detection.shadow_consistency_score],
    ["composite_confidence", detection.composite_confidence],
  ] as const;

  return (
    <section className="border border-slate-800 bg-slate-950 p-4">
      <div className="flex items-center justify-between gap-3">
        <h2 className="text-sm font-semibold uppercase tracking-wide text-slate-300">Confidence Breakdown</h2>
        <span
          className={`border px-2 py-1 text-xs font-semibold ${
            detection.class_name === "UNKNOWN_ANOMALY"
              ? "border-rose-400 bg-rose-950 text-rose-100"
              : "border-cyan-500 bg-cyan-950 text-cyan-100"
          }`}
        >
          {detection.class_name}
        </span>
      </div>
      <div className="mt-4 space-y-4">
        {rows.map(([label, value]) => (
          <div key={label}>
            <div className="mb-1 flex items-center justify-between text-xs text-slate-300">
              <span>{label}</span>
              <span className="font-mono">{formatScore(value)}</span>
            </div>
            <div className="h-3 border border-slate-700 bg-slate-900">
              <div className={`h-full ${scoreColor(value)}`} style={{ width: `${Math.max(0, Math.min(1, value)) * 100}%` }} />
            </div>
          </div>
        ))}
      </div>
      {detection.geo && (
        <div className="mt-4 border-t border-slate-800 pt-4 text-sm text-slate-300">
          <div className="flex items-center gap-2">
            <MapPin size={16} />
            <span className="font-mono">{detection.geo.lat.toFixed(6)}, {detection.geo.lon.toFixed(6)}</span>
            <span className="border border-amber-400 px-2 py-0.5 text-xs text-amber-100">{detection.geo.geolocation_source}</span>
          </div>
        </div>
      )}
    </section>
  );
}

function PriorityList({ items, detections }: { items: PriorityItem[]; detections: SessionDetection[] }) {
  const byId = new Map(detections.map((detection) => [String(detection.detection_id), detection]));
  return (
    <section className="border border-slate-800 bg-slate-950 p-4">
      <h2 className="text-sm font-semibold uppercase tracking-wide text-slate-300">Priority List</h2>
      {items.length === 0 ? (
        <p className="mt-4 text-sm text-slate-400">Run detection first; priorities are requested from the backend for current detections.</p>
      ) : (
        <div className="mt-4 overflow-x-auto">
          <table className="w-full min-w-[720px] border-collapse text-left text-sm">
            <thead className="border-b border-slate-800 text-xs uppercase text-slate-400">
              <tr>
                <th className="py-2 pr-3">Rank</th>
                <th className="py-2 pr-3">Hazard</th>
                <th className="py-2 pr-3">Composite</th>
                <th className="py-2 pr-3">Priority</th>
                <th className="py-2 pr-3">Breakdown</th>
                <th className="py-2 pr-3">Coordinate</th>
              </tr>
            </thead>
            <tbody>
              {items.map((item) => {
                const detection = byId.get(item.detection_id);
                return (
                  <tr key={item.detection_id} className="border-b border-slate-900">
                    <td className="py-3 pr-3 font-mono">{item.rank}</td>
                    <td className="py-3 pr-3">{item.class_name}</td>
                    <td className="py-3 pr-3 font-mono">{formatScore(item.composite_confidence)}</td>
                    <td className="py-3 pr-3 font-mono">{formatScore(item.priority_score)}</td>
                    <td className="py-3 pr-3 font-mono text-xs text-slate-300">
                      conf {formatScore(item.score_breakdown.confidence_component)} · hazard {formatScore(item.score_breakdown.hazard_component)} · size {formatScore(item.score_breakdown.size_component)} · prox {formatScore(item.score_breakdown.proximity_component)}
                    </td>
                    <td className="py-3 pr-3 text-xs text-slate-300">
                      {detection?.geo ? (
                        <span>
                          {detection.geo.lat.toFixed(5)}, {detection.geo.lon.toFixed(5)}
                          <span className="ml-2 border border-amber-500 px-1.5 py-0.5 text-amber-100">
                            {detection.geo.geolocation_source}
                          </span>
                        </span>
                      ) : (
                        "not requested"
                      )}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}
    </section>
  );
}

function SeabedMap({ detections, selectedId, onSelect }: { detections: SessionDetection[]; selectedId: number | null; onSelect: (id: number) => void }) {
  const geolocated = detections.filter((detection) => detection.geo);
  const maxOffset = Math.max(
    1,
    ...geolocated.map((detection) => Math.abs(detection.geo?.east_offset_m ?? 0)),
    ...geolocated.map((detection) => Math.abs(detection.geo?.north_offset_m ?? 0)),
  );

  return (
    <section className="border border-slate-800 bg-slate-950 p-4">
      <div className="flex items-center justify-between gap-3">
        <h2 className="text-sm font-semibold uppercase tracking-wide text-slate-300">3D Map</h2>
        <span className="text-xs text-slate-400">Seabed terrain is procedural; markers use API geolocation output.</span>
      </div>
      <div className="mt-4 h-72 border border-slate-800 bg-slate-900">
        <Canvas camera={{ position: [0, 9, 12], fov: 45 }}>
          <ambientLight intensity={0.7} />
          <directionalLight position={[4, 7, 5]} intensity={1} />
          <mesh rotation={[-Math.PI / 2, 0, 0]}>
            <planeGeometry args={[18, 18, 48, 48]} />
            <meshStandardMaterial color="#1e5662" wireframe />
          </mesh>
          {geolocated.map((detection) => {
            const x = ((detection.geo?.east_offset_m ?? 0) / maxOffset) * 7;
            const z = -((detection.geo?.north_offset_m ?? 0) / maxOffset) * 7;
            const selected = detection.detection_id === selectedId;
            return (
              <mesh
                key={detection.detection_id}
                position={[x, selected ? 0.75 : 0.45, z]}
                onClick={() => onSelect(detection.detection_id)}
              >
                <sphereGeometry args={[selected ? 0.38 : 0.25, 24, 24]} />
                <meshStandardMaterial color={detection.class_name === "UNKNOWN_ANOMALY" ? "#fb7185" : selected ? "#facc15" : "#22d3ee"} />
              </mesh>
            );
          })}
        </Canvas>
      </div>
    </section>
  );
}

export default function App() {
  const [file, setFile] = useState<File | null>(null);
  const [imageUrl, setImageUrl] = useState<string | null>(null);
  const [detectResponse, setDetectResponse] = useState<DetectResponse | null>(null);
  const [detections, setDetections] = useState<SessionDetection[]>([]);
  const [priorities, setPriorities] = useState<PriorityItem[]>([]);
  const [selectedId, setSelectedId] = useState<number | null>(null);
  const [isDetecting, setIsDetecting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [backendOnline, setBackendOnline] = useState<boolean | null>(null);

  const selectedDetection = useMemo(
    () => detections.find((detection) => detection.detection_id === selectedId),
    [detections, selectedId],
  );

  useEffect(() => {
    let active = true;
    async function checkHealth() {
      try {
        const response = await fetch(`${API_BASE}/health`, { cache: "no-store" });
        if (active) setBackendOnline(response.ok);
        if (!response.ok) logEvent("backend_health_failed", { status: response.status });
      } catch (healthError) {
        if (active) setBackendOnline(false);
        logEvent("backend_health_failed", {
          message: healthError instanceof Error ? healthError.message : String(healthError),
        });
      }
    }
    checkHealth();
    const interval = window.setInterval(checkHealth, 5000);
    return () => {
      active = false;
      window.clearInterval(interval);
    };
  }, []);

  useEffect(() => {
    return () => {
      if (imageUrl) URL.revokeObjectURL(imageUrl);
    };
  }, [imageUrl]);

  function handleFileChange(event: ChangeEvent<HTMLInputElement>) {
    const selected = event.target.files?.[0] ?? null;
    setFile(selected);
    setDetectResponse(null);
    setDetections([]);
    setPriorities([]);
    setSelectedId(null);
    setError(null);
    if (imageUrl) URL.revokeObjectURL(imageUrl);
    setImageUrl(selected ? URL.createObjectURL(selected) : null);
    if (selected && new URLSearchParams(window.location.search).get("autodetect") === "1") {
      window.setTimeout(() => {
        runDetection(selected);
      }, 0);
    }
  }

  async function geolocateDetections(rows: Detection[]): Promise<SessionDetection[]> {
    return Promise.all(
      rows.map(async (detection) => {
        const center = bboxCenter(detection.bbox_xyxy);
        const geo = await fetchJson<GeolocateResponse>(`${API_BASE}/geolocate`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            pixel_x: center.x,
            pixel_y: center.y,
            image_width: DETECTION_SPACE_SIZE,
            image_height: DETECTION_SPACE_SIZE,
            heading_degrees: 0,
            meters_per_pixel: 0.5,
          }),
        });
        return { ...detection, geo };
      }),
    );
  }

  async function rankDetections(rows: SessionDetection[]) {
    if (rows.length === 0) {
      setPriorities([]);
      return;
    }
    const response = await fetchJson<{ ranked_detections: PriorityItem[] }>(`${API_BASE}/priority`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        detections: rows.map((detection) => ({
          detection_id: String(detection.detection_id),
          class_name: detection.class_name,
          bbox_xyxy: detection.bbox_xyxy,
          composite_confidence: detection.composite_confidence,
        })),
      }),
    });
    setPriorities(response.ranked_detections);
  }

  async function runDetection(targetFile: File | null) {
    if (!targetFile) {
      setError("Select a sonar image before running detection.");
      return;
    }
    setIsDetecting(true);
    setError(null);
    try {
      const form = new FormData();
      form.append("file", targetFile);
      const response = await fetchJson<DetectResponse>(`${API_BASE}/detect?conf=0.25`, {
        method: "POST",
        body: form,
      });
      setDetectResponse(response);
      const geolocated = await geolocateDetections(response.detections);
      setDetections(geolocated);
      setSelectedId(geolocated[0]?.detection_id ?? null);
      await rankDetections(geolocated);
      logEvent("detect_flow_completed", {
        filename: response.filename,
        detection_count: response.detections.length,
        processing_latency_ms: Number(response.processing_latency_ms.toFixed(2)),
      });
    } catch (submitError) {
      setError(submitError instanceof Error ? submitError.message : String(submitError));
    } finally {
      setIsDetecting(false);
    }
  }

  async function handleSubmit(event: FormEvent) {
    event.preventDefault();
    await runDetection(file);
  }

  return (
    <main className="min-h-screen bg-[#0b1014] text-slate-100">
      <header className="border-b border-slate-800 bg-[#10171c]">
        <div className="mx-auto flex max-w-7xl items-center justify-between px-5 py-4">
          <div className="flex items-center gap-3">
            <Radar className="text-cyan-300" size={28} />
            <div>
              <h1 className="text-xl font-semibold">ABYSS</h1>
              <p className="text-sm text-slate-400">Sonar detection and triage dashboard</p>
            </div>
          </div>
          <div
            className={`flex items-center gap-2 border px-3 py-2 text-sm ${
              backendOnline ? "border-emerald-500 text-emerald-200" : "border-rose-500 text-rose-200"
            }`}
          >
            {backendOnline ? <CheckCircle2 size={16} /> : <ServerCrash size={16} />}
            {backendOnline === null ? "Checking backend" : backendOnline ? "Backend reachable" : "Backend unavailable"}
          </div>
        </div>
      </header>

      <div className="mx-auto grid max-w-7xl gap-5 px-5 py-5 lg:grid-cols-[minmax(0,1.25fr)_minmax(360px,0.75fr)]">
        <section className="border border-slate-800 bg-slate-950 p-4">
          <form onSubmit={handleSubmit} className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
            <label className="flex min-w-0 flex-1 items-center gap-3 border border-slate-700 bg-slate-900 px-3 py-2">
              <Upload size={18} className="shrink-0 text-cyan-300" />
              <input type="file" accept="image/*" onChange={handleFileChange} className="min-w-0 flex-1 text-sm text-slate-300 file:mr-3 file:border-0 file:bg-cyan-900 file:px-3 file:py-1.5 file:text-cyan-50" />
            </label>
            <button
              type="submit"
              disabled={isDetecting || !file}
              className="flex items-center justify-center gap-2 border border-cyan-500 bg-cyan-950 px-4 py-2 text-sm font-semibold text-cyan-50 disabled:cursor-not-allowed disabled:border-slate-700 disabled:bg-slate-900 disabled:text-slate-500"
            >
              {isDetecting ? <Loader2 className="animate-spin" size={18} /> : <Crosshair size={18} />}
              Detect
            </button>
          </form>

          {error && (
            <div className="mt-4 flex items-start gap-2 border border-rose-500 bg-rose-950 px-3 py-2 text-sm text-rose-100">
              <AlertTriangle size={18} className="mt-0.5 shrink-0" />
              <span>{error}</span>
            </div>
          )}

          <div className="mt-4">
            <SonarCanvas imageUrl={imageUrl} detections={detections} selectedId={selectedId} onSelect={setSelectedId} />
          </div>

          <div className="mt-4 grid gap-3 text-sm text-slate-300 sm:grid-cols-3">
            <div className="border border-slate-800 bg-slate-900 p-3">
              <div className="text-xs uppercase text-slate-500">File</div>
              <div className="mt-1 truncate">{detectResponse?.filename ?? file?.name ?? "none"}</div>
            </div>
            <div className="border border-slate-800 bg-slate-900 p-3">
              <div className="text-xs uppercase text-slate-500">Detections</div>
              <div className="mt-1 font-mono">{detections.length}</div>
            </div>
            <div className="border border-slate-800 bg-slate-900 p-3">
              <div className="text-xs uppercase text-slate-500">Backend Latency</div>
              <div className="mt-1 font-mono">
                {detectResponse ? `${detectResponse.processing_latency_ms.toFixed(2)} ms` : "not run"}
              </div>
            </div>
          </div>
        </section>

        <div className="space-y-5">
          <ConfidenceBreakdown detection={selectedDetection} />
          <SeabedMap detections={detections} selectedId={selectedId} onSelect={setSelectedId} />
        </div>

        <div className="lg:col-span-2">
          <PriorityList items={priorities} detections={detections} />
        </div>
      </div>
    </main>
  );
}
