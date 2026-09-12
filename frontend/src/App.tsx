import { AlertTriangle, CheckCircle2, Crosshair, Loader2, Radar, ServerCrash, Upload } from "lucide-react";
import { ChangeEvent, FormEvent, useEffect, useMemo, useRef, useState } from "react";

const API_BASE = import.meta.env.VITE_ABYSS_API_BASE ?? "http://127.0.0.1:8000";

type Detection = {
  detection_id: number;
  class_id: number;
  class_name: string;
  bbox_xyxy: [number, number, number, number];
  detector_confidence: number;
  logistic_fused_probability: number;
  temperature_calibrated_probability: number;
};
type DetectResponse = {
  filename: string;
  image_shape: [number, number, number];
  processing_latency_ms: number;
  detections: Detection[];
};
type GeoResponse = {
  lat: number;
  lon: number;
  geolocation_source: "provided_metadata" | "simulated";
};
type SessionDetection = Detection & { geo: GeoResponse };
type PriorityItem = {
  rank: number;
  detection_id: string;
  class_name: string;
  priority_score: number;
  score_breakdown: Record<string, number>;
};

const score = (value: number) => value.toFixed(4);
const errorMessage = (error: unknown) => error instanceof Error ? error.message : String(error);

async function apiJson<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, { ...init, signal: AbortSignal.timeout(120000) });
  if (!response.ok) throw new Error(`${path} failed (${response.status}): ${await response.text()}`);
  return (await response.json()) as T;
}

function checkedDetectionResponse(value: DetectResponse): DetectResponse {
  const [height, width] = value.image_shape ?? [];
  if (!Number.isInteger(height) || !Number.isInteger(width) || height <= 0 || width <= 0 || !Array.isArray(value.detections)) {
    throw new Error("Detection response has no valid image dimensions or detection list.");
  }
  for (const detection of value.detections) {
    const [x1, y1, x2, y2] = detection.bbox_xyxy ?? [];
    const numbers = [x1, y1, x2, y2, detection.detector_confidence,
      detection.logistic_fused_probability, detection.temperature_calibrated_probability];
    if (numbers.some((number) => !Number.isFinite(number)) || x1 < 0 || y1 < 0 || x2 > width || y2 > height || x2 <= x1 || y2 <= y1) {
      throw new Error("Detection response contains an invalid box or confidence value.");
    }
  }
  return value;
}

function ImageOverlay({ imageUrl, response, selectedId, onSelect }: {
  imageUrl: string | null;
  response: DetectResponse | null;
  selectedId: number | null;
  onSelect: (id: number) => void;
}) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const height = response?.image_shape[0] ?? 640;
  const width = response?.image_shape[1] ?? 640;

  useEffect(() => {
    const canvas = canvasRef.current;
    const context = canvas?.getContext("2d");
    if (!canvas || !context) return;
    context.fillStyle = "#727272";
    context.fillRect(0, 0, width, height);
    if (!imageUrl) return;
    let active = true;
    const image = new Image();
    image.onload = () => {
      if (!active) return;
      const scale = Math.min(width / image.width, height / image.height);
      const drawWidth = Math.round(image.width * scale);
      const drawHeight = Math.round(image.height * scale);
      const padX = Math.floor((width - drawWidth) / 2);
      const padY = Math.floor((height - drawHeight) / 2);
      context.fillStyle = "#727272";
      context.fillRect(0, 0, width, height);
      context.drawImage(image, padX, padY, drawWidth, drawHeight);
      for (const detection of response?.detections ?? []) {
        const [x1, y1, x2, y2] = detection.bbox_xyxy;
        const selected = detection.detection_id === selectedId;
        context.strokeStyle = selected ? "#facc15" : "#22d3ee";
        context.lineWidth = selected ? 4 : 2;
        context.strokeRect(x1, y1, x2 - x1, y2 - y1);
        context.font = "bold 15px sans-serif";
        const labelWidth = Math.min(context.measureText(detection.class_name).width + 14, width - x1);
        context.fillStyle = selected ? "#854d0e" : "#155e75";
        context.fillRect(x1, Math.max(0, y1 - 24), labelWidth, 23);
        context.fillStyle = "white";
        context.fillText(detection.class_name, x1 + 7, Math.max(17, y1 - 7));
      }
    };
    image.onerror = () => {
      if (!active) return;
      context.fillStyle = "#fca5a5";
      context.font = "16px sans-serif";
      context.fillText("Image could not be displayed", 20, 32);
    };
    image.src = imageUrl;
    return () => { active = false; };
  }, [height, imageUrl, response, selectedId, width]);

  function selectBox(event: React.MouseEvent<HTMLCanvasElement>) {
    const bounds = event.currentTarget.getBoundingClientRect();
    const x = (event.clientX - bounds.left) * width / bounds.width;
    const y = (event.clientY - bounds.top) * height / bounds.height;
    const hit = response?.detections.find((detection) => {
      const [x1, y1, x2, y2] = detection.bbox_xyxy;
      return x >= x1 && x <= x2 && y >= y1 && y <= y2;
    });
    if (hit) onSelect(hit.detection_id);
  }

  return <canvas ref={canvasRef} width={width} height={height} onClick={selectBox}
    className="mx-auto block w-full max-w-[640px] border border-slate-700 bg-slate-900"
    style={{ aspectRatio: `${width} / ${height}` }} aria-label="Sonar image with API bounding boxes" />;
}

function ConfidenceBreakdown({ detection }: { detection: SessionDetection | undefined }) {
  const rows = detection ? [
    ["Raw detector confidence", detection.detector_confidence],
    ["Logistic-fused probability", detection.logistic_fused_probability],
    ["Temperature-calibrated probability", detection.temperature_calibrated_probability],
  ] as const : [];
  return <section className="border border-slate-700 bg-slate-950 p-4">
    <h2 className="text-base font-semibold">Confidence Breakdown</h2>
    {detection ? <>
      <p className="mt-1 text-sm text-cyan-200">{detection.class_name} · detection {detection.detection_id}</p>
      <dl className="mt-4 space-y-3">{rows.map(([label, value]) =>
        <div key={label} className="flex items-center justify-between gap-3 border-b border-slate-800 pb-2 text-sm">
          <dt className="text-slate-300">{label}</dt><dd className="font-mono text-white">{score(value)}</dd>
        </div>)}</dl>
      <p className="mt-4 text-sm text-slate-300">{detection.geo.lat.toFixed(6)}, {detection.geo.lon.toFixed(6)}
        <span className="ml-2 border border-amber-500 px-2 py-0.5 text-xs text-amber-200">
          {detection.geo.geolocation_source === "simulated" ? "Simulated" : "Provided metadata"}
        </span>
      </p>
    </> : <p className="mt-3 text-sm text-slate-400">No detection selected.</p>}
  </section>;
}

function PriorityList({ items, detections }: { items: PriorityItem[]; detections: SessionDetection[] }) {
  const byId = new Map(detections.map((detection) => [String(detection.detection_id), detection]));
  return <section className="border border-slate-700 bg-slate-950 p-4">
    <h2 className="text-base font-semibold">Priority List</h2>
    {items.length === 0 ? <p className="mt-3 text-sm text-slate-400">No ranked detections.</p> :
      <div className="mt-3 overflow-x-auto"><table className="w-full min-w-[960px] border-collapse text-left text-sm">
        <thead className="border-b border-slate-700 text-slate-400"><tr>
          <th className="py-2 pr-3">Rank</th><th className="pr-3">Class</th><th className="pr-3">Raw</th>
          <th className="pr-3">Fused</th><th className="pr-3">Calibrated</th><th className="pr-3">Priority</th>
          <th className="pr-3">Breakdown</th><th>Coordinate · source</th>
        </tr></thead>
        <tbody>{items.map((item) => {
          const detection = byId.get(item.detection_id);
          if (!detection) return null;
          const breakdown = item.score_breakdown;
          return <tr key={item.detection_id} className="border-b border-slate-800 align-top">
            <td className="py-3 pr-3 font-mono">{item.rank}</td><td className="py-3 pr-3">{item.class_name}</td>
            <td className="py-3 pr-3 font-mono">{score(detection.detector_confidence)}</td>
            <td className="py-3 pr-3 font-mono">{score(detection.logistic_fused_probability)}</td>
            <td className="py-3 pr-3 font-mono">{score(detection.temperature_calibrated_probability)}</td>
            <td className="py-3 pr-3 font-mono">{score(item.priority_score)}</td>
            <td className="py-3 pr-3 font-mono text-xs text-slate-300">
              <div>Confidence {score(breakdown.confidence_component)}</div>
              <div>Hazard {score(breakdown.hazard_component)} · Size {score(breakdown.size_component)} · Proximity {score(breakdown.proximity_component)}</div>
            </td>
            <td className="py-3 font-mono text-xs">
              <span>{detection.geo.lat.toFixed(6)}, {detection.geo.lon.toFixed(6)}</span>
              <span className="ml-2 border border-amber-500 px-1.5 py-0.5 text-amber-200">
                {detection.geo.geolocation_source === "simulated" ? "Simulated" : "Provided metadata"}
              </span>
            </td>
          </tr>;
        })}</tbody>
      </table></div>}
  </section>;
}

export default function App() {
  const [file, setFile] = useState<File | null>(null);
  const [imageUrl, setImageUrl] = useState<string | null>(null);
  const [response, setResponse] = useState<DetectResponse | null>(null);
  const [detections, setDetections] = useState<SessionDetection[]>([]);
  const [priorities, setPriorities] = useState<PriorityItem[]>([]);
  const [selectedId, setSelectedId] = useState<number | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [backendOnline, setBackendOnline] = useState<boolean | null>(null);
  const selected = useMemo(() => detections.find((detection) => detection.detection_id === selectedId), [detections, selectedId]);

  useEffect(() => {
    let active = true;
    async function checkHealth() {
      try {
        const health = await apiJson<{ status: string }>("/health");
        if (active) setBackendOnline(health.status === "ok");
      } catch {
        if (active) setBackendOnline(false);
      }
    }
    void checkHealth();
    const interval = window.setInterval(checkHealth, 3000);
    return () => { active = false; window.clearInterval(interval); };
  }, []);
  useEffect(() => () => { if (imageUrl) URL.revokeObjectURL(imageUrl); }, [imageUrl]);

  function chooseFile(event: ChangeEvent<HTMLInputElement>) {
    const next = event.target.files?.[0] ?? null;
    setFile(next);
    setImageUrl(next ? URL.createObjectURL(next) : null);
    setResponse(null);
    setDetections([]);
    setPriorities([]);
    setSelectedId(null);
    setError(null);
    if (next && new URLSearchParams(window.location.search).get("autodetect") === "1") {
      window.setTimeout(() => { void runDetection(next); }, 0);
    }
  }

  async function runDetection(target: File | null) {
    if (!target) { setError("Select an image before detecting."); return; }
    setBusy(true);
    setError(null);
    setResponse(null);
    setDetections([]);
    setPriorities([]);
    try {
      const form = new FormData();
      form.append("file", target);
      const detected = checkedDetectionResponse(await apiJson<DetectResponse>("/detect?conf=0.25", { method: "POST", body: form }));
      setBackendOnline(true);
      setResponse(detected);
      if (detected.detections.length === 0) return;
      const [height, width] = detected.image_shape;
      const geolocated = await Promise.all(detected.detections.map(async (detection): Promise<SessionDetection> => {
        const [x1, y1, x2, y2] = detection.bbox_xyxy;
        const geo = await apiJson<GeoResponse>("/geolocate", {
          method: "POST", headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ pixel_x: (x1 + x2) / 2, pixel_y: (y1 + y2) / 2,
            image_width: width, image_height: height }),
        });
        if (!Number.isFinite(geo.lat) || !Number.isFinite(geo.lon) ||
            !["provided_metadata", "simulated"].includes(geo.geolocation_source)) {
          throw new Error("Geolocation response is missing coordinates or a source tag.");
        }
        return { ...detection, geo };
      }));
      setDetections(geolocated);
      setSelectedId(geolocated[0].detection_id);
      const priority = await apiJson<{ ranked_detections: PriorityItem[] }>("/priority", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ detections: geolocated.map((detection) => ({
          detection_id: String(detection.detection_id), class_name: detection.class_name,
          bbox_xyxy: detection.bbox_xyxy,
          composite_confidence: detection.temperature_calibrated_probability,
        })) }),
      });
      if (!Array.isArray(priority.ranked_detections) || priority.ranked_detections.length !== geolocated.length ||
          priority.ranked_detections.some((item) => !Number.isFinite(item.priority_score) ||
            !geolocated.some((detection) => String(detection.detection_id) === item.detection_id) ||
            ["confidence_component", "hazard_component", "size_component", "proximity_component"]
              .some((key) => !Number.isFinite(item.score_breakdown?.[key])))) {
        throw new Error("Priority response is incomplete or does not match detections.");
      }
      setPriorities(priority.ranked_detections);
    } catch (failure) {
      setError(errorMessage(failure));
      if (failure instanceof TypeError) setBackendOnline(false);
    } finally {
      setBusy(false);
    }
  }

  function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    void runDetection(file);
  }

  return <main className="min-h-screen bg-[#0b1014] text-slate-100">
    <header className="border-b border-slate-700 bg-[#141c21]"><div className="mx-auto flex max-w-7xl flex-wrap items-center justify-between gap-3 px-4 py-4">
      <div className="flex items-center gap-3"><Radar size={27} className="text-cyan-300" /><div><h1 className="text-xl font-semibold">ABYSS</h1><p className="text-sm text-slate-400">Sonar detection</p></div></div>
      <div className={`flex items-center gap-2 border px-3 py-2 text-sm ${backendOnline === true ? "border-emerald-500 text-emerald-200" : "border-rose-500 text-rose-200"}`}>
        {backendOnline === true ? <CheckCircle2 size={16} /> : <ServerCrash size={16} />}
        {backendOnline === null ? "Checking backend" : backendOnline ? "Backend reachable" : "Backend unavailable"}
      </div>
    </div></header>
    <div className="mx-auto max-w-7xl space-y-4 px-4 py-5">
      {backendOnline === false && <div role="alert" className="flex gap-2 border border-rose-500 bg-rose-950 p-3 text-sm text-rose-100"><AlertTriangle size={18} />Backend unavailable. Detection and priority requests cannot complete.</div>}
      <div className="grid gap-4 lg:grid-cols-[minmax(0,1.35fr)_minmax(300px,0.65fr)]">
        <section className="border border-slate-700 bg-slate-950 p-4">
          <form onSubmit={submit} className="flex flex-wrap items-center gap-3">
            <label className="flex min-w-0 flex-1 items-center gap-2 border border-slate-700 bg-slate-900 p-2"><Upload size={17} className="shrink-0 text-cyan-300" /><input type="file" accept="image/*" onChange={chooseFile} className="min-w-0 flex-1 text-sm file:mr-2 file:border-0 file:bg-cyan-900 file:px-2 file:py-1 file:text-white" /></label>
            <button type="submit" disabled={!file || busy} className="flex items-center gap-2 border border-cyan-500 bg-cyan-950 px-4 py-2 text-sm font-semibold disabled:cursor-not-allowed disabled:border-slate-700 disabled:text-slate-500">
              {busy ? <Loader2 size={17} className="animate-spin" /> : <Crosshair size={17} />}Detect
            </button>
          </form>
          {error && <div role="alert" className="mt-3 flex items-start gap-2 border border-rose-500 bg-rose-950 p-3 text-sm text-rose-100"><AlertTriangle size={18} className="shrink-0" />{error}</div>}
          <div className="mt-4"><ImageOverlay imageUrl={imageUrl} response={response} selectedId={selectedId} onSelect={setSelectedId} /></div>
          <div className="mt-3 flex flex-wrap gap-x-6 gap-y-1 text-sm text-slate-300">
            <span>File: {response?.filename ?? file?.name ?? "None"}</span>
            <span>Detections: {response ? response.detections.length : "Not run"}</span>
            <span>Backend latency: {response ? `${response.processing_latency_ms.toFixed(1)} ms` : "Not run"}</span>
          </div>
        </section>
        <ConfidenceBreakdown detection={selected} />
      </div>
      <PriorityList items={priorities} detections={detections} />
    </div>
  </main>;
}
