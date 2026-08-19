import {
  Activity,
  Brain,
  Camera,
  CheckCircle2,
  Clock3,
  Cpu,
  Gauge,
  Image as ImageIcon,
  Layers3,
  Radio,
  ScanLine,
  TriangleAlert,
  Wifi,
  WifiOff,
  XCircle
} from "lucide-react";
import { useEffect, useMemo, useRef, useState } from "react";
import { getTwinState, mediaUrl, twinSocketUrl } from "./api";
import TwinScene3D from "./TwinScene3D";
import type { TwinAsset, TwinInspectionEvent, TwinSnapshot } from "./types";

type SocketState = "connecting" | "live" | "reconnecting";

const visibleAssetIds = ["FEEDER_01", "CAMERA_01", "INSPECTION_01", "AI_01", "CONVEYOR_01", "ACTUATOR_REJECT_01"];
const stationLabels = ["Feeder", "Camera", "Inspection", "AI decision", "Sort gate", "Verified bins"];

function titleize(value: string): string {
  return value.replace(/[._]/g, " ").replace(/\b\w/g, (letter) => letter.toUpperCase());
}

function eventTime(value: string): string {
  return new Intl.DateTimeFormat(undefined, {
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit"
  }).format(new Date(value));
}

function metricNumber(asset: TwinAsset | undefined, key: string): number {
  const value = asset?.metrics[key];
  return typeof value === "number" ? value : 0;
}

function AssetState({ asset }: { asset?: TwinAsset }) {
  return (
    <div className={`twin-asset-state state-${asset?.status ?? "offline"}`}>
      <span className="state-dot" />
      <div>
        <strong>{asset?.name ?? "Asset unavailable"}</strong>
        <small>{titleize(asset?.status ?? "offline")}</small>
      </div>
    </div>
  );
}

function DecisionIcon({ decision }: { decision?: string | null }) {
  if (decision === "reject") return <XCircle size={18} />;
  if (decision === "review") return <TriangleAlert size={18} />;
  if (decision === "pass") return <CheckCircle2 size={18} />;
  return <ScanLine size={18} />;
}

export default function DigitalTwin() {
  const [snapshot, setSnapshot] = useState<TwinSnapshot | null>(null);
  const [socketState, setSocketState] = useState<SocketState>("connecting");
  const [selectedEventId, setSelectedEventId] = useState("");
  const newestEventId = useRef("");

  useEffect(() => {
    let socket: WebSocket | null = null;
    let retryTimer: number | undefined;
    let stopped = false;

    getTwinState().then(setSnapshot).catch(() => undefined);

    const connect = () => {
      if (stopped) return;
      setSocketState((current) => (current === "connecting" ? "connecting" : "reconnecting"));
      socket = new WebSocket(twinSocketUrl());
      socket.onopen = () => setSocketState("live");
      socket.onmessage = (message) => {
        const next = JSON.parse(message.data) as TwinSnapshot;
        setSnapshot(next);
        if (next.last_event?.event_id && next.last_event.event_id !== newestEventId.current) {
          newestEventId.current = next.last_event.event_id;
          setSelectedEventId(next.last_event.event_id);
        }
      };
      socket.onerror = () => socket?.close();
      socket.onclose = () => {
        if (stopped) return;
        setSocketState("reconnecting");
        retryTimer = window.setTimeout(connect, 1400);
      };
    };

    connect();
    return () => {
      stopped = true;
      if (retryTimer) window.clearTimeout(retryTimer);
      socket?.close();
    };
  }, []);

  const assets = useMemo(
    () => new Map((snapshot?.assets ?? []).map((asset) => [asset.asset_id, asset])),
    [snapshot?.assets]
  );

  if (!snapshot) {
    return (
      <section id="machine" className="twin-section">
        <div className="twin-loading">
          <Radio size={22} />
          Connecting to inspection line...
        </div>
      </section>
    );
  }

  const cycle = snapshot.active_cycle;
  const selectedEvent =
    snapshot.timeline.find((event) => event.event_id === selectedEventId) ?? snapshot.last_event ?? snapshot.timeline[0];
  const operatorImage = cycle
    ? cycle.decision
      ? cycle.evidence_image_url
      : cycle.source_image_url
    : selectedEvent?.evidence_image_url ?? selectedEvent?.source_image_url;
  const operatorDecision = cycle?.decision ?? (!cycle ? selectedEvent?.decision : null);
  const operatorDefect = cycle?.defect_type ?? (!cycle ? selectedEvent?.defect_type : null);
  const operatorSource = cycle?.source_name ?? selectedEvent?.source_name ?? "No active frame";
  const activeBatchSize = cycle?.batch_size ?? selectedEvent?.batch_size ?? 0;
  const activeBatchId = cycle?.batch_id ?? selectedEvent?.batch_id ?? snapshot.batch.batch_id;
  const completedBatchEvents = new Map(
    snapshot.timeline
      .filter((event) => event.batch_id === activeBatchId && event.batch_position)
      .map((event) => [event.batch_position as number, event])
  );
  const rejectBin = assets.get("BIN_REJECT_01");

  return (
    <section id="machine" className="twin-section ops-twin-section">
      <div className="twin-heading">
        <div>
          <p className="eyebrow">Backend-synchronized production state</p>
          <h2>Live Inspection Line Digital Twin</h2>
          <span>
            {snapshot.plant.name} / {snapshot.inspection_line.name}
          </span>
        </div>
        <div className={`socket-state ${socketState}`}>
          {socketState === "live" ? <Wifi size={16} /> : <WifiOff size={16} />}
          {titleize(socketState)}
        </div>
      </div>

      <div className="twin-context-strip">
        <div>
          <span>Current batch</span>
          <strong>{activeBatchId}</strong>
        </div>
        <div>
          <span>Recipe</span>
          <strong>{snapshot.recipe.recipe_id}</strong>
        </div>
        <div>
          <span>AI model</span>
          <strong>{snapshot.model.status}</strong>
        </div>
        <div>
          <span>Line mode</span>
          <strong>{snapshot.inspection_line.operating_mode}</strong>
        </div>
        <div>
          <span>Inspected</span>
          <strong>{snapshot.batch.inspected_count.toLocaleString()}</strong>
        </div>
      </div>

      <div className="twin-status-strip" aria-label="Live asset states">
        {visibleAssetIds.map((assetId) => (
          <AssetState key={assetId} asset={assets.get(assetId)} />
        ))}
      </div>

      <div className="ops-twin-layout">
        <div className="ops-twin-visual">
          <TwinScene3D snapshot={snapshot} operatorImageUrl={operatorImage} />

          <div className="ops-scene-topline">
            <div>
              <span>LINE 01</span>
              <strong>{cycle ? `Image ${cycle.batch_position} of ${cycle.batch_size}` : "Line ready"}</strong>
            </div>
            <div className={`ops-phase phase-${cycle?.phase ?? "idle"}`}>
              <Activity size={14} />
              {cycle ? titleize(cycle.phase) : "Standing by"}
            </div>
          </div>

          <div className="ops-scene-telemetry">
            <span>
              <Camera size={14} /> 24 FPS
            </span>
            <span>
              <Layers3 size={14} /> {snapshot.recipe.inspection_sides} views
            </span>
            <span>
              <Brain size={14} /> {snapshot.model.model_kind}
            </span>
          </div>

          <div className="ops-station-key" aria-label="Inspection station order">
            {stationLabels.map((label, index) => (
              <span key={label} className={cycle && Math.floor((cycle.progress_pct / 100) * stationLabels.length) === index ? "active" : ""}>
                <i>{index + 1}</i>
                {label}
              </span>
            ))}
          </div>
        </div>

        <aside className="ops-console">
          <div className="ops-console-header">
            <div>
              <p className="eyebrow">HMI / operator panel</p>
              <h3>Inspection Console</h3>
            </div>
            <span className="ops-recording">
              <i />
              Live
            </span>
          </div>

          <div className="ops-console-screen">
            {operatorImage ? (
              <img src={mediaUrl(operatorImage)} alt={`Inspection frame for ${operatorSource}`} />
            ) : (
              <div className="ops-screen-empty">
                <ImageIcon size={24} />
                Waiting for camera frame
              </div>
            )}
            <span className="screen-corner top-left" />
            <span className="screen-corner top-right" />
            <span className="screen-corner bottom-left" />
            <span className="screen-corner bottom-right" />
            <div className={`ops-screen-decision ${operatorDecision ?? "processing"}`}>
              <DecisionIcon decision={operatorDecision} />
              {operatorDecision ? titleize(operatorDecision) : cycle ? titleize(cycle.status_message) : "Last inspection"}
            </div>
          </div>

          <div className="ops-current-record">
            <div>
              <span>Current frame</span>
              <strong title={operatorSource}>{operatorSource}</strong>
            </div>
            <div>
              <span>Score / threshold</span>
              <strong>{cycle ? `${cycle.score} / ${cycle.threshold}` : selectedEvent?.score ?? "n/a"}</strong>
            </div>
            <div>
              <span>Finding</span>
              <strong>{operatorDefect ? titleize(operatorDefect) : operatorDecision === "pass" ? "No defect" : "Pending"}</strong>
            </div>
            <div>
              <span>Queue</span>
              <strong>{cycle ? `${cycle.queue_depth} remaining` : "Clear"}</strong>
            </div>
          </div>

          <div className="ops-cycle-progress">
            <div>
              <span>{cycle?.status_message ?? "Waiting for next product"}</span>
              <strong>{cycle ? `${Math.round(cycle.progress_pct)}%` : "Ready"}</strong>
            </div>
            <span className="ops-progress-track">
              <i style={{ width: `${cycle?.progress_pct ?? 0}%` }} />
            </span>
          </div>

          {activeBatchSize > 1 && (
            <div className="ops-batch-sequence">
              <div className="ops-subheading">
                <span>Batch sequence</span>
                <strong>{activeBatchSize} images</strong>
              </div>
              <div className="ops-batch-slots">
                {Array.from({ length: activeBatchSize }, (_, index) => {
                  const position = index + 1;
                  const event = completedBatchEvents.get(position);
                  const isCurrent = cycle?.batch_position === position;
                  const status = event?.decision ?? (isCurrent ? "processing" : position < (cycle?.batch_position ?? 0) ? "complete" : "queued");
                  return (
                    <div key={position} className={`ops-batch-slot ${status} ${isCurrent ? "current" : ""}`}>
                      <span>{position}</span>
                      <strong>{event?.decision ? titleize(event.decision) : isCurrent ? titleize(cycle?.phase ?? "processing") : titleize(status)}</strong>
                    </div>
                  );
                })}
              </div>
            </div>
          )}

          <div className="ops-sort-counters">
            <div className="good">
              <span>Good</span>
              <strong>{snapshot.counters.good}</strong>
            </div>
            <div className="review">
              <span>Review</span>
              <strong>{snapshot.counters.review}</strong>
            </div>
            <div className="reject">
              <span>Reject</span>
              <strong>{snapshot.counters.reject}</strong>
              <small>{metricNumber(rejectBin, "fill_pct")}% full</small>
            </div>
          </div>

          <div className="ops-event-log">
            <div className="ops-subheading">
              <span>Event timeline</span>
              <Clock3 size={15} />
            </div>
            <div className="ops-event-list">
              {snapshot.timeline.slice(0, 6).map((event: TwinInspectionEvent) => (
                <button
                  key={event.event_id}
                  className={`ops-event ${event.decision ?? "system"} ${selectedEvent?.event_id === event.event_id ? "selected" : ""}`}
                  onClick={() => setSelectedEventId(event.event_id)}
                >
                  <span className="ops-event-marker" />
                  <div>
                    <strong>
                      {event.batch_position ? `Image ${event.batch_position}: ` : ""}
                      {titleize(event.event_type)}
                    </strong>
                    <span>{event.message}</span>
                    <small>{eventTime(event.created_at)}</small>
                  </div>
                </button>
              ))}
            </div>
          </div>

          <div className="ops-model-footer">
            <Cpu size={16} />
            <span>{snapshot.model.model_kind}</span>
            <Gauge size={16} />
            <span>Threshold {snapshot.model.threshold ?? "adaptive"}</span>
          </div>
        </aside>
      </div>
    </section>
  );
}
