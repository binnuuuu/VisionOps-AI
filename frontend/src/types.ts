export type Product = {
  id: string;
  name: string;
  sku?: string | null;
  product_family: "round_tablet" | "oblong_tablet" | "capsule" | "softgel" | "blister_pack";
  shape: string;
  diameter_mm?: number | null;
  length_mm?: number | null;
  width_mm?: number | null;
  height_mm?: number | null;
  cavity_rows: number;
  cavity_cols: number;
  inspection_sides: number;
  inspection_channels: string[];
  sorting_mode: string;
  inspection_mode: string;
  notes?: string | null;
  status: "draft" | "ready";
  good_sample_count: number;
  threshold?: number | null;
  model_path?: string | null;
  active_model_version_id?: string | null;
  model_kind?: string | null;
  approved_model_count: number;
  created_at: string;
  updated_at: string;
};

export type DefectRegion = {
  cavity: number;
  row: number;
  col: number;
  score: number;
  bbox: number[];
  defect_type: string;
  severity: string;
  confidence: number;
  area_pct: number;
};

export type Inspection = {
  id: string;
  product_id: string;
  product_name: string;
  source_name: string;
  batch_id?: string | null;
  model_version_id?: string | null;
  decision: "pass" | "review" | "reject";
  score: number;
  threshold: number;
  defect_regions: DefectRegion[];
  image_url: string;
  heatmap_url: string;
  created_at: string;
  duration_ms: number;
  timings_ms: Record<string, number>;
  model_cache_hit: boolean;
};

export type DatasetEntry = {
  id: string;
  name: string;
  priority: number;
  task: string;
  domain_fit: string;
  relevant_categories: string[];
  image_count?: number;
  license: string;
  source_url: string;
  download_method: string;
  notes: string;
};

export type SampleInspectionRecord = {
  id: string;
  pack_id: string;
  label: "normal" | "anomaly" | string;
  display_name: string;
  defect_type?: string | null;
  source?: string | null;
  image_url: string;
  index: number;
};

export type SampleInspectionPack = {
  pack_id: string;
  name: string;
  category: string;
  source_dataset?: string | null;
  license?: string | null;
  counts: Record<string, number>;
  records: SampleInspectionRecord[];
};

export type TrainingRun = {
  id: string;
  run_name: string;
  source: string;
  category?: string | null;
  created_at?: string | null;
  model_path?: string | null;
  threshold?: number | null;
  normal_images?: number | null;
  memory_patches?: number | null;
  metrics?: Record<string, number | string | null> | null;
};

export type ModelVersion = {
  id: string;
  product_id?: string | null;
  product_name?: string | null;
  version: number;
  model_kind: string;
  status: "candidate" | "approved" | "active" | "archived";
  model_path: string;
  threshold?: number | null;
  training_samples?: number | null;
  metrics?: Record<string, number | string | null> | null;
  created_at: string;
  approved_at?: string | null;
  approved_by?: string | null;
  activated_at?: string | null;
  notes?: string | null;
};

export type AuditEvent = {
  id: string;
  event_type: string;
  message: string;
  product_id?: string | null;
  inspection_id?: string | null;
  model_version_id?: string | null;
  details: Record<string, unknown>;
  created_at: string;
};

export type BatchReport = {
  product_id?: string | null;
  product_name?: string | null;
  total_inspections: number;
  pass_count: number;
  review_count: number;
  reject_count: number;
  reject_rate: number;
  avg_score: number;
  avg_duration_ms: number;
  p95_duration_ms: number;
  defect_type_counts: Record<string, number>;
  generated_at: string;
};

export type PerformanceStats = {
  total_inspections: number;
  avg_duration_ms: number;
  p50_duration_ms: number;
  p95_duration_ms: number;
  avg_inference_ms: number;
  cache_hit_rate: number;
  decision_counts: Record<string, number>;
  target_latency_ms: number;
};

export type AssistantAnswer = {
  answer: string;
  intent: string;
  context: {
    batch_id?: string | null;
    product_id?: string | null;
    inspection_id?: string | null;
    twin_sequence?: number;
    data_updated_at?: string;
  };
  suggested_questions: string[];
  generated_at: string;
};

export type TwinAssetType =
  | "feeder"
  | "camera"
  | "inspection_unit"
  | "ai_decision_unit"
  | "conveyor"
  | "actuator"
  | "sorting_bin"
  | "operator_panel";

export type TwinAsset = {
  asset_id: string;
  line_id: string;
  asset_type: TwinAssetType;
  name: string;
  status: string;
  current_recipe?: string | null;
  metrics: Record<string, number | string | boolean | string[] | null>;
};

export type TwinInspectionEvent = {
  event_id: string;
  event_type: string;
  message: string;
  created_at: string;
  decision?: "pass" | "review" | "reject" | null;
  asset_id?: string | null;
  inspection_id?: string | null;
  defect_type?: string | null;
  score?: number | null;
  evidence_image_url?: string | null;
  source_image_url?: string | null;
  source_name?: string | null;
  batch_id?: string | null;
  batch_position?: number | null;
  batch_size?: number | null;
};

export type TwinSnapshot = {
  sequence: number;
  updated_at: string;
  connection_state: string;
  plant: {
    plant_id: string;
    name: string;
    status: string;
    location: string;
  };
  inspection_line: {
    line_id: string;
    plant_id: string;
    name: string;
    status: string;
    operating_mode: string;
  };
  assets: TwinAsset[];
  recipe: {
    recipe_id: string;
    name: string;
    product_family: string;
    inspection_sides: number;
    inspection_channels: string[];
  };
  batch: {
    batch_id: string;
    status: string;
    inspected_count: number;
    good_count: number;
    review_count: number;
    reject_count: number;
  };
  model: {
    model_id: string;
    name: string;
    status: string;
    model_kind: string;
    threshold?: number | null;
  };
  counters: {
    good: number;
    review: number;
    reject: number;
  };
  route_state: string;
  pulse_active: boolean;
  active_cycle?: {
    inspection_id: string;
    source_name: string;
    product_name: string;
    batch_id: string;
    batch_position: number;
    batch_size: number;
    queue_depth: number;
    phase: string;
    status_message: string;
    progress_pct: number;
    decision?: "pass" | "review" | "reject" | null;
    score: number;
    threshold: number;
    defect_type?: string | null;
    source_image_url: string;
    evidence_image_url: string;
    started_at: string;
  } | null;
  last_event?: TwinInspectionEvent | null;
  timeline: TwinInspectionEvent[];
};
