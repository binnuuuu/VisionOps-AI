import {
  Activity,
  AlertTriangle,
  Brain,
  Camera,
  ChevronLeft,
  ChevronRight,
  CheckCircle2,
  ClipboardList,
  Cpu,
  Database,
  FileText,
  Gauge,
  Images,
  ListChecks,
  PackageCheck,
  PackagePlus,
  Play,
  RefreshCcw,
  ShieldCheck,
  Timer,
  Upload,
  Zap,
  XCircle
} from "lucide-react";
import { FormEvent, useEffect, useMemo, useState } from "react";
import {
  activateModelVersion,
  approveModelVersion,
  createProduct,
  getAuditEvents,
  getBatchReport,
  getDatasets,
  getInspections,
  getModelVersions,
  getPerformanceStats,
  getProducts,
  getSampleInspections,
  getTrainingRuns,
  inspectBatch,
  inspectProduct,
  inspectSampleBatch,
  mediaUrl,
  teachProduct
} from "./api";
import DigitalTwin from "./DigitalTwin";
import DashboardAssistant from "./DashboardAssistant";
import type {
  AuditEvent,
  BatchReport,
  DatasetEntry,
  Inspection,
  ModelVersion,
  PerformanceStats,
  Product,
  SampleInspectionPack,
  SampleInspectionRecord,
  TrainingRun
} from "./types";

type ProductFamily = Product["product_family"];

const decisionCopy = {
  pass: { label: "Good", icon: CheckCircle2 },
  review: { label: "Review", icon: AlertTriangle },
  reject: { label: "Reject", icon: XCircle }
};

const productFamilies: Array<{ value: ProductFamily; label: string; shape: string; defaultRows: number; defaultCols: number }> = [
  { value: "capsule", label: "Capsule", shape: "capsule", defaultRows: 1, defaultCols: 1 },
  { value: "round_tablet", label: "Round tablet", shape: "round", defaultRows: 1, defaultCols: 1 },
  { value: "oblong_tablet", label: "Oval / oblong tablet", shape: "oval_oblong", defaultRows: 1, defaultCols: 1 },
  { value: "softgel", label: "Softgel", shape: "softgel", defaultRows: 1, defaultCols: 1 },
  { value: "blister_pack", label: "Blister pack", shape: "blister_grid", defaultRows: 4, defaultCols: 3 }
];

const familyLabels: Record<ProductFamily, string> = Object.fromEntries(
  productFamilies.map((family) => [family.value, family.label])
) as Record<ProductFamily, string>;

const defectCatalog: Record<ProductFamily, string[]> = {
  round_tablet: [
    "bright dots",
    "dark dots",
    "cracks",
    "colour deviation",
    "chips",
    "edge dark dots",
    "engraving defects",
    "coating defects"
  ],
  oblong_tablet: ["dots", "coating defects", "cracks", "lamination", "chips", "print defects", "engraving defects"],
  capsule: ["dents", "tip defect", "length mismatch", "joint defect", "holes", "double cap", "print defects"],
  softgel: ["twins", "wrong size", "damaged seam", "dirt", "hair", "bubbles", "broken shell"],
  blister_pack: ["missing tablet", "broken tablet", "empty pocket", "foil tear", "foreign particle", "wrong colour"]
};

const channelLabels: Record<string, string> = {
  colour: "Colour",
  color: "Colour",
  backlight: "Backlight",
  "3d": "3D",
  brightfield: "Brightfield",
  darkfield: "Darkfield"
};

function labelize(value?: string | null): string {
  if (!value) return "none";
  if (value === "visual_anomaly") return "unclassified deviation";
  if (value === "broken_capsule_or_leak") return "broken capsule / leak";
  if (value === "hair_or_fiber_contamination") return "hair / fiber contamination";
  if (value === "bubble_or_fill_void") return "bubble / fill void";
  if (value === "surface_particle_or_stain") return "surface particle / stain";
  if (value === "capsule_surface_deviation") return "capsule surface deviation";
  return value.replace(/_/g, " ");
}

function formatMs(value?: number | null): string {
  if (value === undefined || value === null) return "0 ms";
  return `${Math.round(value)} ms`;
}

function percent(value?: number | null): string {
  if (value === undefined || value === null) return "0%";
  return `${Math.round(value * 100)}%`;
}

function shortId(value?: string | null): string {
  return value ? value.slice(0, 8) : "none";
}

function regionTitle(cavity: number): string {
  return cavity > 0 ? `Pocket ${cavity}` : "Product surface";
}

function usesObjectAnomalyModel(product?: Product): boolean {
  return product?.model_kind?.startsWith("patchcore-lite") ?? false;
}

function inferProductFamily(product?: Product): ProductFamily {
  if (!product) return "capsule";
  if (usesObjectAnomalyModel(product) || /capsule/i.test(product.name)) return "capsule";
  return product.product_family ?? "blister_pack";
}

function channelList(product?: Product): string[] {
  return product?.inspection_channels?.length ? product.inspection_channels : ["colour", "backlight", "3d"];
}

function recipeLayoutLabel(product: Product): string {
  const family = inferProductFamily(product);
  if (family === "blister_pack") return `${product.cavity_rows} x ${product.cavity_cols} blister grid`;
  return `${product.inspection_sides || 6} side loose-product inspection`;
}

function dimensionSummary(product?: Product): string {
  if (!product) return "no recipe selected";
  const family = inferProductFamily(product);
  const dims = [
    product.diameter_mm ? `diameter ${product.diameter_mm} mm` : "",
    product.length_mm ? `L ${product.length_mm} mm` : "",
    product.width_mm ? `W ${product.width_mm} mm` : "",
    product.height_mm ? `H ${product.height_mm} mm` : ""
  ].filter(Boolean);
  if (dims.length) return dims.join(" · ");
  return family === "round_tablet" ? "diameter and height pending" : "shape constraints pending";
}

function fileLooksLikeCapsule(file?: File | null): boolean {
  return /capsule/i.test(file?.name ?? "");
}

function filesLookLikeCapsules(files?: FileList | null): boolean {
  return Array.from(files ?? []).some(fileLooksLikeCapsule);
}

function App() {
  const [products, setProducts] = useState<Product[]>([]);
  const [inspections, setInspections] = useState<Inspection[]>([]);
  const [datasets, setDatasets] = useState<DatasetEntry[]>([]);
  const [samplePack, setSamplePack] = useState<SampleInspectionPack | null>(null);
  const [selectedSampleIds, setSelectedSampleIds] = useState<string[]>([]);
  const [sampleFilter, setSampleFilter] = useState<"all" | "normal" | "anomaly">("all");
  const [trainingRuns, setTrainingRuns] = useState<TrainingRun[]>([]);
  const [modelVersions, setModelVersions] = useState<ModelVersion[]>([]);
  const [auditEvents, setAuditEvents] = useState<AuditEvent[]>([]);
  const [batchReport, setBatchReport] = useState<BatchReport | null>(null);
  const [performance, setPerformance] = useState<PerformanceStats | null>(null);
  const [selectedProductId, setSelectedProductId] = useState("");
  const [teachProductId, setTeachProductId] = useState("");
  const [teachFiles, setTeachFiles] = useState<FileList | null>(null);
  const [inspectFile, setInspectFile] = useState<File | null>(null);
  const [batchFiles, setBatchFiles] = useState<FileList | null>(null);
  const [batchResults, setBatchResults] = useState<Inspection[]>([]);
  const [batchIndex, setBatchIndex] = useState(0);
  const [latest, setLatest] = useState<Inspection | null>(null);
  const [busy, setBusy] = useState("");
  const [notice, setNotice] = useState("");

  async function refresh(reportProductId = selectedProductId) {
    const [
      nextProducts,
      nextInspections,
      nextDatasets,
      nextSamplePack,
      nextTrainingRuns,
      nextModelVersions,
      nextAuditEvents,
      nextBatchReport,
      nextPerformance
    ] = await Promise.all([
      getProducts(),
      getInspections(),
      getDatasets(),
      getSampleInspections(),
      getTrainingRuns(),
      getModelVersions(),
      getAuditEvents(),
      getBatchReport(reportProductId || undefined),
      getPerformanceStats()
    ]);
    setProducts(nextProducts);
    setInspections(nextInspections);
    setDatasets(nextDatasets);
    setSamplePack(nextSamplePack);
    setTrainingRuns(nextTrainingRuns);
    setModelVersions(nextModelVersions);
    setAuditEvents(nextAuditEvents);
    setBatchReport(nextBatchReport);
    setPerformance(nextPerformance);

    const preferredReady =
      nextProducts.find((product) => product.status === "ready" && inferProductFamily(product) === "capsule") ??
      nextProducts.find((product) => product.status === "ready");
    setSelectedProductId((current) => current || preferredReady?.id || nextProducts[0]?.id || "");
    setTeachProductId((current) => current || preferredReady?.id || nextProducts[0]?.id || "");
    setLatest((current) => {
      if (!current) return nextInspections[0] || null;
      return nextInspections.find((inspection) => inspection.id === current.id) || current;
    });
  }

  useEffect(() => {
    refresh().catch((error) => setNotice(error.message));
  }, []);

  useEffect(() => {
    getBatchReport(selectedProductId || undefined)
      .then(setBatchReport)
      .catch(() => undefined);
  }, [selectedProductId]);

  const readyProducts = useMemo(() => products.filter((product) => product.status === "ready"), [products]);
  const selectedProduct = products.find((product) => product.id === selectedProductId);
  const selectedFamily = inferProductFamily(selectedProduct);
  const selectedChannels = channelList(selectedProduct);
  const selectedDefects = defectCatalog[selectedFamily];
  const surfaceImageCount = (selectedProduct?.inspection_sides || (selectedFamily === "blister_pack" ? 1 : 6)) * selectedChannels.length;
  const activeModel = modelVersions.find((version) => version.id === selectedProduct?.active_model_version_id);
  const capsuleProduct = useMemo(
    () => products.find((product) => product.status === "ready" && inferProductFamily(product) === "capsule"),
    [products]
  );
  const displayedInspection = batchResults[batchIndex] ?? latest;
  const latestDecision = displayedInspection ? decisionCopy[displayedInspection.decision] : null;
  const LatestIcon = latestDecision?.icon ?? Activity;
  const defectTypeCounts = Object.entries(batchReport?.defect_type_counts ?? {}).sort((a, b) => b[1] - a[1]);
  const latencyOnTarget = performance ? performance.p95_duration_ms <= performance.target_latency_ms : false;
  const hasBatchReview = batchResults.length > 1;
  const selectedBatchPosition = batchResults.length ? batchIndex + 1 : 0;
  const sampleRecords = samplePack?.records ?? [];
  const selectedSamples = sampleRecords.filter((sample) => selectedSampleIds.includes(sample.id));
  const visibleSamples = sampleRecords.filter((sample) => sampleFilter === "all" || sample.label === sampleFilter);
  const selectedSampleCounts = selectedSamples.reduce(
    (counts, sample) => ({
      normal: counts.normal + (sample.label === "normal" ? 1 : 0),
      anomaly: counts.anomaly + (sample.label === "anomaly" ? 1 : 0)
    }),
    { normal: 0, anomaly: 0 }
  );

  function showPreviousBatchImage() {
    if (!batchResults.length) return;
    setBatchIndex((current) => (current - 1 + batchResults.length) % batchResults.length);
  }

  function showNextBatchImage() {
    if (!batchResults.length) return;
    setBatchIndex((current) => (current + 1) % batchResults.length);
  }

  function scrollToDigitalTwin() {
    window.requestAnimationFrame(() => {
      document.getElementById("machine")?.scrollIntoView({ behavior: "smooth", block: "start" });
    });
  }

  function setDemoSelection(mode: "normal" | "anomaly" | "mixed") {
    const normal = sampleRecords.filter((sample) => sample.label === "normal").slice(0, 3);
    const anomaly = sampleRecords.filter((sample) => sample.label === "anomaly").slice(0, 5);
    const nextSamples = mode === "normal" ? normal : mode === "anomaly" ? anomaly : [...normal.slice(0, 2), ...anomaly.slice(0, 4)];
    setSampleFilter(mode === "mixed" ? "all" : mode);
    setSelectedSampleIds(nextSamples.map((sample) => sample.id));
  }

  function toggleSample(sample: SampleInspectionRecord) {
    setSelectedSampleIds((current) =>
      current.includes(sample.id) ? current.filter((id) => id !== sample.id) : [...current, sample.id]
    );
  }

  async function onCreateProduct(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const form = event.currentTarget;
    setBusy("Creating recipe");
    setNotice("");
    try {
      const product = await createProduct(form);
      form.reset();
      setNotice(`Created ${product.name}. Teach it with good samples to activate inspection.`);
      await refresh(product.id);
      setTeachProductId(product.id);
      setSelectedProductId(product.id);
    } catch (error) {
      setNotice(error instanceof Error ? error.message : "Could not create recipe.");
    } finally {
      setBusy("");
    }
  }

  async function onTeach() {
    if (!teachProductId || !teachFiles) {
      setNotice("Choose a product and upload at least 3 good samples.");
      return;
    }
    setBusy("Teaching recipe");
    setNotice("");
    try {
      const product = await teachProduct(teachProductId, teachFiles);
      setNotice(`${product.name} is ready with model ${product.active_model_version_id ? shortId(product.active_model_version_id) : "v1"}.`);
      await refresh(product.id);
      setSelectedProductId(product.id);
    } catch (error) {
      setNotice(error instanceof Error ? error.message : "Teaching failed.");
    } finally {
      setBusy("");
    }
  }

  async function onInspect() {
    if (!selectedProductId || !inspectFile) {
      setNotice("Choose a ready product and upload one inspection image.");
      return;
    }
    scrollToDigitalTwin();
    setBusy("Inspecting");
    setNotice("");
    try {
      let productId = selectedProductId;
      let routeNotice = "";
      if (fileLooksLikeCapsule(inspectFile) && selectedProduct && inferProductFamily(selectedProduct) === "blister_pack" && capsuleProduct) {
        productId = capsuleProduct.id;
        setSelectedProductId(capsuleProduct.id);
        routeNotice = `Switched to ${capsuleProduct.name}. `;
      }
      const inspection = await inspectProduct(productId, inspectFile);
      setBatchResults([]);
      setBatchIndex(0);
      setLatest(inspection);
      setNotice(`${routeNotice}${decisionCopy[inspection.decision].label} result saved for ${inspection.source_name}.`);
      await refresh(productId);
    } catch (error) {
      setNotice(error instanceof Error ? error.message : "Inspection failed.");
    } finally {
      setBusy("");
    }
  }

  async function onInspectBatch() {
    if (!selectedProductId || !batchFiles?.length) {
      setNotice("Choose a ready product and upload batch images.");
      return;
    }
    scrollToDigitalTwin();
    setBusy("Running batch");
    setNotice("");
    try {
      let productId = selectedProductId;
      let routeNotice = "";
      if (filesLookLikeCapsules(batchFiles) && selectedProduct && inferProductFamily(selectedProduct) === "blister_pack" && capsuleProduct) {
        productId = capsuleProduct.id;
        setSelectedProductId(capsuleProduct.id);
        routeNotice = `Switched to ${capsuleProduct.name}. `;
      }
      const results = await inspectBatch(productId, batchFiles);
      const firstReject = results.findIndex((inspection) => inspection.decision === "reject");
      const firstReview = results.findIndex((inspection) => inspection.decision === "review");
      const focusIndex = Math.max(0, firstReject >= 0 ? firstReject : firstReview);
      const nextFocus = results[focusIndex] ?? results[0] ?? null;
      setBatchResults(results);
      setBatchIndex(focusIndex);
      setLatest(nextFocus);
      const rejected = results.filter((inspection) => inspection.decision === "reject").length;
      const review = results.filter((inspection) => inspection.decision === "review").length;
      setNotice(`${routeNotice}Batch completed: ${results.length} images, ${rejected} reject, ${review} review.`);
      await refresh(productId);
    } catch (error) {
      setNotice(error instanceof Error ? error.message : "Batch inspection failed.");
    } finally {
      setBusy("");
    }
  }

  async function onRunSampleBatch() {
    if (!selectedProductId || !selectedSampleIds.length) {
      setNotice("Choose a ready product and select normal or anomaly demo samples.");
      return;
    }
    scrollToDigitalTwin();
    setBusy("Running sample demo");
    setNotice("");
    try {
      let productId = selectedProductId;
      let routeNotice = "";
      if (selectedProduct && inferProductFamily(selectedProduct) === "blister_pack" && capsuleProduct) {
        productId = capsuleProduct.id;
        setSelectedProductId(capsuleProduct.id);
        routeNotice = `Switched to ${capsuleProduct.name}. `;
      }
      const results = await inspectSampleBatch(productId, selectedSampleIds);
      const firstReject = results.findIndex((inspection) => inspection.decision === "reject");
      const firstReview = results.findIndex((inspection) => inspection.decision === "review");
      const focusIndex = Math.max(0, firstReject >= 0 ? firstReject : firstReview);
      const nextFocus = results[focusIndex] ?? results[0] ?? null;
      setBatchResults(results);
      setBatchIndex(focusIndex);
      setLatest(nextFocus);
      const passed = results.filter((inspection) => inspection.decision === "pass").length;
      const rejected = results.filter((inspection) => inspection.decision === "reject").length;
      setNotice(`${routeNotice}Sample demo completed: ${results.length} images, ${passed} good, ${rejected} reject.`);
      await refresh(productId);
    } catch (error) {
      setNotice(error instanceof Error ? error.message : "Sample demo failed.");
    } finally {
      setBusy("");
    }
  }

  async function onApproveModel(version: ModelVersion) {
    setBusy("Approving model");
    setNotice("");
    try {
      await approveModelVersion(version.id);
      setNotice(`Approved ${version.product_name ?? "model"} v${version.version}.`);
      await refresh(selectedProductId);
    } catch (error) {
      setNotice(error instanceof Error ? error.message : "Could not approve model.");
    } finally {
      setBusy("");
    }
  }

  async function onActivateModel(version: ModelVersion) {
    setBusy("Activating model");
    setNotice("");
    try {
      await activateModelVersion(version.id);
      setNotice(`Activated ${version.product_name ?? "model"} v${version.version}.`);
      await refresh(version.product_id ?? selectedProductId);
      if (version.product_id) setSelectedProductId(version.product_id);
    } catch (error) {
      setNotice(error instanceof Error ? error.message : "Could not activate model.");
    } finally {
      setBusy("");
    }
  }

  return (
    <main className="app-shell">
      <aside className="side-rail" aria-label="Inspection navigation">
        <div className="brand-mark">
          <ShieldCheck size={26} />
          <div>
            <strong>VisionOps</strong>
            <span>SPINE-style HMI</span>
          </div>
        </div>
        <nav>
          <a href="#machine">
            <Gauge size={18} /> Digital Twin
          </a>
          <a href="#inspection">
            <Camera size={18} /> Inspect
          </a>
          <a href="#teaching">
            <Brain size={18} /> Teach
          </a>
          <a href="#production">
            <Cpu size={18} /> Sorting
          </a>
          <a href="#registry">
            <ListChecks size={18} /> Models
          </a>
          <a href="#datasets">
            <Database size={18} /> Data
          </a>
        </nav>
      </aside>

      <section className="workspace">
        <header className="topbar">
          <div>
            <p className="eyebrow">AI tablet, capsule, and softgel inspection cell</p>
            <h1>Vision Machine Control</h1>
          </div>
          <button className="icon-button" onClick={() => refresh()} title="Refresh station data">
            <RefreshCcw size={18} />
          </button>
        </header>

        {notice && <div className="notice">{notice}</div>}
        {busy && <div className="busy">{busy}...</div>}

        <section className="metric-grid" aria-label="Station status">
          <div className="metric">
            <span>Recipes</span>
            <strong>{products.length}</strong>
            <small>{readyProducts.length} ready</small>
          </div>
          <div className="metric">
            <span>Surface Images</span>
            <strong>{surfaceImageCount}</strong>
            <small>{selectedChannels.map((channel) => channelLabels[channel] ?? channel).join(" + ")}</small>
          </div>
          <div className="metric">
            <span>Model Versions</span>
            <strong>{modelVersions.length}</strong>
            <small>{modelVersions.filter((version) => version.status === "active").length} active</small>
          </div>
          <div className="metric">
            <span>Latency P95</span>
            <strong>
              <Timer size={22} />
              {formatMs(performance?.p95_duration_ms)}
            </strong>
            <small className={latencyOnTarget ? "ok-text" : "warn-text"}>target {formatMs(performance?.target_latency_ms)}</small>
          </div>
          <div className="metric metric-decision">
            <span>Last Sort</span>
            <strong>
              <LatestIcon size={24} />
              {latestDecision?.label ?? "Idle"}
            </strong>
            <small>{displayedInspection ? `score ${displayedInspection.score}` : "waiting for product"}</small>
          </div>
        </section>

        <DigitalTwin />

        <section id="inspection" className="main-grid">
          <div className="panel inspection-panel">
            <div className="panel-heading">
              <div>
                <p className="eyebrow">Runtime station</p>
                <h2>Inspect Product</h2>
              </div>
              <Activity size={22} />
            </div>

            <label>
              Product recipe
              <select value={selectedProductId} onChange={(event) => setSelectedProductId(event.target.value)}>
                <option value="">Select ready product</option>
                {products.map((product) => (
                  <option key={product.id} value={product.id} disabled={product.status !== "ready"}>
                    {product.name} {product.status !== "ready" ? "(teach first)" : ""}
                  </option>
                ))}
              </select>
            </label>

            {selectedProduct && (
              <div className="recipe-card">
                <div>
                  <strong>{selectedProduct.name}</strong>
                  <span>{familyLabels[selectedFamily]}</span>
                  <span>{recipeLayoutLabel(selectedProduct)}</span>
                  <span>{dimensionSummary(selectedProduct)}</span>
                </div>
                <span className={`status ${selectedProduct.status}`}>{selectedProduct.status}</span>
              </div>
            )}

            {activeModel && (
              <div className="model-chip">
                <Zap size={17} />
                <span>Active v{activeModel.version}</span>
                <strong>{activeModel.model_kind}</strong>
              </div>
            )}

            <label className="drop-zone">
              <Upload size={24} />
              <span>{inspectFile?.name ?? "Single product image or multi-view contact sheet"}</span>
              <input type="file" accept="image/*" onChange={(event) => setInspectFile(event.target.files?.[0] ?? null)} />
            </label>

            <button className="primary-action" onClick={onInspect} disabled={!!busy || !selectedProductId || !inspectFile}>
              <Play size={18} />
              Run Inspection
            </button>

            <div className="panel-divider" />

            <label className="drop-zone compact">
              <ClipboardList size={22} />
              <span>{batchFiles?.length ? `${batchFiles.length} product images selected` : "Inline batch image set"}</span>
              <input type="file" accept="image/*" multiple onChange={(event) => setBatchFiles(event.target.files)} />
            </label>

            <button className="secondary-action full-width" onClick={onInspectBatch} disabled={!!busy || !selectedProductId || !batchFiles?.length}>
              <ClipboardList size={18} />
              Run Batch
            </button>

            <div className="panel-divider" />

            <div className="sample-runner">
              <div className="sample-runner-head">
                <div>
                  <p className="eyebrow">Demo</p>
                  <h3>Sample Inspection Run</h3>
                </div>
                <Images size={20} />
              </div>

              <div className="sample-preset-row">
                <button className="secondary-action small-action" type="button" onClick={() => setDemoSelection("mixed")} disabled={!sampleRecords.length}>
                  Mixed
                </button>
                <button className="secondary-action small-action" type="button" onClick={() => setDemoSelection("normal")} disabled={!sampleRecords.length}>
                  Normal
                </button>
                <button className="secondary-action small-action" type="button" onClick={() => setDemoSelection("anomaly")} disabled={!sampleRecords.length}>
                  Anomaly
                </button>
              </div>

              <div className="sample-filter-row" role="tablist" aria-label="Sample filters">
                {(["all", "normal", "anomaly"] as const).map((filter) => (
                  <button
                    key={filter}
                    className={sampleFilter === filter ? "active" : ""}
                    type="button"
                    onClick={() => setSampleFilter(filter)}
                  >
                    {filter}
                  </button>
                ))}
              </div>

              <div className="sample-grid">
                {visibleSamples.map((sample) => {
                  const selected = selectedSampleIds.includes(sample.id);
                  return (
                    <button
                      key={sample.id}
                      className={`sample-card ${sample.label} ${selected ? "selected" : ""}`}
                      type="button"
                      onClick={() => toggleSample(sample)}
                      title={sample.display_name}
                    >
                      <img src={mediaUrl(sample.image_url)} alt={sample.display_name} />
                      <span>{sample.label}</span>
                      <strong>{sample.defect_type ? labelize(sample.defect_type) : sample.display_name}</strong>
                    </button>
                  );
                })}
                {!visibleSamples.length && <div className="empty-inline">No samples available.</div>}
              </div>

              <div className="sample-summary">
                <span>
                  Selected <strong>{selectedSampleIds.length}</strong>
                </span>
                <span>
                  Normal <strong>{selectedSampleCounts.normal}</strong>
                </span>
                <span>
                  Anomaly <strong>{selectedSampleCounts.anomaly}</strong>
                </span>
              </div>

              <button className="primary-action" onClick={onRunSampleBatch} disabled={!!busy || !selectedProductId || !selectedSampleIds.length}>
                <Play size={18} />
                Run Selected Samples
              </button>
            </div>
          </div>

          <div className="panel result-panel">
            <div className="panel-heading">
              <div>
                <p className="eyebrow">Visual output</p>
                <h2>Defect Heatmap</h2>
              </div>
              {displayedInspection && <span className={`decision-pill ${displayedInspection.decision}`}>{decisionCopy[displayedInspection.decision].label}</span>}
            </div>

            {displayedInspection ? (
              <>
                {hasBatchReview && (
                  <div className="batch-review-bar">
                    <button className="icon-button" onClick={showPreviousBatchImage} title="Previous batch image">
                      <ChevronLeft size={18} />
                    </button>
                    <div>
                      <strong>
                        Image {selectedBatchPosition} of {batchResults.length}
                      </strong>
                      <span>{displayedInspection.source_name}</span>
                    </div>
                    <button className="icon-button" onClick={showNextBatchImage} title="Next batch image">
                      <ChevronRight size={18} />
                    </button>
                  </div>
                )}

                <div className="image-stage">
                  {hasBatchReview && (
                    <button className="stage-nav stage-nav-left" onClick={showPreviousBatchImage} title="Previous batch image">
                      <ChevronLeft size={22} />
                    </button>
                  )}
                  <img src={mediaUrl(displayedInspection.heatmap_url)} alt="Inspection heatmap" />
                  {hasBatchReview && (
                    <button className="stage-nav stage-nav-right" onClick={showNextBatchImage} title="Next batch image">
                      <ChevronRight size={22} />
                    </button>
                  )}
                </div>
                {hasBatchReview && (
                  <div className="batch-result-strip" aria-label="Batch inspection results">
                    {batchResults.map((inspection, index) => (
                      <button
                        key={inspection.id}
                        className={`batch-result-chip ${inspection.decision} ${index === batchIndex ? "active" : ""}`}
                        onClick={() => setBatchIndex(index)}
                        title={inspection.source_name}
                      >
                        <span>{index + 1}</span>
                        <strong>{decisionCopy[inspection.decision].label}</strong>
                        <small>{inspection.source_name}</small>
                      </button>
                    ))}
                  </div>
                )}
                <div className="result-data">
                  <span>
                    Product: <strong>{displayedInspection.product_name}</strong>
                  </span>
                  <span>
                    Score: <strong>{displayedInspection.score}</strong>
                  </span>
                  <span>
                    Threshold: <strong>{displayedInspection.threshold}</strong>
                  </span>
                  <span>
                    Duration: <strong>{formatMs(displayedInspection.duration_ms)}</strong>
                  </span>
                  <span>
                    Batch: <strong>{displayedInspection.batch_id ?? "none"}</strong>
                  </span>
                  <span>
                    Model: <strong>{shortId(displayedInspection.model_version_id)}</strong>
                  </span>
                  <span>
                    Cache: <strong>{displayedInspection.model_cache_hit ? "hit" : "miss"}</strong>
                  </span>
                </div>
                <div className="timing-strip">
                  <span>load {formatMs(displayedInspection.timings_ms.model_load)}</span>
                  <span>preprocess {formatMs(displayedInspection.timings_ms.preprocess)}</span>
                  <span>inference {formatMs(displayedInspection.timings_ms.inference)}</span>
                  <span>localize {formatMs(displayedInspection.timings_ms.localization)}</span>
                  <span>heatmap {formatMs(displayedInspection.timings_ms.heatmap)}</span>
                </div>
                <div className="defect-list">
                  {displayedInspection.defect_regions.length ? (
                    displayedInspection.defect_regions.map((region) => (
                      <article key={`${displayedInspection.id}-${region.cavity}-${region.defect_type}`} className="defect-card">
                        <div>
                          <strong>{regionTitle(region.cavity)}</strong>
                          <span>{labelize(region.defect_type)}</span>
                        </div>
                        <span className={`severity ${region.severity}`}>{region.severity}</span>
                        <small>
                          score {region.score} · conf {Math.round(region.confidence * 100)}% · area {region.area_pct}%
                        </small>
                      </article>
                    ))
                  ) : (
                    <span>No confirmed defect above threshold.</span>
                  )}
                </div>
              </>
            ) : (
              <div className="empty-state">No inspection result selected.</div>
            )}
          </div>
        </section>

        <section className="section-grid">
          <div className="panel taxonomy-panel">
            <div className="panel-heading">
              <div>
                <p className="eyebrow">Defect library</p>
                <h2>{familyLabels[selectedFamily]} checks</h2>
              </div>
              <AlertTriangle size={22} />
            </div>
            <div className="taxonomy-grid">
              {selectedDefects.map((defect) => (
                <span key={defect}>{defect}</span>
              ))}
            </div>
          </div>

          <div className="panel sorter-panel">
            <div className="panel-heading">
              <div>
                <p className="eyebrow">Active sorting</p>
                <h2>Bin Verification</h2>
              </div>
              <PackageCheck size={22} />
            </div>
            <div className="sorter-bins">
              <div className="bin good-bin">
                <span>Good outlet</span>
                <strong>{batchReport?.pass_count ?? 0}</strong>
              </div>
              <div className="bin review-bin">
                <span>Review tray</span>
                <strong>{batchReport?.review_count ?? 0}</strong>
              </div>
              <div className="bin reject-bin">
                <span>Reject bin</span>
                <strong>{batchReport?.reject_count ?? 0}</strong>
              </div>
            </div>
          </div>
        </section>

        <section id="teaching" className="section-grid">
          <div className="panel">
            <div className="panel-heading">
              <div>
                <p className="eyebrow">Recipe setup</p>
                <h2>Add New Product</h2>
              </div>
              <PackagePlus size={22} />
            </div>
            <form className="form-grid" onSubmit={onCreateProduct}>
              <label>
                Product name
                <input name="name" placeholder="Blue oblong tablet 250 mg" required />
              </label>
              <label>
                SKU / batch code
                <input name="sku" placeholder="optional" />
              </label>
              <label>
                Product family
                <select name="product_family" defaultValue="capsule">
                  {productFamilies.map((family) => (
                    <option key={family.value} value={family.value}>
                      {family.label}
                    </option>
                  ))}
                </select>
              </label>
              <label>
                Shape tooling
                <select name="shape" defaultValue="capsule">
                  <option value="round">Round</option>
                  <option value="oval_oblong">Oval / oblong</option>
                  <option value="capsule">Capsule</option>
                  <option value="softgel">Softgel</option>
                  <option value="blister_grid">Blister grid</option>
                </select>
              </label>
              <label>
                Diameter mm
                <input name="diameter_mm" type="number" min="0" max="100" step="0.1" placeholder="round only" />
              </label>
              <label>
                Length mm
                <input name="length_mm" type="number" min="0" max="100" step="0.1" placeholder="25" />
              </label>
              <label>
                Width mm
                <input name="width_mm" type="number" min="0" max="100" step="0.1" placeholder="8" />
              </label>
              <label>
                Height mm
                <input name="height_mm" type="number" min="0" max="100" step="0.1" placeholder="6" />
              </label>
              <label>
                Pockets / rows
                <input name="cavity_rows" type="number" min="1" max="20" defaultValue="1" required />
              </label>
              <label>
                Pockets / columns
                <input name="cavity_cols" type="number" min="1" max="20" defaultValue="1" required />
              </label>
              <label>
                Inspection sides
                <input name="inspection_sides" type="number" min="1" max="6" defaultValue="6" required />
              </label>
              <label>
                Optical stack
                <select name="inspection_channels" defaultValue="colour,backlight,3d">
                  <option value="colour,backlight,3d">Colour + backlight + 3D</option>
                  <option value="colour,backlight">Colour + backlight</option>
                  <option value="colour,brightfield,darkfield">Colour + brightfield + darkfield</option>
                  <option value="colour">Colour only</option>
                </select>
              </label>
              <label>
                Sorting mode
                <select name="sorting_mode" defaultValue="active_sorting_with_verification">
                  <option value="active_sorting_with_verification">Active sorting with verification</option>
                  <option value="software_reject_queue">Software reject queue</option>
                  <option value="inspection_only">Inspection only</option>
                </select>
              </label>
              <label>
                Notes
                <input name="notes" placeholder="lighting, tooling, batch notes" />
              </label>
              <button className="secondary-action" type="submit" disabled={!!busy}>
                <PackagePlus size={18} />
                Create Recipe
              </button>
            </form>
          </div>

          <div className="panel">
            <div className="panel-heading">
              <div>
                <p className="eyebrow">Auto teaching</p>
                <h2>Good Sample Learning</h2>
              </div>
              <Brain size={22} />
            </div>
            <label>
              Product recipe
              <select value={teachProductId} onChange={(event) => setTeachProductId(event.target.value)}>
                <option value="">Select product</option>
                {products.map((product) => (
                  <option key={product.id} value={product.id}>
                    {product.name}
                  </option>
                ))}
              </select>
            </label>
            <label className="drop-zone compact">
              <Upload size={22} />
              <span>{teachFiles?.length ? `${teachFiles.length} good samples selected` : "3+ good product images"}</span>
              <input type="file" accept="image/*" multiple onChange={(event) => setTeachFiles(event.target.files)} />
            </label>
            <button className="primary-action" onClick={onTeach} disabled={!!busy || !teachProductId || !teachFiles}>
              <Brain size={18} />
              Teach Product
            </button>
          </div>
        </section>

        <section id="production" className="section-grid">
          <div className="panel telemetry-panel">
            <div className="panel-heading">
              <div>
                <p className="eyebrow">Performance</p>
                <h2>Runtime Telemetry</h2>
              </div>
              <Cpu size={22} />
            </div>
            <div className="diagnostic-grid">
              <div>
                <span>P50</span>
                <strong>{formatMs(performance?.p50_duration_ms)}</strong>
              </div>
              <div>
                <span>P95</span>
                <strong>{formatMs(performance?.p95_duration_ms)}</strong>
              </div>
              <div>
                <span>Inference</span>
                <strong>{formatMs(performance?.avg_inference_ms)}</strong>
              </div>
              <div>
                <span>Cache</span>
                <strong>{percent(performance?.cache_hit_rate)}</strong>
              </div>
            </div>
            <div className="decision-breakdown">
              {(["pass", "review", "reject"] as const).map((decision) => (
                <span key={decision} className={`decision-pill ${decision}`}>
                  {decisionCopy[decision].label} {performance?.decision_counts[decision] ?? 0}
                </span>
              ))}
            </div>
          </div>

          <div className="panel report-panel">
            <div className="panel-heading">
              <div>
                <p className="eyebrow">Batch QA</p>
                <h2>Inspection Report</h2>
              </div>
              <FileText size={22} />
            </div>
            <div className="report-grid">
              <div>
                <span>Total</span>
                <strong>{batchReport?.total_inspections ?? 0}</strong>
              </div>
              <div>
                <span>Good</span>
                <strong>{batchReport?.pass_count ?? 0}</strong>
              </div>
              <div>
                <span>Review</span>
                <strong>{batchReport?.review_count ?? 0}</strong>
              </div>
              <div>
                <span>Reject</span>
                <strong>{batchReport?.reject_count ?? 0}</strong>
              </div>
              <div>
                <span>Reject Rate</span>
                <strong>{percent(batchReport?.reject_rate)}</strong>
              </div>
              <div>
                <span>Avg Score</span>
                <strong>{batchReport?.avg_score ?? 0}</strong>
              </div>
            </div>
            <div className="defect-type-list">
              {defectTypeCounts.length ? (
                defectTypeCounts.map(([defectType, count]) => (
                  <span key={defectType}>
                    {labelize(defectType)} <strong>{count}</strong>
                  </span>
                ))
              ) : (
                <span>No defect types recorded.</span>
              )}
            </div>
          </div>
        </section>

        <section id="registry" className="section-grid">
          <div className="panel model-version-panel">
            <div className="panel-heading">
              <div>
                <p className="eyebrow">Model registry</p>
                <h2>Model Versions</h2>
              </div>
              <ListChecks size={22} />
            </div>
            <div className="version-list">
              {modelVersions.map((version) => (
                <article key={version.id} className="version-card">
                  <div className="version-head">
                    <div>
                      <strong>
                        {version.product_name ?? "Research model"} v{version.version}
                      </strong>
                      <span>{version.model_kind}</span>
                    </div>
                    <span className={`status ${version.status}`}>{version.status}</span>
                  </div>
                  <div className="run-metrics">
                    <span>samples {version.training_samples ?? "n/a"}</span>
                    <span>threshold {version.threshold ?? "n/a"}</span>
                    <span>id {shortId(version.id)}</span>
                  </div>
                  <div className="version-actions">
                    <button
                      className="secondary-action small-action"
                      onClick={() => onApproveModel(version)}
                      disabled={!!busy || version.status === "approved" || version.status === "active"}
                    >
                      <CheckCircle2 size={16} />
                      Approve
                    </button>
                    <button
                      className="primary-action small-action"
                      onClick={() => onActivateModel(version)}
                      disabled={!!busy || version.status === "active" || !version.product_id}
                    >
                      <Zap size={16} />
                      Activate
                    </button>
                  </div>
                </article>
              ))}
              {!modelVersions.length && <div className="empty-inline">No model versions yet.</div>}
            </div>
          </div>

          <div className="panel audit-panel">
            <div className="panel-heading">
              <div>
                <p className="eyebrow">Audit trail</p>
                <h2>QA Events</h2>
              </div>
              <ClipboardList size={22} />
            </div>
            <div className="audit-list">
              {auditEvents.slice(0, 12).map((event) => (
                <article key={event.id} className="audit-event">
                  <span>{event.event_type}</span>
                  <strong>{event.message}</strong>
                  <small>{new Date(event.created_at).toLocaleString()}</small>
                </article>
              ))}
              {!auditEvents.length && <div className="empty-inline">No QA events yet.</div>}
            </div>
          </div>
        </section>

        <section className="section-grid">
          <div className="panel queue-panel">
            <div className="panel-heading">
              <div>
                <p className="eyebrow">Inspection records</p>
                <h2>History</h2>
              </div>
            </div>
            <div className="table-wrap">
              <table>
                <thead>
                  <tr>
                    <th>Decision</th>
                    <th>Product</th>
                    <th>Score</th>
                    <th>Batch</th>
                    <th>Model</th>
                    <th>Time</th>
                  </tr>
                </thead>
                <tbody>
                  {inspections.map((inspection) => (
                    <tr
                      key={inspection.id}
                      onClick={() => {
                        setBatchResults([]);
                        setBatchIndex(0);
                        setLatest(inspection);
                      }}
                    >
                      <td>
                        <span className={`decision-dot ${inspection.decision}`} /> {decisionCopy[inspection.decision].label}
                      </td>
                      <td>{inspection.product_name}</td>
                      <td>{inspection.score}</td>
                      <td>{inspection.batch_id ?? "none"}</td>
                      <td>{shortId(inspection.model_version_id)}</td>
                      <td>{new Date(inspection.created_at).toLocaleTimeString()}</td>
                    </tr>
                  ))}
                  {!inspections.length && (
                    <tr>
                      <td colSpan={6}>No inspections yet.</td>
                    </tr>
                  )}
                </tbody>
              </table>
            </div>
          </div>
        </section>

        <section id="datasets" className="section-grid">
          <div className="panel dataset-panel">
            <div className="panel-heading">
              <div>
                <p className="eyebrow">Training inputs</p>
                <h2>Dataset Sources</h2>
              </div>
              <Database size={22} />
            </div>
            <div className="dataset-grid">
              {datasets.map((dataset) => (
                <article key={dataset.id} className="dataset-card">
                  <div>
                    <strong>{dataset.name}</strong>
                    <span>{dataset.task.replace("_", " ")}</span>
                  </div>
                  <p>{dataset.domain_fit}</p>
                  <small>
                    {dataset.license} · {dataset.image_count ?? "n/a"} images
                  </small>
                  <a href={dataset.source_url} target="_blank" rel="noreferrer">
                    Open source
                  </a>
                </article>
              ))}
            </div>
          </div>

          <div className="panel model-panel">
            <div className="panel-heading">
              <div>
                <p className="eyebrow">Research runs</p>
                <h2>Training Runs</h2>
              </div>
              <Brain size={22} />
            </div>
            <div className="run-list">
              {trainingRuns.map((run) => (
                <article key={run.id} className="run-card">
                  <div>
                    <strong>{run.run_name}</strong>
                    <span>
                      {run.source}
                      {run.category ? ` · ${run.category}` : ""}
                    </span>
                  </div>
                  <div className="run-metrics">
                    <span>normal {run.normal_images ?? "n/a"}</span>
                    <span>patches {run.memory_patches ?? "n/a"}</span>
                    <span>threshold {run.threshold ?? "n/a"}</span>
                    {run.metrics?.f1 !== undefined && <span>f1 {String(run.metrics.f1)}</span>}
                  </div>
                </article>
              ))}
              {!trainingRuns.length && <div className="empty-inline">No training runs yet.</div>}
            </div>
          </div>
        </section>
      </section>
      <DashboardAssistant selectedProductId={selectedProductId || undefined} inspection={displayedInspection} />
    </main>
  );
}

export default App;
