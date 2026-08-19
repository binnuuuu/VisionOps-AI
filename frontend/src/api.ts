import type {
  AssistantAnswer,
  AuditEvent,
  BatchReport,
  DatasetEntry,
  Inspection,
  ModelVersion,
  PerformanceStats,
  Product,
  SampleInspectionPack,
  TrainingRun,
  TwinSnapshot
} from "./types";

export const API_URL = import.meta.env.VITE_API_URL ?? "http://127.0.0.1:8000";

async function unwrap<T>(response: Response): Promise<T> {
  if (!response.ok) {
    let message = `Request failed with ${response.status}`;
    try {
      const body = await response.json();
      message = body.detail ?? message;
    } catch {
      // Keep the fallback message.
    }
    throw new Error(message);
  }
  return response.json() as Promise<T>;
}

export function mediaUrl(path: string): string {
  if (path.startsWith("http")) return path;
  return `${API_URL}${path}`;
}

export function twinSocketUrl(): string {
  const base = API_URL || window.location.origin;
  const url = new URL("/api/twin/ws", base);
  url.protocol = url.protocol === "https:" ? "wss:" : "ws:";
  return url.toString();
}

export async function getTwinState(): Promise<TwinSnapshot> {
  return unwrap<TwinSnapshot>(await fetch(`${API_URL}/api/twin/state`));
}

export async function getProducts(): Promise<Product[]> {
  return unwrap<Product[]>(await fetch(`${API_URL}/api/products`));
}

export async function getInspections(): Promise<Inspection[]> {
  return unwrap<Inspection[]>(await fetch(`${API_URL}/api/inspections`));
}

export async function getDatasets(): Promise<DatasetEntry[]> {
  const data = await unwrap<{ datasets: DatasetEntry[] }>(await fetch(`${API_URL}/api/datasets/catalog`));
  return data.datasets;
}

export async function getSampleInspections(): Promise<SampleInspectionPack> {
  return unwrap<SampleInspectionPack>(await fetch(`${API_URL}/api/sample-inspections`));
}

export async function getTrainingRuns(): Promise<TrainingRun[]> {
  return unwrap<TrainingRun[]>(await fetch(`${API_URL}/api/training/runs`));
}

export async function getModelVersions(): Promise<ModelVersion[]> {
  return unwrap<ModelVersion[]>(await fetch(`${API_URL}/api/model-versions`));
}

export async function approveModelVersion(versionId: string): Promise<ModelVersion> {
  const body = new FormData();
  body.append("approved_by", "QA demo");
  const data = await unwrap<{ model_version: ModelVersion }>(
    await fetch(`${API_URL}/api/model-versions/${versionId}/approve`, {
      method: "POST",
      body
    })
  );
  return data.model_version;
}

export async function activateModelVersion(versionId: string): Promise<ModelVersion> {
  const data = await unwrap<{ model_version: ModelVersion }>(
    await fetch(`${API_URL}/api/model-versions/${versionId}/activate`, {
      method: "POST"
    })
  );
  return data.model_version;
}

export async function getAuditEvents(): Promise<AuditEvent[]> {
  return unwrap<AuditEvent[]>(await fetch(`${API_URL}/api/audit/events`));
}

export async function getBatchReport(productId?: string): Promise<BatchReport> {
  const params = productId ? `?product_id=${encodeURIComponent(productId)}` : "";
  return unwrap<BatchReport>(await fetch(`${API_URL}/api/reports/batch${params}`));
}

export async function getPerformanceStats(): Promise<PerformanceStats> {
  return unwrap<PerformanceStats>(await fetch(`${API_URL}/api/performance`));
}

export async function askDashboardAssistant(
  question: string,
  context: { productId?: string; inspectionId?: string; batchId?: string }
): Promise<AssistantAnswer> {
  return unwrap<AssistantAnswer>(
    await fetch(`${API_URL}/api/assistant/ask`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify({
        question,
        product_id: context.productId,
        inspection_id: context.inspectionId,
        batch_id: context.batchId
      })
    })
  );
}

export async function createProduct(form: HTMLFormElement): Promise<Product> {
  const body = new FormData(form);
  const data = await unwrap<{ product: Product }>(
    await fetch(`${API_URL}/api/products`, {
      method: "POST",
      body
    })
  );
  return data.product;
}

export async function teachProduct(productId: string, files: FileList): Promise<Product> {
  const body = new FormData();
  Array.from(files).forEach((file) => body.append("files", file));
  const data = await unwrap<{ product: Product }>(
    await fetch(`${API_URL}/api/products/${productId}/teach`, {
      method: "POST",
      body
    })
  );
  return data.product;
}

export async function inspectProduct(productId: string, file: File): Promise<Inspection> {
  const body = new FormData();
  body.append("product_id", productId);
  body.append("file", file);
  const data = await unwrap<{ inspection: Inspection }>(
    await fetch(`${API_URL}/api/inspect/upload`, {
      method: "POST",
      body
    })
  );
  return data.inspection;
}

export async function inspectBatch(productId: string, files: FileList): Promise<Inspection[]> {
  const body = new FormData();
  body.append("product_id", productId);
  Array.from(files).forEach((file) => body.append("files", file));
  return unwrap<Inspection[]>(
    await fetch(`${API_URL}/api/inspect/batch`, {
      method: "POST",
      body
    })
  );
}

export async function inspectSampleBatch(productId: string, sampleIds: string[]): Promise<Inspection[]> {
  return unwrap<Inspection[]>(
    await fetch(`${API_URL}/api/inspect/samples`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify({
        product_id: productId,
        sample_ids: sampleIds,
        batch_id: "B-2026-DEMO"
      })
    })
  );
}
