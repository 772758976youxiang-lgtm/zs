export type Row = {
  id?: string;
  text: string;
  units?: string[];
  split?: string;
  status?: string;
  source?: string;
  group_id?: string;
  spans?: number[][];
  gap_labels?: number[];
  conflict?: boolean;
};
import type { components } from "./generated/api";
export type Dataset = components["schemas"]["DatasetInfo"];
export type Model = components["schemas"]["ModelInfo"];
export type Run = components["schemas"]["RunInfo"] & {
  metrics: Record<string, any>;
};
export type Checkpoint = components["schemas"]["CheckpointInfo"];
let token = "";
export async function session() {
  token = (await (await fetch("/api/session")).json()).token;
}
export function url(path: string) {
  return `${path}${path.includes("?") ? "&" : "?"}token=${encodeURIComponent(token)}`;
}
export async function api<T = any>(path: string, body?: unknown): Promise<T> {
  const response = await fetch(path, {
    method: body === undefined ? "GET" : "POST",
    headers: {
      "X-Local-Token": token,
      ...(body instanceof FormData
        ? {}
        : { "Content-Type": "application/json" }),
    },
    body:
      body === undefined
        ? undefined
        : body instanceof FormData
          ? body
          : JSON.stringify(body),
  });
  const value = await response.json();
  if (!response.ok)
    throw new Error(
      value.error?.message || value.detail || `请求失败 ${response.status}`,
    );
  return value;
}
