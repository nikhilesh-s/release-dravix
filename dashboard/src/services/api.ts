const RAW_BASE = import.meta.env.VITE_API_BASE_URL || (import.meta.env.DEV ? 'http://localhost:8000' : '');
export const API_BASE_URL = RAW_BASE.replace(/\/$/, '');
const TOKEN_KEY = 'dravix-token-v2';

export type PropertyKey = 'TTI' | 'pHRR' | 'THR' | 'avg_HRR';
export const PROPERTY_LABELS: Record<PropertyKey, string> = {
  TTI: 'Time to ignition',
  pHRR: 'Peak heat release rate',
  THR: 'Total heat release',
  avg_HRR: 'Average heat release rate',
};

export type Formulation = {
  name?: string;
  matrix_canonical: string;
  material_form: string;
  components: string;
  heat_flux_kWm2?: number;
  thickness_mm?: number;
  fr_total_wtpct?: number;
  is_control?: number;
};

export type Cfrps = {
  score: number;
  sigma: number;
  interval95: [number, number];
  coverage: number;
  used: string[];
  missing: string[];
  subscores: Record<string, number>;
  indices: Record<string, number | null>;
  weights_used: Record<string, number>;
};

export type PredictionResult = {
  name: string | null;
  properties: Record<PropertyKey, number>;
  uncertainty: null;
  cfrps: Cfrps;
  warnings: string[];
};

export type PredictResponse = {
  results: PredictionResult[];
  units: Record<PropertyKey, string>;
  weights_provisional: boolean;
  notes: string[];
};

export class ApiError extends Error {
  status: number;
  constructor(message: string, status: number) {
    super(message);
    this.status = status;
  }
}

export const getToken = () => window.sessionStorage.getItem(TOKEN_KEY);
export const clearToken = () => window.sessionStorage.removeItem(TOKEN_KEY);

async function request<T>(path: string, body: unknown, auth = true): Promise<T> {
  if (!API_BASE_URL) throw new ApiError('API URL not configured (set VITE_API_BASE_URL and redeploy)', 0);
  const token = getToken();
  const response = await fetch(`${API_BASE_URL}${path}`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      ...(auth && token ? { Authorization: `Bearer ${token}` } : {}),
    },
    body: JSON.stringify(body),
  });
  const payload = await response.json().catch(() => null);
  if (!response.ok) {
    if (response.status === 401 && auth) {
      clearToken();
      window.dispatchEvent(new Event('dravix-logout'));
    }
    const detail = payload?.detail;
    throw new ApiError(typeof detail === 'string' ? detail : `Request failed (${response.status})`, response.status);
  }
  return payload as T;
}

/** Fill fr_total_wtpct from FR + synergist amounts when blank, as engine/demo.html does. */
export function withFrTotal(f: Formulation): Formulation {
  if (f.fr_total_wtpct !== undefined) return f;
  const sum = (re: RegExp) => [...f.components.matchAll(re)].reduce((acc, m) => acc + Number(m[1]), 0);
  const fr = sum(/\|(?:FR|synergist)\|\s*([\d.]+)/gi);
  // phr is parts per hundred resin, not wt%: normalise by the total
  const total = /phr/i.test(f.components) ? sum(/\|\s*([\d.]+)\s*(?:wt%|phr|g)?\s*(?:;|$)/gi) : 0;
  return { ...f, fr_total_wtpct: total > 0 ? (100 * fr) / total : fr };
}

export type Health = { model: { matrices: string[]; forms: string[]; data_sha256: string } };

export const api = {
  health: async () => (await fetch(`${API_BASE_URL}/health`)).json() as Promise<Health>,
  login: async (password: string) => {
    const { token } = await request<{ token: string }>('/login', { password }, false);
    window.sessionStorage.setItem(TOKEN_KEY, token);
  },
  predict: (items: Formulation[]) => request<PredictResponse>('/predict', items.map(withFrTotal)),
};
