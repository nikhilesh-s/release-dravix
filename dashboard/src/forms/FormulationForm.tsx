import { useEffect, useState, type FormEvent } from 'react';
import { api, type Formulation } from '../services/api';

export const FALLBACK_FORMS = ['bulk', 'fibre_composite', 'film', 'foam', 'other', 'wood_plastic'];

/** Example inputs only (engine/example.json); no stored results. */
export const PRESETS: Formulation[] = [
  { name: 'PP + 20 wt% APP', matrix_canonical: 'PP', material_form: 'bulk', components: 'PP|matrix|80 wt%; APP|FR|20 wt%', heat_flux_kWm2: 50, thickness_mm: 3 },
  { name: 'Neat PP (control)', matrix_canonical: 'PP', material_form: 'bulk', components: 'PP|matrix|100 wt%', heat_flux_kWm2: 50, thickness_mm: 3, is_control: 1 },
  { name: 'EP + 20 wt% APP', matrix_canonical: 'EP', material_form: 'bulk', components: 'EP|matrix|80 wt%; APP|FR|20 wt%', heat_flux_kWm2: 50, thickness_mm: 3 },
];

/** Matrix and form vocabularies the engine was trained on (from GET /health). */
export function useEngineOptions() {
  const [options, setOptions] = useState({ matrices: [] as string[], forms: FALLBACK_FORMS });
  useEffect(() => {
    api.health().then((h) => setOptions({ matrices: h.model.matrices, forms: h.model.forms })).catch(() => undefined);
  }, []);
  return options;
}

export const inputClass = 'w-full rounded-xl border border-[#762123]/10 bg-[#f8f8f8] px-4 py-3 text-sm';
const labelClass = 'mb-2 block text-sm text-[var(--dravix-ink-soft)]';

const num = (v: string) => (v.trim() === '' ? undefined : Number(v));

type Props = {
  form: Formulation;
  onChange: (next: Formulation) => void;
  onSubmit: () => void;
  loading?: boolean;
};

function FormulationForm({ form, onChange, onSubmit, loading = false }: Props) {
  const { matrices, forms } = useEngineOptions();
  const set = (patch: Partial<Formulation>) => onChange({ ...form, ...patch });
  const submit = (event: FormEvent) => {
    event.preventDefault();
    onSubmit();
  };

  return (
    <form onSubmit={submit} className="dravix-card rounded-[1.75rem] p-6">
      <div className="flex flex-wrap gap-2">
        {PRESETS.map((preset) => (
          <button key={preset.name} type="button" onClick={() => onChange(preset)} className="rounded-full border border-[var(--dravix-border)] bg-white px-3 py-1 text-xs text-[var(--dravix-ink)]">
            {preset.name}
          </button>
        ))}
      </div>
      <div className="mt-6 grid gap-4 md:grid-cols-2">
        <label className={labelClass}>
          Name (optional)
          <input value={form.name ?? ''} onChange={(e) => set({ name: e.target.value })} className={`mt-2 ${inputClass}`} />
        </label>
        <label className={labelClass}>
          Matrix polymer
          <input required list="matrix-options" value={form.matrix_canonical} onChange={(e) => set({ matrix_canonical: e.target.value })} className={`mt-2 ${inputClass}`} placeholder="PP" />
          <datalist id="matrix-options">
            {matrices.map((m) => <option key={m} value={m} />)}
          </datalist>
        </label>
        <label className={labelClass}>
          Material form
          <select value={form.material_form} onChange={(e) => set({ material_form: e.target.value })} className={`mt-2 ${inputClass}`}>
            {forms.map((f) => <option key={f} value={f}>{f}</option>)}
          </select>
        </label>
        <label className={labelClass}>
          Heat flux (kW/m²)
          <input required type="number" min="1" max="200" step="any" value={form.heat_flux_kWm2 ?? ''} onChange={(e) => set({ heat_flux_kWm2: num(e.target.value) })} className={`mt-2 ${inputClass}`} />
        </label>
        <label className={labelClass}>
          Thickness (mm)
          <input type="number" min="0.01" step="any" value={form.thickness_mm ?? ''} onChange={(e) => set({ thickness_mm: num(e.target.value) })} className={`mt-2 ${inputClass}`} />
        </label>
        <label className={labelClass}>
          FR total (wt%, blank = sum of FR + synergist)
          <input type="number" min="0" max="100" step="any" value={form.fr_total_wtpct ?? ''} onChange={(e) => set({ fr_total_wtpct: num(e.target.value) })} className={`mt-2 ${inputClass}`} />
        </label>
      </div>
      <label className={`mt-4 ${labelClass}`}>
        Components (name|role|amount; …), roles: matrix, FR, synergist, filler
        <textarea required rows={3} value={form.components} onChange={(e) => set({ components: e.target.value })} className={`mt-2 ${inputClass}`} />
      </label>
      <label className="mt-2 flex items-center gap-2 text-sm text-[var(--dravix-ink-soft)]">
        <input type="checkbox" checked={form.is_control === 1} onChange={(e) => set({ is_control: e.target.checked ? 1 : 0 })} />
        Neat polymer control (no additives)
      </label>
      <button type="submit" disabled={loading} className="mt-6 rounded-full bg-gradient-to-r from-[#784F74] to-[#E8967F] px-5 py-3 text-sm text-white disabled:opacity-60">
        {loading ? 'Predicting…' : 'Run analysis'}
      </button>
    </form>
  );
}

export default FormulationForm;
