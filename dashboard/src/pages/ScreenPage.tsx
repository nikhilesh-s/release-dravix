import { useState } from 'react';
import LeaderboardChart from '../charts/LeaderboardChart';
import MaterialCard from '../components/MaterialCard';
import SectionBanner from '../components/SectionBanner';
import { PRESETS, inputClass, useEngineOptions } from '../forms/FormulationForm';
import PageContainer from '../layout/PageContainer';
import { api, type Formulation, type PredictResponse } from '../services/api';

export type RankedRow = { rank: number; name: string; score: number; warnings: string[]; properties: PredictResponse['results'][0]['properties'] };

function ScreenPage() {
  const { matrices, forms } = useEngineOptions();
  const [rows, setRows] = useState<Formulation[]>(PRESETS);
  const [ranking, setRanking] = useState<RankedRow[]>([]);
  const [units, setUnits] = useState<PredictResponse['units'] | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const setRow = (i: number, patch: Partial<Formulation>) => setRows(rows.map((r, j) => (j === i ? { ...r, ...patch } : r)));

  const runRanking = async () => {
    setLoading(true);
    setError('');
    try {
      const res = await api.predict(rows);
      setUnits(res.units);
      setRanking(
        res.results
          .map((r, i) => ({ name: r.name || `Formulation ${i + 1}`, score: r.cfrps.score, warnings: r.warnings ?? [], properties: r.properties }))
          .sort((a, b) => a.score - b.score)
          .map((r, i) => ({ ...r, rank: i + 1 })),
      );
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Ranking failed');
    } finally {
      setLoading(false);
    }
  };

  return (
    <PageContainer eyebrow="Screen" title="Screen">
      <SectionBanner title="Ranking" subtitle="Rank a shortlist by CFRPS risk score (lowest risk first). Compare at the same heat flux." />
      <MaterialCard title="Ranking input table" subtitle="One formulation per row. Example rows are preloaded.">
        <datalist id="screen-matrix-options">
          {matrices.map((m) => <option key={m} value={m} />)}
        </datalist>
        <div className="space-y-3">
          {rows.map((row, i) => (
            <div key={i} className="grid gap-2 rounded-[1.25rem] border border-[var(--dravix-border)] p-3 md:grid-cols-[1.2fr_0.7fr_0.9fr_2fr_0.6fr_0.6fr_auto]">
              <input aria-label="Name" value={row.name ?? ''} onChange={(e) => setRow(i, { name: e.target.value })} className={inputClass} placeholder="Name" />
              <input aria-label="Matrix" list="screen-matrix-options" value={row.matrix_canonical} onChange={(e) => setRow(i, { matrix_canonical: e.target.value })} className={inputClass} placeholder="Matrix" />
              <select aria-label="Form" value={row.material_form} onChange={(e) => setRow(i, { material_form: e.target.value })} className={inputClass}>
                {forms.map((f) => <option key={f} value={f}>{f}</option>)}
              </select>
              <input aria-label="Components" value={row.components} onChange={(e) => setRow(i, { components: e.target.value })} className={inputClass} placeholder="name|role|amount; ..." />
              <input aria-label="Heat flux kW/m2" type="number" value={row.heat_flux_kWm2 ?? ''} onChange={(e) => setRow(i, { heat_flux_kWm2: e.target.value ? Number(e.target.value) : undefined })} className={inputClass} placeholder="kW/m²" />
              <input aria-label="Thickness mm" type="number" value={row.thickness_mm ?? ''} onChange={(e) => setRow(i, { thickness_mm: e.target.value ? Number(e.target.value) : undefined })} className={inputClass} placeholder="mm" />
              <button type="button" onClick={() => setRows(rows.filter((_, j) => j !== i))} className="rounded-full border border-[var(--dravix-border)] px-3 py-2 text-xs text-[var(--dravix-ink)]">
                Remove
              </button>
            </div>
          ))}
        </div>
        <div className="mt-4 flex gap-3">
          <button
            onClick={runRanking}
            disabled={loading || rows.length === 0}
            className="rounded-full bg-gradient-to-r from-[#784F74] to-[#E8967F] px-4 py-2 text-sm text-white disabled:opacity-60"
          >
            {loading ? 'Ranking…' : 'Run ranking'}
          </button>
          <button
            onClick={() => setRows([...rows, { ...PRESETS[0], name: '' }])}
            disabled={rows.length >= 50}
            className="rounded-full border border-[var(--dravix-border)] px-4 py-2 text-sm text-[var(--dravix-ink)] disabled:opacity-60"
          >
            Add row
          </button>
        </div>
        {error ? <div className="mt-4 text-sm text-[#9E2A2A]">{error}</div> : null}
      </MaterialCard>

      <div className="grid gap-6 xl:grid-cols-[0.95fr_1.05fr]">
        <MaterialCard title="Ranked formulations" subtitle="CFRPS 0-100, higher = more fire risk. Weights provisional.">
          <div className="space-y-3">
            {ranking.map((row) => (
              <div key={row.rank} className="flex flex-wrap items-center justify-between gap-3 rounded-[1.25rem] border border-[var(--dravix-border)] p-4">
                <div>
                  <div className="text-sm text-[var(--dravix-ink-soft)]">#{row.rank}</div>
                  <div className="text-lg text-[var(--dravix-ink)]">{row.name}</div>
                </div>
                <div className="text-right text-sm text-[var(--dravix-ink-soft)]">
                  <div className="text-xl font-light text-[var(--dravix-ink)]">CFRPS {row.score.toFixed(1)}</div>
                  TTI {row.properties.TTI.toFixed(0)} {units?.TTI} • pHRR {row.properties.pHRR.toFixed(0)} {units?.pHRR} • THR{' '}
                  {row.properties.THR.toFixed(1)} {units?.THR}
                  {row.warnings.length ? <div className="mt-1 text-xs text-[#a65a1f]">Warning: {row.warnings.join('; ')}</div> : null}
                </div>
              </div>
            ))}
          </div>
        </MaterialCard>
        <LeaderboardChart ranking={ranking} />
      </div>
    </PageContainer>
  );
}

export default ScreenPage;
