import MaterialCard from './MaterialCard';
import { PROPERTY_LABELS, type PredictResponse, type PropertyKey } from '../services/api';

type Props = {
  response: PredictResponse | null;
};

const INDEX_LABELS: Record<string, string> = {
  ignition_resistance: 'Ignition',
  heat_release: 'Heat release',
  thermal_stability: 'Thermal stability',
};

function PredictionResultCard({ response }: Props) {
  const result = response?.results[0];
  if (!response || !result) {
    return (
      <MaterialCard
        title="Prediction result"
        subtitle="Run a formulation to populate the CFRPS risk score and the four predicted cone-calorimeter properties."
      />
    );
  }

  const { cfrps } = result;
  return (
    <MaterialCard title={result.name || 'Formulation'}>
      <div className="grid gap-5 md:grid-cols-2">
        <div className="rounded-[1.5rem] bg-gradient-to-r from-[#784F74] to-[#E8967F] p-4 text-white">
          <div className="text-xs uppercase tracking-[0.24em] text-white/70">CFRPS risk score</div>
          <div className="mt-2 text-3xl font-light">
            {cfrps.score.toFixed(1)}
            <span className="ml-1 text-base text-white/70">/ 100</span>
          </div>
          <div className="mt-1 text-xs text-white/80">Higher = more fire risk</div>
        </div>
        <div className="rounded-[1.5rem] bg-[#f8f8f8] p-4">
          <div className="text-xs uppercase tracking-[0.24em] text-[var(--dravix-ink-soft)]">Input coverage</div>
          <div className="mt-2 text-3xl font-light text-[var(--dravix-ink)]">{Math.round(cfrps.coverage * 100)}%</div>
          <div className="mt-1 text-xs text-[var(--dravix-ink-soft)]">
            {cfrps.missing.length ? `Not predicted: ${cfrps.missing.join(', ')}` : 'All score inputs present'}
          </div>
        </div>
      </div>

      <div className="mt-5 grid gap-4 sm:grid-cols-2">
        {(Object.keys(PROPERTY_LABELS) as PropertyKey[]).map((key) => (
          <div key={key} className="rounded-[1.25rem] bg-[#f8f8f8] p-4">
            <div className="text-xs uppercase tracking-[0.2em] text-[var(--dravix-ink-soft)]">{PROPERTY_LABELS[key]}</div>
            <div className="mt-2 text-2xl font-light text-[var(--dravix-ink)]">
              {result.properties[key].toFixed(1)}
              <span className="ml-1 text-sm text-[var(--dravix-ink-soft)]">{response.units[key]}</span>
            </div>
          </div>
        ))}
      </div>

      <div className="mt-5 grid gap-4 md:grid-cols-3">
        {Object.entries(cfrps.indices).map(([key, value]) => (
          <div key={key} className="rounded-[1.25rem] border border-[var(--dravix-border)] p-4">
            <div className="text-xs uppercase tracking-[0.2em] text-[var(--dravix-ink-soft)]">{INDEX_LABELS[key] ?? key}</div>
            <div className="mt-2 text-2xl font-light text-[var(--dravix-ink)]">
              {value === null ? 'n/a' : (value * 100).toFixed(0)}
            </div>
            <div className="mt-1 text-xs text-[var(--dravix-ink-soft)]">sub-index, 0-100</div>
          </div>
        ))}
      </div>

      {result.warnings?.length ? (
        <ul className="mt-5 space-y-1 rounded-[1.25rem] bg-[#fff4e8] p-4 text-sm text-[#a65a1f]">
          {result.warnings.map((w) => <li key={w}>Warning: {w}</li>)}
        </ul>
      ) : null}

      <ul className="mt-6 space-y-1 text-sm leading-6 text-[var(--dravix-ink-soft)]">
        {response.weights_provisional ? (
          <li>
            <span className="rounded-full bg-[#fff4e8] px-2 py-0.5 text-xs font-medium text-[#a65a1f]">Weights provisional</span>{' '}
            CFRPS weights are pending sign-off.
          </li>
        ) : null}
        {response.notes
          .filter((note) => !note.includes('provisional'))
          .map((note) => (
            <li key={note}>{note}</li>
          ))}
      </ul>
    </MaterialCard>
  );
}

export default PredictionResultCard;
