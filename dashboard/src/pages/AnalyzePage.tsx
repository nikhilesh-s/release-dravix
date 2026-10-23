import { useState } from 'react';
import PredictionResultCard from '../components/PredictionResultCard';
import SectionBanner from '../components/SectionBanner';
import FormulationForm, { PRESETS } from '../forms/FormulationForm';
import PageContainer from '../layout/PageContainer';
import { api, type PredictResponse } from '../services/api';

function AnalyzePage() {
  const [form, setForm] = useState(PRESETS[0]);
  const [response, setResponse] = useState<PredictResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const runPrediction = async () => {
    setLoading(true);
    setError('');
    try {
      setResponse(await api.predict([form]));
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Prediction failed');
    } finally {
      setLoading(false);
    }
  };

  return (
    <PageContainer eyebrow="Analyze" title="Analyze">
      <SectionBanner
        title="Single-formulation screening"
        subtitle="Predict cone-calorimeter TTI, pHRR, THR and average HRR, then read the composite CFRPS risk score."
      />
      <div className="grid gap-6 xl:grid-cols-[1.08fr_0.92fr]">
        <FormulationForm form={form} onChange={setForm} onSubmit={runPrediction} loading={loading} />
        <div className="space-y-4">
          {error ? <div className="rounded-[1.25rem] bg-[#fff1f1] p-4 text-sm text-[#9E2A2A]">{error}</div> : null}
          <PredictionResultCard response={response} />
        </div>
      </div>
    </PageContainer>
  );
}

export default AnalyzePage;
