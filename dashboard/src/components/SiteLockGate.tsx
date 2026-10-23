import { FlaskConical, LockKeyhole } from 'lucide-react';
import { useState } from 'react';
import chemistryIcon from '../assets/chemistry-svgrepo-com.svg';
import { api } from '../services/api';

type Props = {
  onUnlock: () => void;
  notice?: string;
};

/**
 * Login gate: the password is checked by the API (POST /login), which returns a
 * signed token kept in sessionStorage and sent as a Bearer header on /predict.
 */
function SiteLockGate({ onUnlock, notice }: Props) {
  const [value, setValue] = useState('');
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);

  const submit = async (event: React.FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    setLoading(true);
    try {
      await api.login(value);
      setError('');
      onUnlock();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Login failed');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="flex min-h-screen items-center justify-center bg-[var(--dravix-bg-app)] px-4 py-10">
      <div className="w-full max-w-xl rounded-[2rem] border border-[#762123]/10 bg-white p-8 shadow-[var(--dravix-shadow-soft)] md:p-10">
        <div className="flex items-center gap-4">
          <div className="flex h-16 w-16 items-center justify-center rounded-[1.4rem] border border-[#762123]/10 bg-[#f8f8f8]">
            <img src={chemistryIcon} alt="Dravix" className="h-8 w-8" />
          </div>
          <div>
            <div className="text-[11px] uppercase tracking-[0.28em] text-[var(--dravix-ink-soft)]">Dravix access gate</div>
            <h1 className="mt-2 text-4xl font-light text-[var(--dravix-ink)]">Protected site entry</h1>
          </div>
        </div>

        <div className="mt-8 rounded-[1.5rem] bg-[#f8f8f8] p-5">
          <div className="flex items-center gap-2 text-sm text-[var(--dravix-ink-soft)]">
            <LockKeyhole className="h-4 w-4" />
            Enter the site password to access the Dravix interface.
          </div>
        </div>

        <form onSubmit={submit} className="mt-6 space-y-4">
          <label className="block text-sm text-[var(--dravix-ink-soft)]">
            Site password
            <input
              type="password"
              value={value}
              onChange={(event) => {
                setValue(event.target.value);
                if (error) setError('');
              }}
              placeholder="Enter access password"
              className="mt-2 w-full rounded-xl border border-[#762123]/10 bg-[#f8f8f8] px-4 py-3 text-sm"
            />
          </label>

          {notice && !error ? <div className="text-sm text-[var(--dravix-ink-soft)]">{notice}</div> : null}
          {error ? <div className="text-sm text-[#9E2A2A]">{error}</div> : null}

          <div className="flex flex-wrap items-center gap-3">
            <button
              type="submit"
              disabled={loading}
              className="inline-flex items-center gap-2 rounded-full bg-gradient-to-r from-[#784F74] to-[#E8967F] px-5 py-3 text-sm text-white disabled:opacity-60"
            >
              <FlaskConical className="h-4 w-4" />
              {loading ? 'Checking…' : 'Enter Dravix'}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}

export default SiteLockGate;
