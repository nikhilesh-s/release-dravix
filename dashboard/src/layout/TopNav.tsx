import { ArrowUpRight, Lock } from 'lucide-react';
import { Link, useLocation } from 'react-router-dom';

function labelFromPath(pathname: string): string {
  const key = pathname.replace('/', '') || 'dashboard';
  return key.charAt(0).toUpperCase() + key.slice(1);
}

type Props = {
  onLock: () => void;
};

function TopNav({ onLock }: Props) {
  const location = useLocation();
  const label = labelFromPath(location.pathname);

  return (
    <header className="mb-4 flex flex-wrap items-center justify-between gap-4 rounded-[1.7rem] border border-[#762123]/10 bg-white px-5 py-4 shadow-[var(--dravix-shadow-card)]">
      <div className="min-w-0">
        <div className="text-[11px] uppercase tracking-[0.26em] text-[var(--dravix-ink-soft)]">Dravix engine v2.0 preview</div>
        <div className="mt-1 text-2xl font-light text-[var(--dravix-ink)]">{label}</div>
      </div>
      <div className="flex flex-wrap items-center gap-3 text-sm">
        <button onClick={onLock} title="Log out" aria-label="Log out" className="flex h-11 w-11 items-center justify-center rounded-2xl border border-[#762123]/10 bg-white text-[#762123]">
          <Lock className="h-4 w-4" />
        </button>
        <Link to="/analyze" className="flex items-center gap-2 rounded-full bg-gradient-to-r from-[#784F74] to-[#E8967F] px-5 py-2.5 text-white shadow-sm">
          Run analysis <ArrowUpRight className="h-4 w-4" />
        </Link>
      </div>
    </header>
  );
}

export default TopNav;
