import type { ComponentType } from 'react';
import { Rocket, ScanSearch, ShieldCheck } from 'lucide-react';
import { Link, NavLink } from 'react-router-dom';
import logoSrc from '../assets/chemistry-svgrepo-com.svg';

type NavItem = {
  to: string;
  label: string;
  icon: ComponentType<{ className?: string }>;
};

const navItems: NavItem[] = [
  { to: '/analyze', label: 'Analyze', icon: ShieldCheck },
  { to: '/screen', label: 'Screen', icon: ScanSearch },
];

function Sidebar() {
  return (
    <aside className="hidden shrink-0 rounded-[2rem] border border-[#762123]/10 bg-[#f8f8f8] px-4 py-6 shadow-[var(--dravix-shadow-soft)] lg:flex lg:w-[92px] lg:flex-col lg:items-center lg:justify-between">
      <div className="flex flex-col items-center gap-8">
        <Link to="/" className="flex h-14 w-14 items-center justify-center rounded-[1.4rem] border border-[#762123]/10 bg-white shadow-sm">
          <img src={logoSrc} alt="Dravix" className="h-7 w-7" />
        </Link>
        <nav className="flex flex-col gap-4">
        {navItems.map((item) => {
          const Icon = item.icon;
          return (
            <NavLink
              key={item.to}
              to={item.to}
              title={item.label}
              className="dravix-sidebar-button flex h-12 w-12 items-center justify-center rounded-xl transition-colors"
              aria-label={item.label}
            >
              {({ isActive }) => (
                <span
                  data-active={String(isActive)}
                  className="dravix-sidebar-button flex h-12 w-12 items-center justify-center rounded-xl transition-all"
                >
                  <Icon className="h-5 w-5" />
                </span>
              )}
            </NavLink>
          );
        })}
      </nav>
      </div>
      <div className="flex h-12 w-12 items-center justify-center rounded-xl bg-gradient-to-r from-[#784F74] to-[#E8967F] text-white shadow-sm">
        <Rocket className="h-5 w-5" />
      </div>
    </aside>
  );
}

export default Sidebar;
