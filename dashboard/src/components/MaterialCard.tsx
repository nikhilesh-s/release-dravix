import type { ReactNode } from 'react';

type Props = {
  title: string;
  subtitle?: string;
  children?: ReactNode;
};

function MaterialCard({ title, subtitle, children }: Props) {
  return (
    <div className="dravix-card rounded-[1.75rem] p-5 md:p-6">
      <h3 className="text-xl font-light text-[var(--dravix-ink)]">{title}</h3>
      {subtitle ? <p className="mt-1 text-sm text-[var(--dravix-ink-soft)]">{subtitle}</p> : null}
      {children ? <div className="mt-4">{children}</div> : null}
    </div>
  );
}

export default MaterialCard;
