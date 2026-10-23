import type { PropsWithChildren, ReactNode } from 'react';

type Props = PropsWithChildren<{
  eyebrow?: string;
  title: string;
  description?: string;
  actions?: ReactNode;
}>;

function PageContainer({ eyebrow, title, description, actions, children }: Props) {
  return (
    <div className="space-y-5">
      <section className="dravix-hero relative overflow-hidden rounded-[2rem] border border-[#762123]/10 px-8 py-8 shadow-[var(--dravix-shadow-card)]">
        <div className="absolute right-[-4rem] top-[-4rem] h-40 w-40 rounded-full bg-gradient-to-br from-[#784F74] to-[#E8967F] opacity-15 blur-3xl" />
        <div className="absolute bottom-[-3rem] left-[-2rem] h-32 w-32 rounded-full bg-[#E8967F] opacity-15 blur-3xl" />
        <div className="relative flex flex-wrap items-end justify-between gap-6">
          <div className="max-w-4xl">
            {eyebrow ? (
              <div className="text-[11px] uppercase tracking-[0.26em] text-[var(--dravix-ink-soft)]">{eyebrow}</div>
            ) : null}
            <h1 className="mt-3 text-4xl font-light text-[var(--dravix-ink)] md:text-5xl">{title}</h1>
            {description ? <p className="mt-4 max-w-3xl text-base leading-7 text-[var(--dravix-ink-soft)]">{description}</p> : null}
          </div>
          {actions ? <div className="relative flex flex-wrap gap-3">{actions}</div> : null}
        </div>
      </section>
      {children}
    </div>
  );
}

export default PageContainer;
