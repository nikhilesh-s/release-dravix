type Props = {
  title: string;
  subtitle?: string;
};

/**
 * Lightweight section marker used to separate grouped workflows inside the
 * larger Analyze and Screen tabs without changing the underlying design system.
 */
function SectionBanner({ title, subtitle }: Props) {
  return (
    <div className="rounded-[1.5rem] bg-gradient-to-r from-[#20181A] via-[#762123] to-[#784F74] px-5 py-4 text-white shadow-[var(--dravix-shadow-card)]">
      <div className="text-lg font-light">{title}</div>
      {subtitle ? <div className="mt-1 text-sm text-white/68">{subtitle}</div> : null}
    </div>
  );
}

export default SectionBanner;
