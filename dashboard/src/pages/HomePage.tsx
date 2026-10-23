import chemistryIcon from '../assets/chemistry-svgrepo-com.svg';

function HomePage() {
  return (
    <div className="flex min-h-[calc(100vh-9rem)] items-center justify-center">
      <div className="flex flex-col items-center gap-8 text-center">
        <div className="flex h-32 w-32 items-center justify-center rounded-[2rem] border border-[#762123]/10 bg-white shadow-[var(--dravix-shadow-soft)]">
          <img src={chemistryIcon} alt="Dravix" className="h-16 w-16" />
        </div>
        <h1 className="text-6xl font-light text-[var(--dravix-ink)] md:text-7xl">Dravix</h1>
      </div>
    </div>
  );
}

export default HomePage;
