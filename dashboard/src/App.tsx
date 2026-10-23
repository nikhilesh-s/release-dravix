import { useEffect, useState } from 'react';
import { Navigate, Route, Routes } from 'react-router-dom';
import SiteLockGate from './components/SiteLockGate';
import Sidebar from './layout/Sidebar';
import TopNav from './layout/TopNav';
import AnalyzePage from './pages/AnalyzePage';
import HomePage from './pages/HomePage';
import ScreenPage from './pages/ScreenPage';
import { clearToken, getToken } from './services/api';

function App() {
  const [isUnlocked, setIsUnlocked] = useState(() => Boolean(getToken()));
  const [expired, setExpired] = useState(false);

  useEffect(() => {
    const onLogout = () => {
      setExpired(true);
      setIsUnlocked(false);
    };
    window.addEventListener('dravix-logout', onLogout);
    return () => window.removeEventListener('dravix-logout', onLogout);
  }, []);

  const lockSite = () => {
    clearToken();
    setIsUnlocked(false);
  };

  if (!isUnlocked) {
    return <SiteLockGate notice={expired ? 'Your session expired. Enter the password again.' : ''} onUnlock={() => { setExpired(false); setIsUnlocked(true); }} />;
  }

  return (
    <div className="min-h-screen bg-[var(--dravix-bg-app)] text-[var(--dravix-ink)]">
      <div className="mx-auto flex min-h-screen w-full max-w-[1720px] gap-4 px-4 py-4 lg:px-6">
        <Sidebar />
        <main className="dravix-shell min-w-0 flex-1 rounded-[2rem] border border-[#762123]/10 p-4 md:p-6">
          <TopNav onLock={lockSite} />
          <Routes>
            <Route path="/" element={<HomePage />} />
            <Route path="/analyze" element={<AnalyzePage />} />
            <Route path="/screen" element={<ScreenPage />} />
            <Route path="*" element={<Navigate to="/" replace />} />
          </Routes>
        </main>
      </div>
    </div>
  );
}

export default App;
