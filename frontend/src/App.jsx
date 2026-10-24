import React from 'react';
import { Navigate, Route, Routes, BrowserRouter as Router } from 'react-router-dom';
import { Toaster } from "@/components/ui/sonner";
import ScrollToTop from './components/ScrollToTop.jsx';
import SiteLayout from './layout/site-layout.jsx';
import HomePage from './pages/HomePage.jsx';
import LandingPage from './pages/LandingPage.jsx';
import BenchmarksPage from './pages/BenchmarksPage.jsx';
import DisarmPage from './pages/DisarmPage.jsx';
import DocsPage from './pages/DocsPage.jsx';

function App() {
  return (
    <Router>
      <ScrollToTop />
      <Routes>
        <Route element={<SiteLayout />}>
          <Route path="/" element={<LandingPage />} />
          <Route path="/scanner" element={<HomePage />} />
          <Route path="/layers" element={<Navigate to="/docs#layer-01" replace />} />
          <Route path="/benchmarks" element={<BenchmarksPage />} />
          <Route path="/disarm" element={<DisarmPage />} />
          <Route path="/docs" element={<DocsPage />} />
        </Route>
      </Routes>
      <Toaster />
    </Router>
  );
}

export default App;
