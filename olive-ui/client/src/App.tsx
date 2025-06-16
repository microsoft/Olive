import React from 'react';
import { BrowserRouter as Router, Routes, Route, NavLink } from 'react-router-dom';
import { FiSettings, FiTerminal, FiActivity, FiCpu } from 'react-icons/fi';
import './App.css';
import PassConfiguration from './pages/PassConfiguration';
import CLIInterface from './pages/CLIInterface';
import JobMonitor from './pages/JobMonitor';

const App: React.FC = () => {
  return (
    <Router>
      <div className="app">
        <nav className="sidebar">
          <div className="sidebar-header">
            <div className="logo">
              <div className="logo-icon">
                <FiCpu size={24} />
              </div>
              <div className="logo-text">
                <h1>Olive</h1>
                <p>AI Model Optimization</p>
              </div>
            </div>
          </div>
          
          <div className="nav-content">
            <div className="nav-section">
              <h3 className="nav-section-title">Workflow</h3>
              <NavLink 
                to="/" 
                className={({ isActive }) => `nav-item ${isActive ? 'active' : ''}`}
              >
                <FiSettings className="nav-icon" />
                <span className="nav-text">Pass Configuration</span>
              </NavLink>
            </div>
            
            <div className="nav-section">
              <h3 className="nav-section-title">Tools</h3>
              <NavLink 
                to="/cli" 
                className={({ isActive }) => `nav-item ${isActive ? 'active' : ''}`}
              >
                <FiTerminal className="nav-icon" />
                <span className="nav-text">CLI Interface</span>
              </NavLink>
              <NavLink 
                to="/jobs" 
                className={({ isActive }) => `nav-item ${isActive ? 'active' : ''}`}
              >
                <FiActivity className="nav-icon" />
                <span className="nav-text">Job Monitor</span>
              </NavLink>
            </div>
          </div>
          
          <div className="sidebar-footer">
            <div className="version-info">
              <p>Version 1.0.0</p>
            </div>
          </div>
        </nav>
        
        <main className="main-content">
          <div className="content-wrapper">
            <Routes>
              <Route path="/" element={<PassConfiguration />} />
              <Route path="/cli" element={<CLIInterface />} />
              <Route path="/jobs" element={<JobMonitor />} />
            </Routes>
          </div>
        </main>
      </div>
    </Router>
  );
}

export default App;