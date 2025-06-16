import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { useSearchParams } from 'react-router-dom';
import { FiRefreshCw, FiCheckCircle, FiXCircle, FiClock, FiFolder, FiCopy, FiInfo, FiPlay, FiTrash2 } from 'react-icons/fi';
import '../styles/JobMonitor.css';
import { Job, WorkflowOutput } from '../types';

const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000/api';

const JobMonitor: React.FC = () => {
  const [searchParams] = useSearchParams();
  // Initialize jobs from sessionStorage if available
  const [jobs, setJobs] = useState<Job[]>(() => {
    const cachedJobs = sessionStorage.getItem('olive_jobs');
    return cachedJobs ? JSON.parse(cachedJobs) : [];
  });
  const [selectedJob, setSelectedJob] = useState<string | null>(() => {
    const cachedSelectedJob = sessionStorage.getItem('olive_selected_job');
    return cachedSelectedJob || null;
  });

  useEffect(() => {
    const jobId = searchParams.get('jobId');
    if (jobId) {
      fetchJobStatus(jobId);
      setSelectedJob(jobId);
    } else {
      // On mount, refresh status for all running jobs
      jobs.forEach(job => {
        if (job.status === 'running') {
          fetchJobStatus(job.job_id);
        }
      });
    }
  }, [searchParams]);

  // Cache selected job when it changes
  useEffect(() => {
    if (selectedJob) {
      sessionStorage.setItem('olive_selected_job', selectedJob);
    } else {
      sessionStorage.removeItem('olive_selected_job');
    }
  }, [selectedJob]);

  useEffect(() => {
    if (selectedJob) {
      const job = jobs.find(j => j.job_id === selectedJob);
      if (job?.status === 'running') {
        const interval = setInterval(() => {
          fetchJobStatus(selectedJob);
        }, 2000);

        return () => clearInterval(interval);
      }
    }
  }, [selectedJob, jobs]);

  const fetchJobStatus = async (jobId: string): Promise<void> => {
    try {
      const response = await axios.get<Job>(`${API_URL}/job/${jobId}`);
      const jobData = response.data;
      
      setJobs(prevJobs => {
        const existingIndex = prevJobs.findIndex(j => j.job_id === jobId);
        let newJobs: Job[];
        if (existingIndex >= 0) {
          newJobs = [...prevJobs];
          newJobs[existingIndex] = jobData;
        } else {
          newJobs = [...prevJobs, jobData];
        }
        // Cache jobs in sessionStorage
        sessionStorage.setItem('olive_jobs', JSON.stringify(newJobs));
        return newJobs;
      });

      // Job status updated silently
    } catch (error) {
      console.error('Failed to fetch job status:', error);
    }
  };

  const getStatusIcon = (status: string): React.ReactElement => {
    switch (status) {
      case 'running':
        return <FiClock className="status-icon status-running" />;
      case 'completed':
        return <FiCheckCircle className="status-icon status-success" />;
      case 'failed':
        return <FiXCircle className="status-icon status-error" />;
      default:
        return <FiClock className="status-icon" />;
    }
  };

  const getStatusClass = (status: string): string => {
    switch (status) {
      case 'running':
        return 'status-badge status-running';
      case 'completed':
        return 'status-badge status-success';
      case 'failed':
        return 'status-badge status-error';
      default:
        return 'status-badge';
    }
  };

  const formatDuration = (start: string | undefined, end: string | undefined): string => {
    if (!start) return 'N/A';
    
    const startTime = new Date(start).getTime();
    const endTime = end ? new Date(end).getTime() : Date.now();
    const duration = endTime - startTime;
    
    const seconds = Math.floor(duration / 1000);
    const minutes = Math.floor(seconds / 60);
    const hours = Math.floor(minutes / 60);
    
    if (hours > 0) {
      return `${hours}h ${minutes % 60}m ${seconds % 60}s`;
    } else if (minutes > 0) {
      return `${minutes}m ${seconds % 60}s`;
    } else {
      return `${seconds}s`;
    }
  };

  const copyToClipboard = (text: string) => {
    navigator.clipboard.writeText(text);
  };

  const openFolder = (path: string) => {
    // In a real implementation, this would open the folder in the file explorer
    // For now, we'll just copy the path
    copyToClipboard(path);
  };


  const clearJob = (jobId: string) => {
    setJobs(prevJobs => {
      const newJobs = prevJobs.filter(j => j.job_id !== jobId);
      sessionStorage.setItem('olive_jobs', JSON.stringify(newJobs));
      return newJobs;
    });
    if (selectedJob === jobId) {
      setSelectedJob(null);
    }
  };

  const clearAllJobs = () => {
    if (window.confirm('Are you sure you want to clear all jobs?')) {
      setJobs([]);
      setSelectedJob(null);
      sessionStorage.removeItem('olive_jobs');
      sessionStorage.removeItem('olive_selected_job');
    }
  };

  const renderWorkflowOutput = (output: WorkflowOutput) => {
    return (
      <div className="workflow-output">
        {/* Best Model Section */}
        {output.best_model && (
          <div className="output-section">
            <h5>Best Model</h5>
            <div className="output-info">
              <div className="info-row">
                <span className="info-label">Path:</span>
                <div className="info-value path-value">
                  <code>{output.best_model.model_path}</code>
                  <button 
                    className="icon-button"
                    onClick={() => copyToClipboard(output.best_model!.model_path)}
                    title="Copy path"
                  >
                    <FiCopy />
                  </button>
                  <button 
                    className="icon-button"
                    onClick={() => openFolder(output.best_model!.model_path)}
                    title="Open folder"
                  >
                    <FiFolder />
                  </button>
                </div>
              </div>
              <div className="info-row">
                <span className="info-label">Output Model Type:</span>
                <span className="info-value">{output.best_model.model_type.slice(0, -5)}</span>
              </div>
              <div className="info-row">
                <span className="info-label">Device:</span>
                <span className="info-value">{output.best_model.device || 'N/A'}</span>
              </div>
              <div className="info-row">
                <span className="info-label">Provider:</span>
                <span className="info-value">{output.best_model.execution_provider || 'N/A'}</span>
              </div>
            </div>
          </div>
        )}

        {/* All Output Models */}
        {output.output_models && output.output_models.length > 0 && (
          <div className="output-section">
            <h5>All Output Models ({output.output_models.length})</h5>
            <div className="models-list">
              {output.output_models.map((model, index) => (
                <details key={index} className="model-details">
                  <summary className="model-summary">
                    <span className="model-index">{index + 1}</span>
                    <span className="model-pass">{model.from_pass || 'Unknown Pass'}</span>
                    <span className="model-device">{model.device} / {model.execution_provider}</span>
                  </summary>
                  <div className="model-content">
                    <div className="info-row">
                      <span className="info-label">Path:</span>
                      <div className="info-value path-value">
                        <code>{model.model_path}</code>
                        <button 
                          className="icon-button"
                          onClick={() => copyToClipboard(model.model_path)}
                          title="Copy path"
                        >
                          <FiCopy />
                        </button>
                      </div>
                    </div>
                    {model.metrics && (
                      <div className="model-metrics">
                        <h6>Metrics:</h6>
                        <div className="metrics-grid">
                          {Object.entries(model.metrics).map(([key, value]) => (
                            <div key={key} className="metric-item">
                              <span className="metric-name">{key}:</span>
                              <span className="metric-value">
                                {typeof value === 'object' ? JSON.stringify(value, null, 2) : String(value)}
                              </span>
                            </div>
                          ))}
                        </div>
                      </div>
                    )}
                  </div>
                </details>
              ))}
            </div>
          </div>
        )}

        {/* Metrics Section */}
        {output.best_model?.metrics_value && Object.keys(output.best_model.metrics_value).length > 0 && (
          <div className="output-section">
            <h5>Best Model Metrics</h5>
            <div className="metrics-grid">
              {Object.entries(output.best_model.metrics_value).map(([key, value]) => (
                <div key={key} className="metric-item">
                  <span className="metric-name">{key}:</span>
                  <span className="metric-value">
                    {typeof value === 'object' ? JSON.stringify(value, null, 2) : String(value)}
                  </span>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Available Devices */}
        {output.available_devices && output.available_devices.length > 0 && (
          <div className="output-section">
            <h5>Available Devices</h5>
            <div className="devices-list">
              {output.available_devices.map((device) => (
                <span key={device} className="device-tag">{device}</span>
              ))}
            </div>
          </div>
        )}
      </div>
    );
  };

  const selectedJobData = jobs.find(j => j.job_id === selectedJob);

  return (
    <div className="job-monitor">
      <div className="page-header">
        <div>
          <h1 className="page-title">Job Monitor</h1>
          <p className="page-description">
            Monitor the status of your Olive workflows and commands
          </p>
        </div>
        {jobs.length > 0 && (
          <button 
            className="button button-secondary"
            onClick={clearAllJobs}
          >
            <FiTrash2 /> Clear All
          </button>
        )}
      </div>

      {jobs.length === 0 ? (
        <div className="card">
          <div className="empty-state">
            <div className="empty-state-icon">📊</div>
            <h3 className="empty-state-title">No jobs yet</h3>
            <p className="empty-state-description">
              Run a workflow or CLI command to see jobs here
            </p>
          </div>
        </div>
      ) : (
        <div className="job-monitor-layout">
          <div className="jobs-sidebar">
            <h3 className="sidebar-title">Jobs</h3>
            <div className="jobs-list">
              {jobs.map((job) => (
                <div 
                  key={job.job_id} 
                  className={`job-list-item ${selectedJob === job.job_id ? 'selected' : ''}`}
                  onClick={() => setSelectedJob(job.job_id)}
                >
                  <div className="job-list-header">
                    <div className="job-list-info">
                      <h4 className="job-list-id">Job {job.job_id.substring(0, 8)}</h4>
                      <span className={getStatusClass(job.status)}>
                        {getStatusIcon(job.status)}
                        {job.status}
                      </span>
                    </div>
                    {job.status === 'running' && (
                      <FiRefreshCw className="spinning" />
                    )}
                  </div>
                  {job.created_at && (
                    <div className="job-list-time">
                      {new Date(job.created_at).toLocaleString()}
                    </div>
                  )}
                </div>
              ))}
            </div>
          </div>

          <div className="job-details-panel">
            {selectedJobData ? (
              <>
                <div className="job-details-header">
                  <div>
                    <h2>Job {selectedJobData.job_id.substring(0, 8)}</h2>
                    <div className="job-meta">
                      <span className={getStatusClass(selectedJobData.status)}>
                        {getStatusIcon(selectedJobData.status)}
                        {selectedJobData.status}
                      </span>
                      <span className="job-duration">
                        Duration: {formatDuration(selectedJobData.created_at, selectedJobData.completed_at)}
                      </span>
                    </div>
                  </div>
                  <div className="job-actions">
                    <button 
                      className="button button-danger button-sm"
                      onClick={() => clearJob(selectedJobData.job_id)}
                    >
                      <FiTrash2 /> Clear
                    </button>
                  </div>
                </div>

                <div className="job-details-content">
                  {selectedJobData.status === 'completed' && selectedJobData.workflow_output && (
                    <details className="card" open>
                      <summary className="card-title">
                        <FiInfo /> Output Model
                      </summary>
                      <div className="card-content">
                        {renderWorkflowOutput(selectedJobData.workflow_output)}
                      </div>
                    </details>
                  )}

                  {selectedJobData.error && (
                    <details className="card error-card" open>
                      <summary className="card-title">Error</summary>
                      <div className="card-content">
                        <pre className="error-log">{selectedJobData.error}</pre>
                      </div>
                    </details>
                  )}

                  {selectedJobData.output && (
                    <details className="card">
                      <summary className="card-title">Output Log</summary>
                      <div className="card-content">
                        <pre className="output-log">{selectedJobData.output}</pre>
                      </div>
                    </details>
                  )}

                  {selectedJobData.config?.cli_command ? (
                    <details className="card" open>
                      <summary className="card-title">
                        <FiPlay /> CLI Command
                      </summary>
                      <div className="card-content">
                        <div className="cli-command">
                          <code>{selectedJobData.config.cli_command}</code>
                          <button 
                            className="icon-button"
                            onClick={() => copyToClipboard(selectedJobData.config?.cli_command || '')}
                            title="Copy command"
                          >
                            <FiCopy />
                          </button>
                        </div>
                      </div>
                    </details>
                  ) : (
                    selectedJobData.config && (
                      <details className="card">
                        <summary className="card-title">
                          <FiInfo /> Configuration
                        </summary>
                        <div className="card-content">
                          <pre className="config-log">
                            {JSON.stringify(selectedJobData.config, null, 2)}
                          </pre>
                        </div>
                      </details>
                    )
                  )}
                </div>
              </>
            ) : (
              <div className="empty-state">
                <p>Select a job to view details</p>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}

export default JobMonitor;