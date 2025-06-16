import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { motion, AnimatePresence } from 'framer-motion';
import { FiPlus, FiTrash2, FiPlay, FiSave, FiUpload } from 'react-icons/fi';
import PassSelector from '../components/PassSelector';
import PassConfigForm from '../components/PassConfigForm';
import ModelConfiguration from '../components/ModelConfiguration';
import '../styles/PassConfiguration.css';
import { 
  PassesByCategory, 
  SelectedPass, 
  InputModel, 
  WorkflowConfig,
  Pass 
} from '../types';

const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000/api';

// Helper function to save workflow state
const saveWorkflowState = (inputModel: InputModel, selectedPasses: SelectedPass[], outputDir: string) => {
  sessionStorage.setItem('olive_workflow_state', JSON.stringify({
    inputModel,
    selectedPasses,
    outputDir,
    lastSaved: Date.now()
  }));
};

// Helper function to load workflow state
const loadWorkflowState = () => {
  const saved = sessionStorage.getItem('olive_workflow_state');
  if (saved) {
    try {
      return JSON.parse(saved);
    } catch (e) {
      console.error('Failed to parse saved workflow state:', e);
      return null;
    }
  }
  return null;
};

const PassConfiguration: React.FC = () => {
  // Initialize passes from cache if available
  const [passes, setPasses] = useState<PassesByCategory>(() => {
    const cachedPasses = sessionStorage.getItem('olive_passes');
    return cachedPasses ? JSON.parse(cachedPasses) : {};
  });
  
  // Initialize form state from saved state
  const savedState = loadWorkflowState();
  
  const [selectedPasses, setSelectedPasses] = useState<SelectedPass[]>(
    savedState?.selectedPasses || []
  );
  const [inputModel, setInputModel] = useState<InputModel>(
    savedState?.inputModel || {
      type: 'HfModel',
      model_path: '',
      task: 'text-generation'
    }
  );
  const [outputDir, setOutputDir] = useState<string>(
    savedState?.outputDir || 'olive_outputs'
  );
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [showPassSelector, setShowPassSelector] = useState<boolean>(false);
  const [isLoadingPasses, setIsLoadingPasses] = useState<boolean>(false);

  // Save state whenever it changes
  useEffect(() => {
    saveWorkflowState(inputModel, selectedPasses, outputDir);
  }, [inputModel, selectedPasses, outputDir]);

  useEffect(() => {
    fetchPasses();
  }, []);

  const fetchPasses = async (): Promise<void> => {
    // Skip if already loading
    if (isLoadingPasses) return;
    
    setIsLoadingPasses(true);
    try {
      const response = await axios.get<PassesByCategory>(`${API_URL}/passes`, {
        timeout: 30000, // 30 seconds timeout
        headers: {
          'Accept': 'application/json',
          'Accept-Encoding': 'gzip, deflate',
        }
      });
      
      // Only update passes if we got valid data
      if (response.data && Object.keys(response.data).length > 0) {
        setPasses(response.data);
        // Cache the passes data in sessionStorage
        sessionStorage.setItem('olive_passes', JSON.stringify(response.data));
      } else {
        // Try to load from cache if API returns empty
        const cachedPasses = sessionStorage.getItem('olive_passes');
        if (cachedPasses) {
          setPasses(JSON.parse(cachedPasses));
        }
      }
    } catch (error) {
      console.error('Fetch passes error:', error);
      // Try to load from cache on error
      const cachedPasses = sessionStorage.getItem('olive_passes');
      if (cachedPasses) {
        setPasses(JSON.parse(cachedPasses));
      }
    } finally {
      setIsLoadingPasses(false);
    }
  };

  const addPass = (pass: Pass): void => {
    const passId = `pass_${Date.now()}`;
    const newPasses = [...selectedPasses, {
      id: passId,
      name: pass.name,
      type: pass.name,
      config: {},
      passInfo: pass
    }];
    setSelectedPasses(newPasses);
    setShowPassSelector(false);
  };

  const removePass = (passId: string): void => {
    setSelectedPasses(selectedPasses.filter(p => p.id !== passId));
  };

  const updatePassConfig = (passId: string, config: Record<string, any>): void => {
    setSelectedPasses(selectedPasses.map(p => 
      p.id === passId ? { ...p, config } : p
    ));
  };

  const runWorkflow = async (): Promise<void> => {
    if (selectedPasses.length === 0) {
      return;
    }

    if (!inputModel.model_path) {
      return;
    }

    setIsLoading(true);

    try {
      // Build workflow configuration (without python_env)
      const workflowConfig: WorkflowConfig = {
        input_model: inputModel,
        passes: selectedPasses.reduce((acc, pass) => {
          acc[pass.id] = {
            type: pass.type,
            config: pass.config
          };
          return acc;
        }, {} as Record<string, { type: string; config: Record<string, any> }>),
        output_dir: outputDir
      };

      // Validate workflow
      const validationResponse = await axios.post<{ valid: boolean; errors: string[] }>(
        `${API_URL}/workflow/validate`, 
        workflowConfig
      );
      
      if (!validationResponse.data.valid) {
        setIsLoading(false);
        return;
      }

      // Run workflow
      const runResponse = await axios.post<{ job_id: string; status: string }>(
        `${API_URL}/workflow/run`, 
        workflowConfig
      );
      
      // Redirect to job monitor
      window.location.href = `/jobs?jobId=${runResponse.data.job_id}`;
    } catch (error) {
      console.error(error);
    } finally {
      setIsLoading(false);
    }
  };

  const saveConfiguration = (): void => {
    const config = {
      input_model: inputModel,
      passes: selectedPasses.reduce((acc, pass) => {
        acc[pass.id] = {
          type: pass.type,
          config: pass.config
        };
        return acc;
      }, {} as Record<string, { type: string; config: Record<string, any> }>),
      output_dir: outputDir
    };

    const blob = new Blob([JSON.stringify(config, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'olive_config.json';
    a.click();
    URL.revokeObjectURL(url);
  };

  const loadConfiguration = (event: React.ChangeEvent<HTMLInputElement>): void => {
    const file = event.target.files?.[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = (e) => {
        try {
          const config = JSON.parse(e.target?.result as string);
          setInputModel(config.input_model || {});
          setOutputDir(config.output_dir || 'olive_outputs');
          
          // Convert loaded passes to the expected format
          const loadedPasses = Object.entries(config.passes || {}).map(([id, pass]: [string, any]) => ({
            id,
            name: pass.type,
            type: pass.type,
            config: pass.config || {},
            passInfo: { name: pass.type } as Pass
          }));
          setSelectedPasses(loadedPasses);
        } catch (error) {
          console.error('Invalid configuration file:', error);
        }
      };
      reader.readAsText(file);
    }
  };

  const clearWorkflow = (): void => {
    if (window.confirm('Are you sure you want to clear the current workflow configuration?')) {
      setInputModel({
        type: 'HfModel',
        model_path: '',
        task: 'text-generation'
      });
      setSelectedPasses([]);
      setOutputDir('olive_outputs');
      // Clear saved state
      sessionStorage.removeItem('olive_workflow_state');
    }
  };

  return (
    <div className="pass-configuration">
      <div className="page-header">
        <h1 className="page-title">Pass Configuration</h1>
        <p className="page-description">
          Build and configure your optimization workflow by selecting and configuring passes
        </p>
      </div>

      {/* Model Configuration */}
      <ModelConfiguration
        modelConfig={inputModel}
        onChange={setInputModel}
      />

      {/* Selected Passes */}
      <div className="card">
        <div className="card-header">
          <h2 className="card-title">Optimization Passes</h2>
          <div className="card-actions">
            <button 
              className="button button-secondary"
              onClick={() => setShowPassSelector(true)}
            >
              <FiPlus /> Add Pass
            </button>
          </div>
        </div>

        <AnimatePresence>
          {selectedPasses.length === 0 ? (
            <div className="empty-state">
              <div className="empty-state-icon">🔧</div>
              <h3 className="empty-state-title">No passes added</h3>
              <p className="empty-state-description">
                Add optimization passes to build your workflow
              </p>
              <button 
                className="button button-primary"
                onClick={() => setShowPassSelector(true)}
                style={{ marginTop: '16px' }}
              >
                <FiPlus /> Add Your First Pass
              </button>
            </div>
          ) : (
            <div className="passes-list">
              {selectedPasses.map((pass, index) => (
                <motion.div
                  key={pass.id}
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, x: -20 }}
                  className="pass-item"
                >
                  <div className="pass-item-header">
                    <div className="pass-item-info">
                      <span className="pass-number">{index + 1}</span>
                      <h3 className="pass-name">{pass.name}</h3>
                      <div className="pass-tags">
                        {pass.passInfo.supported_accelerators?.map(acc => (
                          <span key={acc} className="tag tag-primary">{acc}</span>
                        ))}
                      </div>
                    </div>
                    <button
                      className="button button-danger button-sm"
                      onClick={() => removePass(pass.id)}
                    >
                      <FiTrash2 />
                    </button>
                  </div>
                  <PassConfigForm
                    passName={pass.name}
                    config={pass.config}
                    onChange={(config) => updatePassConfig(pass.id, config)}
                  />
                </motion.div>
              ))}
            </div>
          )}
        </AnimatePresence>
      </div>

      {/* Output Configuration */}
      <div className="card">
        <h2 className="card-title">Output Configuration</h2>
        <div className="form-group">
          <label className="form-label">Output Directory</label>
          <input
            type="text"
            className="form-input"
            value={outputDir}
            onChange={(e) => setOutputDir(e.target.value)}
            placeholder="olive_outputs"
          />
        </div>
      </div>

      {/* Actions */}
      <div className="actions-bar">
        <div className="actions-left">
          <label className="button button-secondary">
            <FiUpload /> Load Config
            <input
              type="file"
              accept=".json"
              onChange={loadConfiguration}
              style={{ display: 'none' }}
            />
          </label>
          <button 
            className="button button-secondary"
            onClick={saveConfiguration}
            disabled={selectedPasses.length === 0}
          >
            <FiSave /> Save Config
          </button>
          <button 
            className="button button-secondary"
            onClick={clearWorkflow}
            style={{ marginLeft: '8px' }}
          >
            <FiTrash2 /> Clear
          </button>
        </div>
        <button 
          className="button button-primary"
          onClick={runWorkflow}
          disabled={isLoading || selectedPasses.length === 0}
        >
          {isLoading ? (
            <>
              <span className="spinner-small"></span> Running...
            </>
          ) : (
            <>
              <FiPlay /> Run Workflow
            </>
          )}
        </button>
      </div>

      {/* Pass Selector Modal */}
      {showPassSelector && (
        <PassSelector
          passes={passes}
          onSelect={addPass}
          onClose={() => setShowPassSelector(false)}
        />
      )}
    </div>
  );
}

export default PassConfiguration;