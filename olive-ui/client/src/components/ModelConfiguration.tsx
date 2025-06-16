import React from 'react';
import { FiFolder, FiGlobe } from 'react-icons/fi';
import '../styles/ModelConfiguration.css';
import { InputModel } from '../types';

interface ModelType {
  value: string;
  label: string;
  icon: string;
}

interface ModelConfigurationProps {
  modelConfig: InputModel;
  onChange: (config: InputModel) => void;
}

const modelTypes: ModelType[] = [
  { value: 'HfModel', label: 'Hugging Face Model', icon: '🤗' },
  { value: 'PyTorchModel', label: 'PyTorch Model', icon: '🔥' },
  { value: 'ONNXModel', label: 'ONNX Model', icon: '🧮' },
  { value: 'OpenVINOModel', label: 'OpenVINO Model', icon: '🌊' }
];

const taskTypes: string[] = [
  'text-generation',
  'text-classification', 
  'token-classification',
  'question-answering',
  'image-classification',
  'object-detection',
  'feature-extraction'
];

const ModelConfiguration: React.FC<ModelConfigurationProps> = ({ modelConfig, onChange }) => {
  const updateConfig = (field: string, value: any): void => {
    onChange({
      ...modelConfig,
      [field]: value
    });
  };

  const renderModelSpecificFields = (): React.ReactElement | null => {
    switch (modelConfig.type) {
      case 'HfModel':
        return (
          <>
            <div className="form-group">
              <label className="form-label">
                Model ID or Path
                <span className="label-hint">e.g., microsoft/phi-2, gpt2, or local path</span>
              </label>
              <div className="input-with-icon">
                {modelConfig.model_path?.startsWith('http') || modelConfig.model_path?.includes('/') ? 
                  <FiGlobe className="input-icon" /> : 
                  <FiFolder className="input-icon" />
                }
                <input
                  type="text"
                  className="form-input"
                  value={modelConfig.model_path || ''}
                  onChange={(e) => updateConfig('model_path', e.target.value)}
                  placeholder="microsoft/phi-2"
                />
              </div>
            </div>
            <div className="form-group">
              <label className="form-label">Task</label>
              <select
                className="form-select"
                value={modelConfig.task || 'text-generation'}
                onChange={(e) => updateConfig('task', e.target.value)}
              >
                {taskTypes.map(task => (
                  <option key={task} value={task}>{task}</option>
                ))}
              </select>
            </div>
            <div className="form-group">
              <label className="form-label">
                Model Class (Optional)
                <span className="label-hint">Leave empty for auto-detection</span>
              </label>
              <input
                type="text"
                className="form-input"
                value={(modelConfig as any).model_class || ''}
                onChange={(e) => updateConfig('model_class', e.target.value)}
                placeholder="AutoModelForCausalLM"
              />
            </div>
          </>
        );

      case 'PyTorchModel':
        return (
          <>
            <div className="form-group">
              <label className="form-label">Model Path</label>
              <input
                type="text"
                className="form-input"
                value={modelConfig.model_path || ''}
                onChange={(e) => updateConfig('model_path', e.target.value)}
                placeholder="/path/to/model.pt"
              />
            </div>
            <div className="form-group">
              <label className="form-label">Model Script (Optional)</label>
              <input
                type="text"
                className="form-input"
                value={(modelConfig as any).model_script || ''}
                onChange={(e) => updateConfig('model_script', e.target.value)}
                placeholder="/path/to/model_script.py"
              />
            </div>
          </>
        );

      case 'ONNXModel':
        return (
          <>
            <div className="form-group">
              <label className="form-label">ONNX Model Path</label>
              <input
                type="text"
                className="form-input"
                value={modelConfig.model_path || ''}
                onChange={(e) => updateConfig('model_path', e.target.value)}
                placeholder="/path/to/model.onnx"
              />
            </div>
          </>
        );

      case 'OpenVINOModel':
        return (
          <>
            <div className="form-group">
              <label className="form-label">Model Path</label>
              <input
                type="text"
                className="form-input"
                value={modelConfig.model_path || ''}
                onChange={(e) => updateConfig('model_path', e.target.value)}
                placeholder="/path/to/model.xml"
              />
            </div>
          </>
        );

      default:
        return null;
    }
  };

  return (
    <div className="card model-configuration">
      <h2 className="card-title">Input Model Configuration</h2>
      
      <div className="model-type-selector">
        {modelTypes.map(type => (
          <div
            key={type.value}
            className={`model-type-card ${modelConfig.type === type.value ? 'selected' : ''}`}
            onClick={() => updateConfig('type', type.value)}
          >
            <span className="model-type-icon">{type.icon}</span>
            <span className="model-type-label">{type.label}</span>
          </div>
        ))}
      </div>

      <div className="model-fields">
        {renderModelSpecificFields()}
      </div>

      <div className="model-advanced">
        <details>
          <summary className="advanced-toggle">Advanced Options</summary>
          <div className="advanced-content">
            <div className="form-group">
              <label className="checkbox-container">
                <input
                  type="checkbox"
                  className="checkbox"
                  checked={(modelConfig as any).load_kwargs?.trust_remote_code || false}
                  onChange={(e) => updateConfig('load_kwargs', {
                    ...(modelConfig as any).load_kwargs,
                    trust_remote_code: e.target.checked
                  })}
                />
                Trust Remote Code
              </label>
            </div>
            <div className="form-group">
              <label className="form-label">Torch Data Type</label>
              <select
                className="form-select"
                value={(modelConfig as any).torch_dtype || ''}
                onChange={(e) => updateConfig('torch_dtype', e.target.value)}
              >
                <option value="">Auto</option>
                <option value="float32">float32</option>
                <option value="float16">float16</option>
                <option value="bfloat16">bfloat16</option>
              </select>
            </div>
          </div>
        </details>
      </div>
    </div>
  );
}

export default ModelConfiguration;