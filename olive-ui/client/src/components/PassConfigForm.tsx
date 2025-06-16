import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { FiChevronDown, FiChevronUp, FiInfo } from 'react-icons/fi';
import Editor from '@monaco-editor/react';
import '../styles/PassConfigForm.css';
import { PassSchema, PassSchemaResponse } from '../types';

const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000/api';

interface PassConfigFormProps {
  passName: string;
  config: Record<string, any>;
  onChange: (config: Record<string, any>) => void;
}

interface FieldInfo {
  type: string;
  default?: any;
  required?: boolean;
  description?: string;
  choices?: any[];
}

const PassConfigForm: React.FC<PassConfigFormProps> = ({ passName, config, onChange }) => {
  const [schema, setSchema] = useState<PassSchema | null>(null);
  const [isExpanded, setIsExpanded] = useState<boolean>(false);
  const [isLoading, setIsLoading] = useState<boolean>(false);

  useEffect(() => {
    fetchPassSchema();
  }, [passName]); // eslint-disable-line react-hooks/exhaustive-deps

  const fetchPassSchema = async (): Promise<void> => {
    setIsLoading(true);
    console.log(`Fetching schema for pass: ${passName}`);
    
    try {
      const response = await axios.get<PassSchemaResponse>(`${API_URL}/pass/${passName}/schema`, {
        timeout: 10000, // 10 second timeout
      });
      console.log('Schema response:', response.data);
      
      const fetchedSchema = response.data.schema;
      console.log('Fetched schema:', fetchedSchema);
      
      if (fetchedSchema && Object.keys(fetchedSchema).length > 0) {
        setSchema(fetchedSchema);
        
        // Initialize config with all parameters and their defaults if config is empty
        if (Object.keys(config).length === 0) {
          const defaultConfig: Record<string, any> = {};
          Object.entries(fetchedSchema).forEach(([key, fieldInfo]) => {
            if (fieldInfo.default !== undefined && fieldInfo.default !== null) {
              defaultConfig[key] = fieldInfo.default;
            }
          });
          onChange(defaultConfig);
        }
      } else {
        // Use default schema if API didn't return one
        const defaultSchema = getDefaultSchema(passName);
        console.log('Using default schema:', defaultSchema);
        setSchema(defaultSchema);
        
        if (Object.keys(config).length === 0 && defaultSchema) {
          const defaultConfig: Record<string, any> = {};
          Object.entries(defaultSchema).forEach(([key, fieldInfo]) => {
            if (fieldInfo.default !== undefined && fieldInfo.default !== null) {
              defaultConfig[key] = fieldInfo.default;
            }
          });
          onChange(defaultConfig);
        }
      }
    } catch (error) {
      console.error('Failed to fetch pass schema:', error);
      console.error('Error details:', (error as any).response?.data || (error as any).message);
      
      // Use a default schema if fetch fails
      const defaultSchema = getDefaultSchema(passName);
      setSchema(defaultSchema);
      
      // Initialize config with defaults
      if (Object.keys(config).length === 0) {
        const defaultConfig: Record<string, any> = {};
        Object.entries(defaultSchema).forEach(([key, fieldInfo]) => {
          if (fieldInfo.default !== undefined && fieldInfo.default !== null) {
            defaultConfig[key] = fieldInfo.default;
          }
        });
        onChange(defaultConfig);
      }
    } finally {
      setIsLoading(false);
    }
  };

  const getDefaultConfig = (): Record<string, any> => {
    // Generate default config from schema if available
    if (schema && Object.keys(schema).length > 0) {
      const defaultConfig: Record<string, any> = {};
      Object.entries(schema).forEach(([key, info]) => {
        if (info.default !== undefined && info.default !== null) {
          defaultConfig[key] = info.default;
        } else if (info.required) {
          // Provide template values for required fields
          switch (info.type) {
            case 'integer':
              defaultConfig[key] = 0;
              break;
            case 'float':
              defaultConfig[key] = 0.0;
              break;
            case 'boolean':
              defaultConfig[key] = false;
              break;
            case 'array':
              defaultConfig[key] = [];
              break;
            case 'object':
              defaultConfig[key] = {};
              break;
            default:
              defaultConfig[key] = '';
          }
        }
      });
      return defaultConfig;
    }
    
    // Fallback template for common pass types
    const passTemplates: Record<string, Record<string, any>> = {
      'LoRA': {
        r: 64,
        alpha: 16,
        lora_dropout: 0.05,
        target_modules: ["q_proj", "v_proj"],
        torch_dtype: "bfloat16",
        train_data_config: "train_data",
        eval_data_config: "eval_data",
        training_args: {
          per_device_train_batch_size: 1,
          per_device_eval_batch_size: 1,
          gradient_accumulation_steps: 4,
          max_steps: 150,
          logging_steps: 50
        }
      },
      'QLoRA': {
        r: 64,
        alpha: 16,
        lora_dropout: 0.05,
        target_modules: ["q_proj", "v_proj"],
        torch_dtype: "bfloat16",
        quant_type: "nf4",
        double_quant: true,
        train_data_config: "train_data",
        training_args: {
          per_device_train_batch_size: 1,
          gradient_accumulation_steps: 4,
          max_steps: 150
        }
      },
      'OnnxQuantization': {
        quant_mode: "static",
        per_channel: true,
        reduce_range: false,
        quant_format: "QDQ",
        activation_type: "int8",
        weight_type: "int8"
      },
      'OnnxStaticQuantization': {
        per_channel: true,
        reduce_range: false,
        quant_format: "QDQ",
        calibrate_method: "MinMax",
        calibration_data_reader: "data_reader_config"
      },
      'OnnxDynamicQuantization': {
        per_channel: true,
        reduce_range: false,
        weight_type: "int8"
      }
    };
    
    // Return template if available, otherwise empty object with comment
    return passTemplates[passName] || {
      "// Note": `Configuration for ${passName}. Add parameters as needed.`,
      "// Example": "parameter_name: value"
    };
  };

  const getDefaultSchema = (passName: string): PassSchema => {
    // Default schemas for common passes
    const defaultSchemas: Record<string, PassSchema> = {
      'OnnxQuantization': {
        quant_mode: { type: 'string', default: 'static', choices: ['static', 'dynamic'], description: 'Quantization mode' },
        per_channel: { type: 'boolean', default: true, description: 'Use per-channel quantization' },
        reduce_range: { type: 'boolean', default: false, description: 'Reduce quantization range' },
        quant_format: { type: 'string', default: 'QDQ', choices: ['QDQ', 'QOperator'], description: 'Quantization format' },
        activation_type: { type: 'string', default: 'int8', choices: ['int8', 'uint8'], description: 'Activation data type' },
        weight_type: { type: 'string', default: 'int8', choices: ['int8', 'uint8'], description: 'Weight data type' },
        calibrate_method: { type: 'string', default: 'MinMax', choices: ['MinMax', 'Entropy'], description: 'Calibration method' }
      },
      'OrtTransformersOptimization': {
        model_type: { type: 'string', default: 'bert', required: true, description: 'Transformer model type' },
        num_heads: { type: 'integer', default: 12, description: 'Number of attention heads' },
        hidden_size: { type: 'integer', default: 768, description: 'Hidden layer size' },
        float16: { type: 'boolean', default: false, description: 'Convert to FP16' },
        use_gpu: { type: 'boolean', default: false, description: 'Optimize for GPU' },
        optimization_options: { type: 'object', default: {}, description: 'Additional optimization options' }
      },
      'LoRA': {
        r: { type: 'integer', default: 16, description: 'LoRA rank' },
        alpha: { type: 'integer', default: 32, description: 'LoRA alpha scaling factor' },
        dropout: { type: 'float', default: 0.1, description: 'LoRA dropout rate' },
        target_modules: { type: 'array', default: ['q_proj', 'v_proj'], description: 'Modules to apply LoRA' },
        bias: { type: 'string', default: 'none', choices: ['none', 'all', 'lora_only'], description: 'Bias configuration' },
        use_gradient_checkpointing: { type: 'boolean', default: false, description: 'Use gradient checkpointing' }
      },
      'OnnxStaticQuantization': {
        calibration_data_reader: { type: 'string', default: null, description: 'Calibration data configuration' },
        per_channel: { type: 'boolean', default: true, description: 'Use per-channel quantization' },
        reduce_range: { type: 'boolean', default: false, description: 'Reduce quantization range' },
        quant_format: { type: 'string', default: 'QDQ', choices: ['QDQ', 'QOperator'], description: 'Quantization format' },
        calibrate_method: { type: 'string', default: 'MinMax', choices: ['MinMax', 'Entropy', 'Percentile'], description: 'Calibration method' }
      },
      'OnnxDynamicQuantization': {
        per_channel: { type: 'boolean', default: true, description: 'Use per-channel quantization' },
        reduce_range: { type: 'boolean', default: false, description: 'Reduce quantization range' },
        weight_type: { type: 'string', default: 'int8', choices: ['int8', 'uint8'], description: 'Weight data type' }
      }
    };

    return defaultSchemas[passName] || {};
  };

  const renderField = (fieldName: string, fieldInfo: FieldInfo): React.ReactElement => {
    const value = config[fieldName] !== undefined ? config[fieldName] : fieldInfo.default;

    const handleChange = (newValue: any): void => {
      onChange({
        ...config,
        [fieldName]: newValue
      });
    };

    // Determine field type
    const fieldType = fieldInfo.type?.toLowerCase() || 'string';

    if (fieldInfo.choices) {
      return (
        <div key={fieldName} className="config-field">
          <label className="config-label">
            {fieldName}
            {fieldInfo.required && <span className="required">*</span>}
            {fieldInfo.description && (
              <span className="field-info" title={fieldInfo.description}>
                <FiInfo />
              </span>
            )}
          </label>
          <select
            className="form-select"
            value={value || ''}
            onChange={(e) => handleChange(e.target.value)}
          >
            <option value="">Select...</option>
            {fieldInfo.choices.map(choice => (
              <option key={choice} value={choice}>{choice}</option>
            ))}
          </select>
        </div>
      );
    }

    if (fieldType === 'boolean') {
      return (
        <div key={fieldName} className="config-field">
          <label className="checkbox-container">
            <input
              type="checkbox"
              className="checkbox"
              checked={value || false}
              onChange={(e) => handleChange(e.target.checked)}
            />
            {fieldName}
            {fieldInfo.description && (
              <span className="field-info" title={fieldInfo.description}>
                <FiInfo />
              </span>
            )}
          </label>
        </div>
      );
    }

    if (fieldType === 'integer' || fieldType === 'float' || fieldType === 'number') {
      return (
        <div key={fieldName} className="config-field">
          <label className="config-label">
            {fieldName}
            {fieldInfo.required && <span className="required">*</span>}
            {fieldInfo.description && (
              <span className="field-info" title={fieldInfo.description}>
                <FiInfo />
              </span>
            )}
          </label>
          <input
            type="number"
            className="form-input"
            value={value || ''}
            onChange={(e) => handleChange(fieldType === 'integer' ? parseInt(e.target.value) : parseFloat(e.target.value))}
            step={fieldType === 'integer' ? 1 : 0.01}
            placeholder={fieldInfo.default?.toString() || ''}
          />
        </div>
      );
    }

    if (fieldType === 'array') {
      return (
        <div key={fieldName} className="config-field">
          <label className="config-label">
            {fieldName}
            {fieldInfo.required && <span className="required">*</span>}
            {fieldInfo.description && (
              <span className="field-info" title={fieldInfo.description}>
                <FiInfo />
              </span>
            )}
          </label>
          <input
            type="text"
            className="form-input"
            value={Array.isArray(value) ? value.join(', ') : value || ''}
            onChange={(e) => handleChange(e.target.value.split(',').map(s => s.trim()).filter(Boolean))}
            placeholder="Enter comma-separated values"
          />
        </div>
      );
    }

    // Default to text input
    return (
      <div key={fieldName} className="config-field">
        <label className="config-label">
          {fieldName}
          {fieldInfo.required && <span className="required">*</span>}
          {fieldInfo.description && (
            <span className="field-info" title={fieldInfo.description}>
              <FiInfo />
            </span>
          )}
        </label>
        <input
          type="text"
          className="form-input"
          value={value || ''}
          onChange={(e) => handleChange(e.target.value)}
          placeholder={fieldInfo.default?.toString() || ''}
        />
      </div>
    );
  };

  if (isLoading) {
    return (
      <div className="pass-config-form">
        <div className="loading">
          <div className="spinner"></div>
        </div>
      </div>
    );
  }

  return (
    <div className="pass-config-form">
      <div 
        className="config-header"
        onClick={() => setIsExpanded(!isExpanded)}
      >
        <span className="config-title">Configuration</span>
        {isExpanded ? <FiChevronUp /> : <FiChevronDown />}
      </div>

      {isExpanded && (
        <div className="config-fields">
          {schema && Object.keys(schema).length > 0 ? (
            <div className="fields-grid">
              {Object.entries(schema).map(([fieldName, fieldInfo]) => 
                renderField(fieldName, fieldInfo)
              )}
            </div>
          ) : (
            <div className="no-schema-message">
              <p>Configuration options are being loaded for this pass...</p>
              <p className="text-muted">If this persists, the pass may not have configurable parameters.</p>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

export default PassConfigForm;