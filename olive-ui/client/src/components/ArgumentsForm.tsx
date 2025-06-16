import React from 'react';
import axios from 'axios';
import { FiFile } from 'react-icons/fi';
import '../styles/ArgumentsForm.css';
import { CLICommand, CLIArgument } from '../types';

const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000/api';

interface ArgumentsFormProps {
  command: CLICommand;
  args: Record<string, any>;
  onChange: (name: string, value: any) => void;
}

const ArgumentsForm: React.FC<ArgumentsFormProps> = ({ command, args, onChange }) => {
  const handleFileUpload = async (argName: string, file: File) => {
    try {
      const formData = new FormData();
      formData.append('file', file);
      
      const response = await axios.post(`${API_URL}/upload`, formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });
      
      // Store the full file path returned by the server
      onChange(argName, response.data.file_path);
    } catch (error) {
      console.error('Failed to upload file:', error);
      // Fallback to just the filename if upload fails
      onChange(argName, file.name);
    }
  };

  const renderArgument = (argName: string, argInfo: CLIArgument): React.ReactElement => {
    const value = args[argName];

    if (argInfo.type === 'boolean') {
      return (
        <div key={argName} className="arg-field">
          <label className="checkbox-container">
            <input
              type="checkbox"
              className="checkbox"
              checked={value || false}
              onChange={(e) => onChange(argName, e.target.checked)}
            />
            <span className="arg-name">
              --{argName}
              {argInfo.required && <span className="required">*</span>}
            </span>
            {argInfo.description && (
              <span className="arg-description">{argInfo.description}</span>
            )}
          </label>
        </div>
      );
    }

    if (argInfo.type === 'choice') {
      return (
        <div key={argName} className="arg-field">
          <label className="arg-label">
            <span className="arg-name">
              --{argName}
              {argInfo.required && <span className="required">*</span>}
            </span>
            {argInfo.description && (
              <span className="arg-description">{argInfo.description}</span>
            )}
          </label>
          <select
            className="form-select"
            value={value || ''}
            onChange={(e) => onChange(argName, e.target.value)}
          >
            <option value="">Select...</option>
            {argInfo.choices?.map(choice => (
              <option key={choice} value={choice}>{choice}</option>
            ))}
          </select>
        </div>
      );
    }

    if (argInfo.type === 'file') {
      return (
        <div key={argName} className="arg-field">
          <label className="arg-label">
            <span className="arg-name">
              --{argName}
              {argInfo.required && <span className="required">*</span>}
            </span>
            {argInfo.description && (
              <span className="arg-description">{argInfo.description}</span>
            )}
          </label>
          <div className="file-input-wrapper">
            <input
              type="file"
              className="file-input-hidden"
              onChange={(e) => {
                if (e.target.files && e.target.files[0]) {
                  handleFileUpload(argName, e.target.files[0]);
                }
              }}
              id={`file-${argName}`}
            />
            <label htmlFor={`file-${argName}`} className="file-dropzone">
              <FiFile className="dropzone-icon" />
              <p className="dropzone-text">
                {value ? (typeof value === 'string' ? value.split('/').pop() || value.split('\\').pop() || value : 'File selected') : 'Click to select a file'}
              </p>
            </label>
          </div>
        </div>
      );
    }

    // Default to text input
    return (
      <div key={argName} className="arg-field">
        <label className="arg-label">
          <span className="arg-name">
            --{argName.replace(/_/g, '-')}
            {argInfo.required && <span className="required">*</span>}
          </span>
          {argInfo.description && (
            <span className="arg-description">{argInfo.description}</span>
          )}
        </label>
        <input
          type="text"
          className="form-input"
          value={value || ''}
          onChange={(e) => onChange(argName, e.target.value)}
          placeholder={argInfo.default?.toString() || ''}
        />
      </div>
    );
  };

  const requiredArgs = Object.entries(command.args).filter(([_, info]) => info.required);
  const optionalArgs = Object.entries(command.args).filter(([_, info]) => !info.required);

  return (
    <div className="arguments-form">
      {requiredArgs.length > 0 && (
        <div className="args-section">
          <h3 className="args-section-title">Required Arguments</h3>
          <div className="args-list">
            {requiredArgs.map(([name, info]) => renderArgument(name, info))}
          </div>
        </div>
      )}

      {optionalArgs.length > 0 && (
        <div className="args-section">
          <h3 className="args-section-title">Optional Arguments</h3>
          <div className="args-list">
            {optionalArgs.map(([name, info]) => renderArgument(name, info))}
          </div>
        </div>
      )}
    </div>
  );
}

export default ArgumentsForm;