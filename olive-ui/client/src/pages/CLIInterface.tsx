import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { FiPlay, FiHelpCircle } from 'react-icons/fi';
import CommandSelector from '../components/CommandSelector';
import ArgumentsForm from '../components/ArgumentsForm';
import '../styles/CLIInterface.css';
import { CLICommand } from '../types';

const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000/api';

const CLIInterface: React.FC = () => {
  const [commands, setCommands] = useState<CLICommand[]>([]);
  const [selectedCommand, setSelectedCommand] = useState<CLICommand | null>(null);
  const [commandArgs, setCommandArgs] = useState<Record<string, any>>({});
  const [isLoading, setIsLoading] = useState<boolean>(false);

  useEffect(() => {
    fetchCommands();
  }, []);

  const fetchCommands = async (): Promise<void> => {
    try {
      const response = await axios.get<CLICommand[]>(`${API_URL}/cli-commands`);
      setCommands(response.data);
    } catch (error) {
      console.error('Failed to fetch CLI commands:', error);
    }
  };

  const selectCommand = (command: CLICommand): void => {
    setSelectedCommand(command);
    // Initialize args with defaults
    const initialArgs: Record<string, any> = {};
    Object.entries(command.args).forEach(([name, arg]) => {
      if (arg.default !== undefined) {
        initialArgs[name] = arg.default;
      }
    });
    setCommandArgs(initialArgs);
  };

  const updateArgs = (name: string, value: any): void => {
    setCommandArgs({
      ...commandArgs,
      [name]: value
    });
  };

  const runCommand = async (): Promise<void> => {
    if (!selectedCommand) {
      return;
    }

    // Validate required args
    const missingArgs: string[] = [];
    Object.entries(selectedCommand.args).forEach(([name, arg]) => {
      if (arg.required && !commandArgs[name]) {
        missingArgs.push(name);
      }
    });

    if (missingArgs.length > 0) {
      return;
    }

    setIsLoading(true);

    try {
      const response = await axios.post<{ job_id: string; status: string }>(`${API_URL}/cli/run`, {
        command: selectedCommand.name,
        args: commandArgs
      });
      
      // Redirect to job monitor
      window.location.href = `/jobs?jobId=${response.data.job_id}`;
    } catch (error) {
      console.error('Failed to run command:', error);
    } finally {
      setIsLoading(false);
    }
  };

  const generateCliCommand = (): string => {
    if (!selectedCommand) return '';

    let cmd = `python -m olive ${selectedCommand.name}`;
    
    Object.entries(commandArgs).forEach(([name, value]) => {
      if (value !== undefined && value !== '') {
        if (typeof value === 'boolean') {
          if (value) {
            cmd += ` --${name}`;
          }
        } else {
          cmd += ` --${name} "${value}"`;
        }
      }
    });

    return cmd;
  };

  return (
    <div className="cli-interface">
      <div className="page-header">
        <h1 className="page-title">CLI Interface</h1>
        <p className="page-description">
          Run Olive commands with a visual interface
        </p>
      </div>

      {/* Command Selector */}
      <CommandSelector
        commands={commands}
        selectedCommand={selectedCommand}
        onSelect={selectCommand}
      />

      {/* Arguments Form */}
      {selectedCommand && (
        <div className="card">
          <div className="card-header">
            <h2 className="card-title">Command Arguments</h2>
            <button className="help-button">
              <FiHelpCircle />
            </button>
          </div>
          
          <ArgumentsForm
            command={selectedCommand}
            args={commandArgs}
            onChange={updateArgs}
          />

          {/* CLI Preview */}
          <div className="cli-preview">
            <h3 className="cli-preview-title">CLI Command Preview</h3>
            <div className="cli-preview-command">
              <code>{generateCliCommand()}</code>
            </div>
          </div>

          {/* Run Button */}
          <div className="actions-bar">
            <button 
              className="button button-primary"
              onClick={runCommand}
              disabled={isLoading}
            >
              {isLoading ? (
                <>
                  <span className="spinner-small"></span> Running...
                </>
              ) : (
                <>
                  <FiPlay /> Run Command
                </>
              )}
            </button>
          </div>
        </div>
      )}
    </div>
  );
}

export default CLIInterface;