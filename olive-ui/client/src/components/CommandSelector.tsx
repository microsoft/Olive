import React from 'react';
import { motion } from 'framer-motion';
import { FiTerminal, FiCpu, FiZap, FiDatabase, FiTool } from 'react-icons/fi';
import '../styles/CommandSelector.css';
import { CLICommand } from '../types';

const commandIcons: Record<string, React.ReactElement> = {
  'run': <FiTerminal />,
  'run-pass': <FiTool />,
  'auto-opt': <FiCpu />,
  'quantize': <FiZap />,
  'finetune': <FiDatabase />,
  'capture-onnx': <FiDatabase />,
  'convert-adapters': <FiTool />,
  'session-params-tuning': <FiCpu />
};

interface CommandSelectorProps {
  commands: CLICommand[];
  selectedCommand: CLICommand | null;
  onSelect: (command: CLICommand) => void;
}

const CommandSelector: React.FC<CommandSelectorProps> = ({ commands, selectedCommand, onSelect }) => {
  return (
    <div className="card command-selector">
      <h2 className="card-title">Select Command</h2>
      
      <div className="commands-grid">
        {commands.map((command, index) => (
          <motion.div
            key={command.name}
            className={`command-card ${selectedCommand?.name === command.name ? 'selected' : ''}`}
            onClick={() => onSelect(command)}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: index * 0.05 }}
            whileHover={{ scale: 1.02 }}
            whileTap={{ scale: 0.98 }}
          >
            <div className="command-icon">
              {commandIcons[command.name] || <FiTerminal />}
            </div>
            <h3 className="command-name">olive {command.name}</h3>
            <p className="command-description">{command.description}</p>
          </motion.div>
        ))}
      </div>
    </div>
  );
}

export default CommandSelector;