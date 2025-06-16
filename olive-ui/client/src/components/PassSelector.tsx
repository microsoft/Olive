import React, { useState } from 'react';
import { motion } from 'framer-motion';
import { FiSearch, FiX, FiCpu, FiZap, FiDatabase } from 'react-icons/fi';
import '../styles/PassSelector.css';
import { Pass, PassesByCategory, PassWithFramework } from '../types';

interface PassSelectorProps {
  passes: PassesByCategory;
  onSelect: (pass: Pass) => void;
  onClose: () => void;
}

const PassSelector: React.FC<PassSelectorProps> = ({ passes, onSelect, onClose }) => {
  const [searchTerm, setSearchTerm] = useState<string>('');
  const [selectedCategory, setSelectedCategory] = useState<string>('all');

  const getCategoryIcon = (category: string): React.ReactElement | null => {
    switch (category) {
      case 'Quantization':
        return <FiZap />;
      case 'Optimization':
        return <FiCpu />;
      case 'Conversion':
        return <FiDatabase />;
      default:
        return null;
    }
  };

  const filterPasses = (): PassWithFramework[] => {
    let filteredPasses: PassWithFramework[] = [];

    // Check if passes object is empty
    if (!passes || Object.keys(passes).length === 0) {
      return filteredPasses;
    }

    Object.entries(passes).forEach(([framework, categories]) => {
      if (typeof categories === 'object' && !Array.isArray(categories)) {
        Object.entries(categories).forEach(([subcategory, passList]) => {
          if (Array.isArray(passList)) {
            passList.forEach(pass => {
              // Include pass if:
              // 1. selectedCategory is 'all' OR
              // 2. selectedCategory matches the framework (e.g., 'ONNX') OR
              // 3. selectedCategory matches the subcategory (e.g., 'Quantization')
              const categoryMatch = selectedCategory === 'all' || 
                                  selectedCategory === framework || 
                                  selectedCategory === subcategory;
              
              const searchMatch = !searchTerm || 
                                pass.name.toLowerCase().includes(searchTerm.toLowerCase());
              
              if (categoryMatch && searchMatch) {
                filteredPasses.push({
                  ...pass,
                  framework,
                  category: subcategory
                });
              }
            });
          }
        });
      } else if (Array.isArray(categories)) {
        // Handle flat array structure (shouldn't happen with current data structure)
        categories.forEach(pass => {
          const categoryMatch = selectedCategory === 'all' || 
                              selectedCategory === framework;
          
          const searchMatch = !searchTerm || 
                            pass.name.toLowerCase().includes(searchTerm.toLowerCase());
          
          if (categoryMatch && searchMatch) {
            filteredPasses.push({
              ...pass,
              framework,
              category: 'Other'
            });
          }
        });
      }
    });

    return filteredPasses;
  };

  const getCategories = (): string[] => {
    const categories = new Set<string>(['all']);
    
    Object.entries(passes).forEach(([framework, cats]) => {
      categories.add(framework);
      if (typeof cats === 'object' && !Array.isArray(cats)) {
        Object.keys(cats).forEach(cat => categories.add(cat));
      }
    });

    return Array.from(categories);
  };

  const filteredPasses = filterPasses();
  const categories = getCategories();

  return (
    <div className="modal-overlay" onClick={onClose}>
      <motion.div 
        className="modal pass-selector-modal"
        onClick={(e) => e.stopPropagation()}
        initial={{ opacity: 0, scale: 0.9 }}
        animate={{ opacity: 1, scale: 1 }}
      >
        <div className="modal-header">
          <h2 className="modal-title">Select Pass</h2>
          <button className="modal-close" onClick={onClose}>
            <FiX />
          </button>
        </div>

        <div className="pass-selector-content">
          {/* Search Bar */}
          <div className="search-container">
            <FiSearch className="search-icon" />
            <input
              type="text"
              className="search-input"
              placeholder="Search passes..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              autoFocus
            />
          </div>

          {/* Category Filters */}
          <div className="category-filters">
            {categories.map(category => (
              <button
                key={category}
                className={`category-filter ${selectedCategory === category ? 'active' : ''}`}
                onClick={() => setSelectedCategory(category)}
              >
                {getCategoryIcon(category)}
                {category === 'all' ? 'All Passes' : category}
              </button>
            ))}
          </div>

          {/* Passes Grid */}
          <div className="passes-grid">
            {filteredPasses.length === 0 ? (
              <div className="empty-state">
                <p>No passes found matching your criteria</p>
              </div>
            ) : (
              filteredPasses.map((pass, index) => (
                <motion.div
                  key={`${pass.framework}-${pass.name}`}
                  className="pass-card"
                  onClick={() => onSelect(pass)}
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: index * 0.05 }}
                  whileHover={{ scale: 1.02 }}
                  whileTap={{ scale: 0.98 }}
                >
                  <div className="pass-card-header">
                    <h3 className="pass-card-title">{pass.name}</h3>
                    <span className="pass-framework">{pass.framework}</span>
                  </div>
                  <div className="pass-card-category">
                    {getCategoryIcon(pass.category)}
                    {pass.category}
                  </div>
                  <div className="pass-card-tags">
                    {pass.supported_accelerators?.map(acc => (
                      <span key={acc} className="tag tag-primary">{acc}</span>
                    ))}
                    {pass.supported_precisions?.slice(0, 3).map(prec => (
                      <span key={prec} className="tag tag-secondary">{prec}</span>
                    ))}
                  </div>
                </motion.div>
              ))
            )}
          </div>
        </div>
      </motion.div>
    </div>
  );
}

export default PassSelector;