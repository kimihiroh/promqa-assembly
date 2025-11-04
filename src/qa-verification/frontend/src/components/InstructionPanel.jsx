// components/GraphPanel.jsx
import React, { memo } from 'react';

const InstructionPanel = memo(function InstructionPanel({ idx, imagePath, activePanel, togglePanel }) {
    const getPanelTitle = () => {
        switch(activePanel) {
            case 'graph_original':
                return 'Original Graph Visualization';
            case 'graph_step':
                return 'Graph Visualization w/ Step Annotation';
            default:
                return 'Parts Visualization';
        }
    };

    return (
        <div
            key={`instruction-panel-${idx}-${activePanel}`}
            className="card flex-grow-1"
            style={{
                flexBasis: "50%",
                maxHeight: "50vh",
                borderLeft: '3px solid #27ae60' // Green border to distinguish from other panels
            }}
        >
            <div
                className="card-header d-flex align-items-center justify-content-between"
                style={{
                    backgroundColor: '#f8f9fa',
                    borderBottom: '1px solid #dee2e6',
                    padding: '8px 12px'
                }}
            >
                <label
                    className="mb-0"
                    style={{
                        fontWeight: 'bold',
                        color: '#2c3e50'
                    }}
                >
                    {getPanelTitle()}
                </label>

                {/* Toggle buttons in the card header with improved styling */}
                <div className="btn-group btn-group-sm" role="group" aria-label="Panel toggle">
                    <button
                        type="button"
                        className={`btn ${activePanel === 'parts' ? 'btn-primary' : 'btn-outline-primary'}`}
                        onClick={() => togglePanel('parts')}
                        style={{
                            fontWeight: activePanel === 'parts' ? 'bold' : 'normal',
                            padding: '4px 8px'
                        }}
                    >
                        Parts
                    </button>
                    <button
                        type="button"
                        className={`btn ${activePanel === 'graph_original' ? 'btn-primary' : 'btn-outline-primary'}`}
                        onClick={() => togglePanel('graph_original')}
                        style={{
                            fontWeight: activePanel === 'graph_original' ? 'bold' : 'normal',
                            padding: '4px 8px'
                        }}
                    >
                        Graph (original)
                    </button>
                    <button
                        type="button"
                        className={`btn ${activePanel === 'graph_step' ? 'btn-primary' : 'btn-outline-primary'}`}
                        onClick={() => togglePanel('graph_step')}
                        style={{
                            fontWeight: activePanel === 'graph_step' ? 'bold' : 'normal',
                            padding: '4px 8px'
                        }}
                    >
                        Graph (w/ steps)
                    </button>
                </div>
            </div>

            <div
                className="card-body d-flex justify-content-center align-items-center overflow-hidden p-2"
                style={{ backgroundColor: '#fcfcfc' }}
            >
                {imagePath ? (
                    <img
                        src={imagePath}
                        alt={imagePath}
                        className="w-100 h-100"
                        style={{
                            objectFit: "contain",
                            borderRadius: "4px",
                            border: '1px solid #eee'
                        }}
                    />
                ) : (
                    <div style={{
                        padding: '15px',
                        textAlign: 'center',
                        color: '#777',
                        border: '1px dashed #ddd',
                        borderRadius: '4px',
                        backgroundColor: '#fff',
                        width: '60%'
                    }}>
                        No graph available
                    </div>
                )}
            </div>
        </div>
    );
});

export default InstructionPanel;
