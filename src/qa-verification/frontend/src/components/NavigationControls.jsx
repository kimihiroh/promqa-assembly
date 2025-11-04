// components/NavigationControls.jsx
import React, { memo } from 'react';

const NavigationControls = memo(function NavigationControls({
    handleBack,
    handleNext,
    handleSave
}) {
    return (
        <div
            className="d-flex justify-content-between"
            style={{
                padding: '8px 12px',
                margin: '10px 0',
                backgroundColor: '#f8f9fa',
                borderRadius: '4px',
                border: '1px solid #dee2e6',
                borderLeft: '3px solid #34495e'  // Dark blue/slate border for consistency
            }}
        >
            <div className="d-flex gap-2">
                <button
                    className="btn btn-outline-secondary"
                    onClick={handleBack}
                    style={{
                        padding: '6px 12px',
                        fontWeight: '500'
                    }}
                >
                    <i className="bi bi-arrow-left" style={{ marginRight: '4px' }}></i> Back
                </button>
                <button
                    className="btn btn-outline-primary"
                    onClick={handleNext}
                    style={{
                        padding: '6px 12px',
                        fontWeight: '500'
                    }}
                >
                    Next <i className="bi bi-arrow-right" style={{ marginLeft: '4px' }}></i>
                </button>
            </div>
            <button
                className="btn btn-success"
                onClick={handleSave}
                style={{
                    padding: '6px 15px',
                    fontWeight: 'bold'
                }}
            >
                <i className="bi bi-save" style={{ marginRight: '5px' }}></i> Save
            </button>
        </div>
    );
});

export default NavigationControls;
