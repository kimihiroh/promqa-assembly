import { Handle, Position } from '@xyflow/react';
import { useState, useEffect } from 'react';

const NodeWithScrew = ({ id, data, selected }) => {
    // Initialize state from data prop
    const [checked, setChecked] = useState(data.checked || false);

    // Update local state when data.checked changes
    useEffect(() => {
        setChecked(data.checked || false);
    }, [data.checked]);

    const toggleCheck = () => {
        const newChecked = !checked;

        // Update local state first
        setChecked(newChecked);

        // Call parent handler function
        if (typeof data.onCheck === 'function') {
            data.onCheck(id, newChecked);
        } else {
            console.warn(`onCheck function not found for node ${id}`);
        }
    };

    return (
        <div
            className={`custom-node ${selected ? 'selected' : ''}`}
            style={{
                border: '1px solid black',
                borderRadius: '5px',
                padding: '10px',
                background: 'white',
                position: 'relative',
                minHeight: '40px',
                minWidth: '100px',
                maxWidth: '150px', // Set a maximum width for consistent wrapping
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center', // Center content horizontally
                boxShadow: selected ? '0 0 0 1px #1a192b' : 'none'
            }}
        >
            {/* Checkbox indicator */}
            <div
                onClick={toggleCheck}
                style={{
                    position: 'absolute',
                    top: '50%',
                    left: '10px',
                    transform: 'translateY(-50%)',
                    width: '16px',
                    height: '16px',
                    borderRadius: '3px',
                    backgroundColor: checked ? '#4CAF50' : 'white',
                    cursor: 'pointer',
                    border: '1px solid #999',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    zIndex: 100
                }}
            >
                {checked && (
                    <div style={{ color: 'white', fontSize: '12px' }}>✓</div>
                )}
            </div>

            <Handle type="target" position={Position.Top} />

            {/* Text container */}
            <div
                style={{
                    width: 'calc(100% - 30px)', // Account for margin and checkbox
                    marginLeft: '20px',
                    whiteSpace: 'normal',
                    wordBreak: 'normal',
                    overflowWrap: 'normal',
                    hyphens: 'none',
                    textAlign: 'center', // Center the text horizontally
                    fontSize: '12px'
                }}
            >
                <span>{data.label}</span>
            </div>

            <Handle type="source" position={Position.Bottom} />
        </div>
    );
};

export default NodeWithScrew;
