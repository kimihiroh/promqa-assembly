import React, { memo, useEffect, useRef } from 'react';

const ActionListPanel = memo(function ActionListPanel({ idx, actions, activePanel, togglePanel }) {
    // Use refs to measure and handle the component's layout
    const containerRef = useRef(null);
    const headerRef = useRef(null);
    const contentRef = useRef(null);

    // Setup resize observer to handle dynamic container size changes
    useEffect(() => {
        if (!containerRef.current || !headerRef.current || !contentRef.current) return;

        const updateContentHeight = () => {
            if (containerRef.current && headerRef.current && contentRef.current) {
                const containerHeight = containerRef.current.clientHeight;
                const headerHeight = headerRef.current.clientHeight;

                // Prevent potential layout thrashing by using requestAnimationFrame
                // This batches the DOM changes to the next paint cycle
                requestAnimationFrame(() => {
                    if (contentRef.current) {
                        // Set the content height to be container height minus header height
                        contentRef.current.style.height = `${containerHeight - headerHeight}px`;
                    }
                });
            }
        };

        // Initial setup - delay slightly to ensure all refs are properly set
        setTimeout(updateContentHeight, 0);

        // Create a resize observer to handle container size changes
        // Use a debounce to prevent too many quick updates
        let rafId = null;

        const resizeObserver = new ResizeObserver(() => {
            // Cancel any pending updates
            if (rafId) cancelAnimationFrame(rafId);

            // Schedule a new update
            rafId = requestAnimationFrame(updateContentHeight);
        });

        resizeObserver.observe(containerRef.current);

        // Cleanup
        return () => {
            if (rafId) cancelAnimationFrame(rafId);
            resizeObserver.disconnect();
        };
    }, []);

    return (
        <div
            ref={containerRef}
            className="card"
            style={{
                height: "49%", // Keep the 40% height
                display: "flex",
                flexDirection: "column",
                borderLeft: '3px solid #e67e22',
                position: 'relative', // Important for children positioning
            }}
        >
            <div
                ref={headerRef}
                className="card-header d-flex align-items-center flex-wrap gap-2 justify-content-between"
                style={{
                    backgroundColor: '#f8f9fa',
                    borderBottom: '1px solid #dee2e6',
                    padding: '8px 12px',
                    width: '100%', // Ensure it takes full width
                }}
            >
                <label
                    className="mb-0"
                    style={{
                        fontWeight: 'bold'
                    }}
                >
                    Actions
                </label>
                {/* Toggle buttons in the card header */}
                <div className="btn-group" role="group" aria-label="Panel toggle">
                    <button
                        type="button"
                        className={`btn btn-sm ${activePanel === 'video' ? 'btn-primary' : 'btn-outline-primary'}`}
                        onClick={() => togglePanel('video')}
                        style={{ fontWeight: activePanel === 'video' ? 'bold' : 'normal' }}
                    >
                        Video
                    </button>
                    <button
                        type="button"
                        className={`btn btn-sm ${activePanel === 'list' ? 'btn-primary' : 'btn-outline-primary'}`}
                        onClick={() => togglePanel('list')}
                        style={{ fontWeight: activePanel === 'list' ? 'bold' : 'normal' }}
                    >
                        List
                    </button>
                </div>
            </div>
            <div
                ref={contentRef}
                className="card-body p-2 overflow-auto"
                style={{
                    backgroundColor: '#fcfcfc',
                    width: '100%',
                    overflowY: 'auto', // Enable vertical scrolling
                }}
            >
                {actions ? (
                    <div style={{ display: 'flex', flexDirection: 'column', gap: '2px' }}>
                        {actions.split('\n').map((action, index) => (
                            <div
                                key={index}
                                style={{
                                    padding: '4px 8px',
                                    borderBottom: index !== actions.split('\n').length - 1 ? '1px solid #f0f0f0' : 'none',
                                    lineHeight: '1.4'
                                }}
                            >
                                {action.trim()}
                            </div>
                        ))}
                    </div>
                ) : (
                    <div
                        style={{
                            padding: '15px',
                            textAlign: 'center',
                            color: '#777',
                            border: '1px dashed #ddd',
                            borderRadius: '4px'
                        }}
                    >
                        No actions available
                    </div>
                )}
            </div>
        </div>
    );
});

export default ActionListPanel;
