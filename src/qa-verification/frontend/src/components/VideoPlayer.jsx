// components/VideoPlayer.js
import React, { useState, useEffect, useRef, memo } from 'react';

// Using memo to prevent unnecessary re-renders
const VideoPlayer = memo(function VideoPlayer({ idx, videoData, sequenceId, activePanel, togglePanel }) {
    const [selectedAngle, setSelectedAngle] = useState("");
    const qualityLevels = ['1080p']
    const [selectedQuality, setSelectedQuality] = useState("");
    const [captionUrl, setCaptionUrl] = useState(null);
    const videoRef = useRef(null);
    const lastPlaybackPositionRef = useRef(0);

    // Set initial selected file when component mounts or when video data changes
    useEffect(() => {
        if (videoData && videoData.angles && videoData.angles.length > 0) {
            setSelectedAngle(videoData.angles[1]);  // 0: center-right
            setSelectedQuality('1080p');
            lastPlaybackPositionRef.current = videoData.start || 0;
        }

        // Clear previous caption URL when idx changes
        if (captionUrl) {
            URL.revokeObjectURL(captionUrl);
            setCaptionUrl(null);
        }

        // Create caption URL if available
        if (videoData && videoData.caption) {
            const vttData = videoData.caption;
            const blob = new Blob([vttData], { type: "text/vtt" });
            const url = URL.createObjectURL(blob);
            setCaptionUrl(url);

            // Cleanup previous URL
            return () => URL.revokeObjectURL(url);
        }
    }, [videoData, idx, sequenceId]);

    // Handle video time bounds and constraints
    useEffect(() => {
        if (!videoRef.current || !selectedAngle || !selectedQuality || !videoData) return;

        const videoElement = videoRef.current;
        videoElement.load();

        // Set initial time position
        videoElement.currentTime = lastPlaybackPositionRef.current;

        const handleTimeUpdate = () => {
            // Save current position regularly
            lastPlaybackPositionRef.current = videoElement.currentTime;

            // Check if we've reached the end boundary
            if (videoData.end && videoElement.currentTime >= videoData.end) {
                videoElement.pause();
                // Reset to start time
                // videoElement.currentTime = videoData.start || 0;
            }
        };

        videoElement.addEventListener('timeupdate', handleTimeUpdate);

        return () => {
            videoElement.removeEventListener('timeupdate', handleTimeUpdate);
        };
    }, [selectedAngle, selectedQuality, sequenceId, idx]);

    // Camera angle mapping function
    const filename2angle = (filename) => {
        const filenameMapping = {
            // Exo cameras
            "C10095_rgb.mp4": "Center-Right", // (v1)
            "C10115_rgb.mp4": "Above", //  (v2)
            "C10118_rgb.mp4": "Center", // (v3)
            "C10119_rgb.mp4": "Center-Left", // (v4)
            "C10379_rgb.mp4": "Left", // (v5)
            "C10390_rgb.mp4": "Top-Right", // (v6)
            "C10395_rgb.mp4": "Right", // (v7)
            "C10404_rgb.mp4": "Top-Left", // (v8)

            // Ego cameras (B/W)
            "HMC_84346135_mono10bit.mp4": "Ego (TL)", // (e1)
            "HMC_21176875_mono10bit.mp4": "Ego (TL)", // (e1)
            "HMC_84347414_mono10bit.mp4": "Ego (TR)", // (e2)
            "HMC_21176623_mono10bit.mp4": "Ego (TR)", // (e2)
            "HMC_84355350_mono10bit.mp4": "Ego (BR)", // (e3)
            "HMC_21110305_mono10bit.mp4": "Ego (BR)", // (e3)
            "HMC_84358933_mono10bit.mp4": "Ego (BL)", // (e4)
            "HMC_21179183_mono10bit.mp4": "Ego (BL)", // (e4)
        };
        return filenameMapping[filename] || filename;
    };

    // Handle camera angle selection change
    const handleAngleChange = (e) => {
        // Store current playback position before changing angle
        if (videoRef.current) {
            lastPlaybackPositionRef.current = videoRef.current.currentTime;
        }
        console.log(e.target.value);
        setSelectedAngle(e.target.value);
    };

    // Handle video quality selection change
    const handleQualityChange = (e) => {
        // Store current playback position before changing quality
        if (videoRef.current) {
            lastPlaybackPositionRef.current = videoRef.current.currentTime;
        }
        console.log(e.target.value);
        setSelectedQuality(e.target.value);
    };

    // Format timestamp for display
    const formatTimestamp = (seconds) => {
        const mins = Math.floor(seconds / 60);
        const secs = Math.floor(seconds % 60);
        return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
    };

    // If video data isn't available yet, render nothing or a loading indicator
    if (!videoData || !sequenceId) {
        return (
            <div
                className="card flex-grow-1 d-flex flex-column align-items-center justify-content-center"
                style={{
                    flexBasis: "49%",
                    maxHeight: "49vh",
                    borderLeft: '3px solid #9b59b6'
                }}
            >
                <p style={{ color: '#666' }}>Loading video data...</p>
            </div>
        );
    }

    const getVideoSource = () => {

        if (!selectedAngle || !sequenceId) return '';

        let videoSource = '';
        if (selectedAngle.startsWith("C10")) {
            if (selectedQuality === '1080p') {
                videoSource = `recordings/${sequenceId}/${selectedAngle}`;
            } else {
                videoSource = `recordings-resolution/${selectedQuality}/${sequenceId}/${selectedAngle}`;
            }
        } else {
            videoSource = `recordings/${sequenceId}/${selectedAngle}`;
        }
        return videoSource
    };

    return (
        <div
            className="card flex-grow-1 d-flex flex-column"
            style={{
                flexBasis: "49%",
                maxHeight: "49vh",
                borderLeft: '3px solid #9b59b6'
            }}
        >
            <div
                className="card-header d-flex align-items-center flex-wrap gap-2 justify-content-between"
                style={{
                    backgroundColor: '#f8f9fa',
                    borderBottom: '1px solid #dee2e6',
                    padding: '8px 12px'
                }}
            >
                <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                    <label className="mb-0" style={{ fontWeight: 'bold' }}>Angle:</label>
                    <select
                        className="form-select form-select-sm"
                        value={selectedAngle}
                        onChange={handleAngleChange}
                        style={{ width: '180px' }}
                    >
                        <option value="">Select angle</option>
                        {videoData.angles.map((file) => (
                            <option key={file} value={file}>
                                {filename2angle(file)}
                            </option>
                        ))}
                    </select>
                </div>
                <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                    <select
                        className="form-select form-select-sm"
                        value={selectedQuality}
                        onChange={handleQualityChange}
                        style={{ width: '100px' }}
                        >
                        <option value="1080p">1080p</option>
                    </select>
                </div>

                <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                    <label style={{ fontWeight: 'bold', marginBottom: 0 }}>Range:</label>
                    <span style={{ fontSize: '0.9rem' }}>
                        {formatTimestamp(videoData.start || 0)} ~ {formatTimestamp(videoData.end || 0)}
                    </span>
                </div>

                {/* Toggle buttons with improved styling */}
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
                className="card-body d-flex justify-content-center align-items-center overflow-hidden p-2 flex-grow-1"
                style={{ backgroundColor: '#f8f9fa' }}
            >
                {selectedAngle ? (
                    <video
                        ref={videoRef}
                        playsInline
                        controls
                        preload="auto"  // or "metadata" for slower connections
                        className="w-100 h-100"
                        style={{
                            objectFit: "contain",
                            borderRadius: "4px",
                            border: '1px solid #dee2e6'
                        }}
                    >
                        <source
                            src={getVideoSource()} type="video/mp4"
                        />
                        {captionUrl && (
                            <track
                                key={`caption-${idx}-${sequenceId}`}
                                src={captionUrl}
                                kind="subtitles"
                                srcLang="en"
                                label="English"
                                default
                            />
                        )}
                    </video>
                ) : (
                    <div style={{
                        color: '#666',
                        textAlign: 'center',
                        padding: '20px',
                        border: '1px dashed #ccc',
                        borderRadius: '4px',
                        backgroundColor: '#fff'
                    }}>
                        Select a camera angle to view video
                    </div>
                )}
            </div>
        </div>
    );
});

export default VideoPlayer;
