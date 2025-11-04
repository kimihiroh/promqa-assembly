import { useState, useEffect, useRef } from "react";
import {
    Panel,
    useNodesState,
    useEdgesState,
} from '@xyflow/react';
import GraphEditor from "./GraphEditor";
import { fetchData, sendData } from './api'

export default function App() {

    const [annotatorId, setAnnotatorId] = useState("kh");
    const annotatorIds = ['all'];
    const [idx, setIdx] = useState(0);
    const [total, setTotal] = useState(0);
    const [toyId, setToyId] = useState(null);
    const [toyIds, setToyIds] = useState([]);
    const [imagePath, setImagePath] = useState(null);
    const [videoDict, setVideoDict] = useState(null);
    const [folders, setFolders] = useState([]);
    const [selectedFolder, setSelectedFolder] = useState("");
    const [selectedFile, setSelectedFile] = useState("");
    const [captionUrl, setCaptionUrl] = useState(null);
    const videoRef = useRef(null); // Reference for video element
    const [nodes, setNodes, onNodesChange] = useNodesState([]);
    const [edges, setEdges, onEdgesChange] = useEdgesState([]);

    function handleCheck(nodeId, isChecked) {
        // Create a new nodes array with the updated checked state
        setNodes(nodes =>
            nodes.map(node =>
                node.id === nodeId
                    ? { ...node, data: { ...node.data, checked: isChecked } }
                    : node
            )
        );
    }

    // fetch data whenever idx changes
    useEffect(() => {
        const handleFetchData = async () => {
            try {
                const data = await fetchData(annotatorId, idx);
                const example = data.example;

                setToyId(example.toy_id)

                // add handleCheck to each node variable
                const nodesWithHandlers = example.nodes.map(node => ({
                    ...node,
                    data: {
                        ...node.data,
                        onCheck: handleCheck
                    }
                }));
                setNodes(nodesWithHandlers);

                setEdges(example.edges)
                setImagePath(example.filepath_image)
                setVideoDict(example.videos)

                setTotal(data.total)
                setToyIds(data.ids)

                const folderNames = Object.keys(example.videos);
                setFolders(folderNames);
                setSelectedFolder(folderNames[0]);
                setSelectedFile(example.videos[folderNames[0]]['angles'][0]);

            } catch (error) {
                console.error('Error fetching data:', error)
            }
        };
        if (idx !== undefined) {
            handleFetchData();  // Only fetch data if idx is available
        }
    }, [idx, annotatorId]);

    const handleNext = () => {
        if (!total || total === 0) {
            console.warn("total is not set yet!");
            return;
        }

        if (idx < total-1) {
            // handleSaveSilent();
            setIdx(idx + 1); // Move to next video and graph task
        } else {
            alert("This is the last example!");
        }
    };

    const handleBack = () => {
        if (0 < idx) {
            // handleSaveSilent();
            setIdx(idx - 1); // Move to the prev video and graph task
        } else {
            alert("This is the first example!");
        }
    };

    const handleSave = async () => {
        const dataToSend = {annotatorId, idx, nodes, edges};
        try {
            const response = await sendData(dataToSend);
            alert("Saved the graph information.")
            console.log("Response from backend:", response)
        } catch (error) {
            console.error("Error sending data:", error)
        }
    };

    const handleSaveSilent = async () => {
        const dataToSend = {annotatorId, idx, nodes, edges};
        try {
            const response = await sendData(dataToSend);
        } catch (error) {
            console.error("Error sending data:", error)
        }
    };

    const handleAnnotatorIdChange = (e) => {
        // handleSaveSilent();
        const newAnnotatorId = e.target.value;
        setAnnotatorId(newAnnotatorId);
    };

    const handleToyIdChange = (e) => {
        // handleSaveSilent();
        const newToyId = e.target.value;
        setToyId(newToyId);
        const newIdx = toyIds.indexOf(newToyId);
        setIdx(newIdx);
    };

    const handleFolderChange = (e) => {
        const newFolder = e.target.value;
        setSelectedFolder(newFolder);

        if (videoDict && videoDict[newFolder]?.angles?.length > 0) {
            setSelectedFile(videoDict[newFolder]['angles'][0]);
        } // Default to the first file in the new folder
    };

    const handleFileChange = (e) => {
        setSelectedFile(e.target.value);
    };

    // when selectedFolder is changed, set the selectedFile as default one
    useEffect(() => {
        if (selectedFolder && videoDict && videoDict[selectedFolder]?.angles?.length > 0) {
            setSelectedFile(videoDict[selectedFolder]['angles'][0]);
        }

        // Set video start time
        if (videoDict) {
            if (videoRef.current) {
                videoRef.current.currentTime = videoDict[selectedFolder]['start'] || 0;
            }
        }

        if (videoDict) {
            const vttData = videoDict[selectedFolder]['caption'];
            if (vttData) {
                const blob = new Blob([vttData], { type: "text/vtt" });
                const url = URL.createObjectURL(blob);
                setCaptionUrl(url);

                // Cleanup previous URL
                return () => URL.revokeObjectURL(url);
            }
        }

    }, [selectedFolder]);

    useEffect(() => {
        if (videoDict) {
            if (videoRef.current) {
                videoRef.current.currentTime = videoDict[selectedFolder]['start'] || 0;
            }
        }
        if (videoDict) {
            const vttData = videoDict[selectedFolder]['caption'];
            if (vttData) {
                const blob = new Blob([vttData], { type: "text/vtt" });
                const url = URL.createObjectURL(blob);
                setCaptionUrl(url);

                // Cleanup previous URL
                return () => URL.revokeObjectURL(url);
            }
        }
    }, [selectedFile]);

    // folder name formetter
    const formatName = (name) => {
        return name
            .replace(/nusar-\d{4}_action_both_\d{4}-/, "") // Remove "nusar-2021_" or similar
            .replace(/user_id_/, "")
            .replace(/_/g, " "); // Replace underscores with spaces for readability
    };

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
        "HMC_84346135_mono10bit.mp4": "Ego (Top-Left, B/W)", // (e1)
        "HMC_21176875_mono10bit.mp4": "Ego (Top-Left, B/W)", // (e1)
        "HMC_84347414_mono10bit.mp4": "Ego (Top-Right, B/W)", // (e2)
        "HMC_21176623_mono10bit.mp4": "Ego (Top-Right, B/W)", // (e2)
        "HMC_84355350_mono10bit.mp4": "Ego (Bottom-Right, B/W)", // (e3)
        "HMC_21110305_mono10bit.mp4": "Ego (Bottom-Right, B/W)", // (e3)
        "HMC_84358933_mono10bit.mp4": "Ego (Bottom-Left, B/W)", // (e4)
        "HMC_21179183_mono10bit.mp4": "Ego (Bottom-Left, B/W)", // (e4)
    };


    const filename2angle = (filename) => {
        return filenameMapping[filename] || filename;
    };

    return (
        <div className="container-fluid">
            <div className="row">
                <div className="col-md-6">
                    <div className="card h-100 d-flex flex-column">
                        <div className="card-header d-flex align-items-center flex-wrap gap-3">
                            <label className="me-2 mb-0">Annotator ID:</label>
                            <select
                                className="form-select form-select-sm"
                                value={annotatorId || ""}
                                onChange={handleAnnotatorIdChange}
                                style={{ width: '100px' }}
                            >
                                <option value="">Select Annotator</option>
                                {annotatorIds.map((option) => (
                                    <option key={option} value={option}>
                                        {option}
                                    </option>
                                ))}
                            </select>

                            <label className="me-2 mb-0">Toy ID ({idx+1} / {total}):</label>
                            <select
                                className="form-select form-select-sm"
                                value={toyId || ""}
                                onChange={handleToyIdChange}
                                style={{ width: '100px' }}
                            >
                                <option value="">Select Toy</option>
                                {toyIds.map((option) => (
                                    <option key={option} value={option}>
                                        {option}
                                    </option>
                                ))}
                            </select>
                        </div>
                        <div className="card-body d-flex flex-column p-2 flex-grow-1">
                            <div className="flex-grow-1" >
                                {
                                    nodes && (
                                        <GraphEditor
                                            idx={idx}
                                            nodes={nodes}
                                            edges={edges}
                                            setNodes={setNodes}
                                            setEdges={setEdges}
                                            onNodesChange={onNodesChange}
                                            onEdgesChange={onEdgesChange}
                                            />
                                    )
                                }
                            </div>
                            <div
                                className="mt-auto d-flex justify-content-between"
                                >
                                <div className="d-flex gap-2">
                                    <button
                                        className="btn btn-outline-secondary"
                                        onClick={handleBack}
                                        >
                                        <i className="bi bi-arrow-left">Back</i>
                                    </button>
                                    <button
                                        className="btn btn-outline-primary"
                                        onClick={handleNext}
                                        >
                                        Next<i className="bi bi-arrow-right"></i>
                                    </button>
                                </div>
                                <button className="btn btn-success" onClick={handleSave}>
                                        <i className="bi bi-save"> Save</i>
                                </button>
                            </div>
                        </div>
                    </div>
                </div>

                <div className="col-md-6 d-flex flex-column">
                    <div
                        className="card flex-grow-1"
                        style={{ flexBasis: "50%", maxHeight: "50vh" }}
                        >
                        <div
                            className="card-body d-flex justify-content-center align-items-center overflow-hidden p-3"
                            >
                            <img
                                src={`images/${imagePath}`}
                                className="w-100 h-100"
                                style={{
                                    objectFit: "contain",
                                    borderRadius: "4px"
                                }}
                            />
                        </div>
                    </div>
                    <div
                        className="card flex-grow-1 d-flex flex-column"
                        style={{ flexBasis: "50%", maxHeight: "50vh" }}
                        >
                        <div className="card-header d-flex align-items-center flex-wrap gap-3">
                            <label className="me-2 mb-0">Recording:</label>
                            <select
                                className="form-select form-select-sm me-2"
                                value={selectedFolder || ""}
                                onChange={handleFolderChange}
                                style={{ width: '250px' }}
                            >
                                <option value="">Select recording</option>
                                {folders.map((folder) => (
                                    <option key={folder} value={folder}>
                                        {formatName(folder)}
                                    </option>
                                ))}
                            </select>

                            {selectedFolder && videoDict?.[selectedFolder]?.angles && (
                                <>
                                    <label className="me-2 mb-0">Angle:</label>
                                    <select
                                        className="form-select form-select-sm"
                                        value={selectedFile || ""}
                                        onChange={handleFileChange}
                                        style={{ width: '200px' }}
                                        >
                                        <option value="">Select angle</option>
                                        {videoDict[selectedFolder]['angles']
                                            .map((file) => (
                                            <option key={file} value={file}>
                                                {filename2angle(file)}
                                            </option>
                                        ))}
                                    </select>
                                </>
                            )}
                        </div>
                        <div
                            className="card-body d-flex justify-content-center align-items-center overflow-hidden p-3 flex-grow-1"
                            >
                            {/* Video Player */}
                            {selectedFile && (
                                <video
                                    id="player"
                                    key={`${selectedFolder}-${selectedFile}`}
                                    ref={videoRef}
                                    playsInline
                                    controls
                                    className="w-100 h-100"
                                    style={{
                                        objectFit: "contain",
                                        borderRadius: "4px"
                                    }}
                                    >
                                    <source
                                        src={`recordings/${selectedFolder}/${selectedFile}`}
                                        type="video/mp4"
                                        />
                                    <track
                                        src={captionUrl}
                                        kind="subtitles"
                                        srcLang="en" label="English" default
                                        />
                                </video>
                            )}
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
}
