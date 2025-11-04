// App.jsx - Main component
import { useState, useEffect, useRef, useCallback } from "react";
import { fetchData, sendData } from './api';

// Import components
import VideoPlayer from './components/VideoPlayer';
import QuestionAnswerPanel from './components/QuestionAnswerPanel';
import ActionListPanel from './components/ActionListPanel';
import InstructionPanel from './components/InstructionPanel';
import NavigationControls from './components/NavigationControls';

export default function App() {
    const [filename, setFilename] = useState("samples_10");
    const [idx, setIdx] = useState(0);
    const [example, setExample] = useState(null);
    const [total, setTotal] = useState(0);
    const [ids, setIds] = useState([]);
    const [result, setResult] = useState({});
    const filenames = [
        "samples_10",
    ];  // hard-coded

    // panel
    const [activePanel, setActivePanel] = useState('parts'); // 'graph_original', 'graph_step'
    const [activeRecordingPanel, setActiveRecordingPanel] = useState('video'); // 'list'

    // Timer state for tracking elapsed time
    const [elapsedTime, setElapsedTime] = useState(0);
    const timerRef = useRef(null);
    const [isTimerRunning, setIsTimerRunning] = useState(false);

    // fetch data whenever idx changes
    useEffect(() => {
        const handleFetchData = async () => {
            try {
                const data = await fetchData(filename, idx);
                setExample(data.example);
                setTotal(data.total);
                setIds(data.ids);
                setResult(JSON.parse(JSON.stringify(data.example.verification)));

                // Reset the timer when a new example is loaded
                setElapsedTime(0);

                // Clear any existing timer
                if (timerRef.current) {
                    clearInterval(timerRef.current);
                }

                // Don't automatically start the timer anymore
                setIsTimerRunning(false);

            } catch (error) {
                console.error('Error fetching data:', error);
            }
        };

        if (filename && idx !== undefined) {
            handleFetchData();  // Only fetch data if idx is available
        }

        // Clear the timer when component unmounts or idx changes
        return () => {
            if (timerRef.current) {
                clearInterval(timerRef.current);
            }
        };
    }, [idx, filename]);

    useEffect(() => {
        // when idx or filename change, reset panels to default
        setActivePanel('parts');
        setActiveRecordingPanel('video');
    }, [idx, filename]);

    const handleFileChange = (e) => {
        const newFilename = e.target.value;
        setFilename(newFilename);
        setIdx(0);
    };

    const handleIdxChange = (e) => {
        const newIdx = parseInt(e.target.value, 10);
        setIdx(newIdx);
    };

    // Timer control functions
    const startTimer = useCallback(() => {
        if (!isTimerRunning) {
            setIsTimerRunning(true);
            timerRef.current = setInterval(() => {
                setElapsedTime(prev => prev + 1);
            }, 1000);
        }
    }, [isTimerRunning]);

    const stopTimer = useCallback(() => {
        if (isTimerRunning) {
            setIsTimerRunning(false);
            if (timerRef.current) {
                clearInterval(timerRef.current);
            }
        }
    }, [isTimerRunning]);

    const resetTimer = useCallback(() => {
        stopTimer();
        setElapsedTime(0);
    }, [stopTimer]);

    const handleNext = useCallback(() => {
        if (!total || total === 0) {
            console.warn("total is not set yet!");
            return;
        }

        if (idx < total - 1) {
            setIdx(idx + 1); // Move to next video and graph task
        } else {
            alert("This is the last example!");
        }
    }, [idx, total]);

    const handleBack = useCallback(() => {
        if (0 < idx) {
            setIdx(idx - 1); // Move to the prev video and graph task
        } else {
            alert("This is the first example!");
        }
    }, [idx]);

    const handleSave = useCallback(async () => {

        // check annotation
        let message = null;
        if (result.question.valid === undefined || result.question.valid === null) {
            message = "Missing Question Annotation!";
        } else if (result.question.valid === true) {
            // check question
            if (
                result.question.multimodal === undefined ||
                result.question.multimodal === null ||
                result.question.procedural === undefined ||
                result.question.procedural === null
            ) {
                message = "Missing Question Annotation!";
            } else {
                // check answers
                result.answers.forEach((_answer) => {
                    if (_answer.correct === undefined || _answer.correct === null) {
                        message = "Missing Answer Annotation!";
                    }
                });

                // check if at least one yes or one human-written answer
                let flag_at_least_one_answer = false;
                result.answers.forEach((_answer) => {
                    if (_answer.correct === true) {
                        flag_at_least_one_answer = true;
                    }
                })
                if (result.comment !== undefined && result.comment !== null) {
                    flag_at_least_one_answer = true;
                }

                if (message === null && !flag_at_least_one_answer) {
                    message = " Missing Human-written Answer!";
                }
            }
        }

        if (message === null) {
            const dataToSend = {
                filename,
                idx,
                result,
            };
            try {
                const response = await sendData(dataToSend);
                console.log("Response from backend:", response);
                handleNext();
            } catch (error) {
                console.error("Error sending data:", error);
            }
        } else {
            alert(message);
        }
    }, [filename, idx, result]);

    // Format timestamp for display (convert seconds to MM:SS format)
    const formatTimestamp = useCallback((seconds) => {
        const mins = Math.floor(seconds / 60);
        const secs = Math.floor(seconds % 60);
        return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
    }, []);

    // Handle state updates for question validation
    const handleQuestionCheck = useCallback((key, value) => {
        setResult(prevResult => ({
            ...prevResult,
            "question": {
                ...(prevResult.question || {}),
                [key]: value
            }
        }));
    }, []);

    // Handle state updates for answer validation
    const handleAnswerCheck = useCallback((idx, value) => {
        setResult(prevResult => {
            if (!prevResult.answers || !prevResult.answers[idx]) return prevResult;

            // Create a copy of the answers array
            const newAnswers = [...prevResult.answers];
            // Update the specific index in the array
            newAnswers[idx] = {
                ...newAnswers[idx],
                correct: value
            };
            // Return the updated state object
            return {
                ...prevResult,
                answers: newAnswers
            };
        });
    }, []);

    // Handle comment updates
    const handleCommentChange = useCallback((comment) => {
        setResult(prevResult => ({
            ...prevResult,
            comment
        }));
    }, []);

    // Toggle panel handler
    const togglePanel = useCallback((panel) => {
        setActivePanel(panel);
    }, []);

    const toggleRecordingPanel = useCallback((panel) => {
        setActiveRecordingPanel(panel);
    }, []);

    return (
        <div className="container-fluid">
            <div className="row">
                <div
                    className="col-md-6 d-flex flex-column"
                    style={{ height: "100vh", overflow: "hidden" }}
                >
                    <QuestionAnswerPanel
                        example={example}
                        result={result}
                        elapsedTime={elapsedTime}
                        formatTimestamp={formatTimestamp}
                        isTimerRunning={isTimerRunning}
                        startTimer={startTimer}
                        stopTimer={stopTimer}
                        resetTimer={resetTimer}
                        handleFileChange={handleFileChange}
                        handleIdxChange={handleIdxChange}
                        handleQuestionCheck={handleQuestionCheck}
                        handleAnswerCheck={handleAnswerCheck}
                        handleCommentChange={handleCommentChange}
                        totalExamples={total}
                        currentIndex={idx}
                        currentFilename={filename}
                        ids={ids}
                        filenames={filenames}
                    />

                    <NavigationControls
                        handleBack={handleBack}
                        handleNext={handleNext}
                        handleSave={handleSave}
                    />
                </div>

                <div className="col-md-6 d-flex flex-column">
                    {/* Conditionally render the active panel with toggle functionality in header */}
                    {activePanel === 'parts' ? (
                        <InstructionPanel
                            idx={idx}
                            imagePath={`images/${example?.filepath_part}`}
                            activePanel={activePanel}
                            togglePanel={togglePanel}
                        />

                    ) : activePanel === 'graph_original' ? (
                        <InstructionPanel
                            idx={idx}
                            imagePath={`raw/${example?.filepath_graph_original}`}
                            activePanel={activePanel}
                            togglePanel={togglePanel}
                        />
                    ) : (
                        <InstructionPanel
                            idx={idx}
                            imagePath={`status/${example?.filepath_graph}`}
                            activePanel={activePanel}
                            togglePanel={togglePanel}
                        />
                    )}

                    {activeRecordingPanel === 'video' ? (
                        <VideoPlayer
                            idx={idx}
                            videoData={example?.video}
                            sequenceId={example?.sequence_id}
                            activePanel={activeRecordingPanel}
                            togglePanel={toggleRecordingPanel}
                        />
                    ) : (
                        <ActionListPanel
                            idx={idx}
                            actions={example?.actions}
                            activePanel={activeRecordingPanel}
                            togglePanel={toggleRecordingPanel}
                        />
                    )}
                </div>
            </div>
        </div>
    );
}
