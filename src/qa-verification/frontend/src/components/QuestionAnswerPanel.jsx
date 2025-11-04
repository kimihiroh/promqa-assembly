// components/QuestionAnswerPanel.js
import React, { memo } from 'react';

// Button component for Yes/No selections with subtle improvements
const ActionButton = ({ onClick, isActive, type, label, disabled }) => (
  <button
    onClick={onClick}
    disabled={disabled}
    style={{
      padding: '4px 8px',
      backgroundColor: isActive ? (type === 'yes' ? '#4CAF50' : '#f44336') : '#f1f1f1',
      color: isActive ? 'white' : 'black',
      border: '1px solid #ccc',
      borderRadius: '4px',
      cursor: disabled ? 'not-allowed' : 'pointer',
      marginRight: type === 'yes' ? '10px' : '0',
      fontWeight: isActive ? 'bold' : 'normal',
      opacity: disabled ? 0.6 : 1
    }}
  >
    {label}
  </button>
);

// Question validation component with minimal styling
const QuestionValidation = ({ result, handleQuestionCheck }) => {
  // Check if question is valid - this will determine if other fields are disabled
  const isQuestionInvalid = result.question.valid === false;

  return (
    <div style={{
      padding: '8px',
      marginBottom: '10px',
      borderLeft: '3px solid #3498db'
    }}>
      <div style={{ fontWeight: 'bold', marginBottom: '5px' }}>Question Validation</div>

      <ValidationRow
        label="Valid?"
        helpText="answerability, clarity, naturalness, external knowledge, etc..."
        isActive={result.question.valid}
        onYesClick={() => handleQuestionCheck('valid', true)}
        onNoClick={() => handleQuestionCheck('valid', false)}
      />

      <ValidationRow
        label="Multimodal?"
        helpText="requires information from both video and instruction"
        isActive={result.question.multimodal}
        onYesClick={() => handleQuestionCheck('multimodal', true)}
        onNoClick={() => handleQuestionCheck('multimodal', false)}
        disabled={isQuestionInvalid}
      />

      <ValidationRow
        label="Procedural?"
        helpText="about steps or about flows of steps"
        isActive={result.question.procedural}
        onYesClick={() => handleQuestionCheck('procedural', true)}
        onNoClick={() => handleQuestionCheck('procedural', false)}
        disabled={isQuestionInvalid}
      />
    </div>
  );
};

// Single validation row with improved layout
const ValidationRow = ({ label, helpText, isActive, onYesClick, onNoClick, disabled = false }) => {
    const getMessage = () => {
        if (label === 'Valid?') {
            return "Press Save & Move Next";
        }
        return null;
    };

    return (
        <div style={{
          display: 'flex',
          alignItems: 'center',
          marginBottom: '4px',
          opacity: disabled ? 0.7 : 1
        }}>
            <div style={{
              display: 'flex',
              alignItems: 'center',
              width: '100%',
              justifyContent: 'space-between'
            }}>
                <label style={{
                  display: 'flex',
                  alignItems: 'center',
                  flex: '1'
                }}>
                    <div style={{
                      marginRight: '5px',
                      fontWeight: '500',
                      minWidth: '90px'
                    }}>{label}</div>
                    <div className='form-text' style={{
                      color: '#666',
                      fontSize: '0.85rem',
                      fontStyle: 'italic'
                    }}>{helpText}</div>
                </label>
                <div style={{ display: 'flex' }}>
                    <ActionButton
                        onClick={onYesClick}
                        isActive={isActive === true}
                        type="yes"
                        label="Yes"
                        disabled={disabled}
                    />
                    <ActionButton
                        onClick={onNoClick}
                        isActive={isActive === false}
                        type="no"
                        label="No"
                        disabled={disabled}
                    />
                </div>
                {label === 'Valid?' && isActive === false && (
                    <div className="message-container" style={{
                      marginLeft: '10px',
                      color: '#e74c3c'
                    }}>{getMessage()}</div>
                )}
            </div>
        </div>
    );
};

// Answer section component with minimal styling
const AnswerSection = ({ example, result, handleAnswerCheck }) => {
  // Check if answers should be disabled (when question is marked as invalid)
  const isDisabled = result.question.valid === false;

  return (
    <div style={{
      padding: '8px',
      marginBottom: '10px',
      borderLeft: '3px solid #2ecc71',
      opacity: isDisabled ? 0.7 : 1
    }}>
      <div style={{ fontWeight: 'bold', marginBottom: '5px' }}>Answers</div>

      {example.answers && (
        <div className="flex" style={{ display: 'flex', flexDirection: 'column' }}>
          {example.answers.map((_answer, index) => (
            <div key={index} style={{
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'space-between',
              padding: '3px 0',
              borderBottom: index !== example.answers.length - 1 ? '1px solid #eee' : 'none'
            }}>
              <label style={{
                marginRight: '5px',
                flex: '1'
              }}>
                * {_answer.trim()}
              </label>
              <div>
                <ActionButton
                  onClick={() => handleAnswerCheck(index, true)}
                  isActive={result.answers[index].correct === true}
                  type="yes"
                  label="Yes"
                  disabled={isDisabled}
                />
                <ActionButton
                  onClick={() => handleAnswerCheck(index, false)}
                  isActive={result.answers[index].correct === false}
                  type="no"
                  label="No"
                  disabled={isDisabled}
                />
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
};

// Header component with minimal styling
const PanelHeader = ({
    currentIndex,
    currentFilename,
    totalExamples,
    ids,
    filenames,
    formatTimestamp,
    elapsedTime,
    isTimerRunning,
    startTimer,
    stopTimer,
    resetTimer,
    handleFileChange,
    handleIdxChange,
}) => (
    <div
        className="card-header d-flex align-items-center flex-wrap gap-3"
        style={{
            backgroundColor: '#f8f9fa',
            borderBottom: '1px solid #dee2e6',
            padding: '8px 12px'
        }}>
        <label className="me-2 mb-0" style={{ fontWeight: 'bold' }}>
            Filename:
        </label>
        <select
            className="form-select form-select-sm"
            value={currentFilename || "sample"}
            onChange={handleFileChange}
            style={{ width: '100px' }}
        >
            <option value="">Select File</option>
            {filenames.map((option) => (
                <option key={option} value={option}>
                    {option}
                </option>
            ))}
        </select>

        <label className="me-2 mb-0" style={{ fontWeight: 'bold' }}>
            ID:
        </label>
        <select
            className="form-select form-select-sm"
            value={currentIndex !== undefined && currentIndex !== '' ? currentIndex : (ids.length > 0 ? ids[0] : '')}
            onChange={handleIdxChange}
            style={{ width: '70px' }}
        >
            <option value="">ID</option>
            {ids.map((option) => (
                <option key={option} value={option}>
                    {parseInt(option) + 1}
                </option>
            ))}
        </select>
        <label>
            / {totalExamples}
        </label>

        <span className="badge bg-secondary">
            Clock: {formatTimestamp(elapsedTime)}
        </span>
        <div className="ms-auto d-flex gap-2">
            <button
                className={`btn btn-sm ${isTimerRunning ? "btn-warning" : "btn-success"}`}
                onClick={isTimerRunning ? stopTimer : startTimer}
            >
                {isTimerRunning ? "Stop" : "Start"}
            </button>
            <button
                className="btn btn-sm btn-secondary"
                onClick={resetTimer}
            >
                Reset
            </button>
        </div>
    </div>
);

// Comments section component with minimal styling
const CommentsSection = ({ result, handleCommentChange }) => {
  // Check if comments should be disabled (when question is marked as invalid)
  const isDisabled = result.question.valid === false;

  return (
    <div style={{
      padding: '8px',
      marginBottom: '10px',
      borderLeft: '3px solid #f1c40f',
    }}>
      <div style={{ fontWeight: 'bold', marginBottom: '5px' }}>
          {isDisabled ? "Comment" : "Additional Answers"}
      </div>

      <input
        className="form-control"
        type="text"
        value={result.comment || ""}
        onChange={(e) => handleCommentChange(e.target.value)}
        placeholder="Use a semi-colon (;) in between..."
        style={{
          padding: '6px',
          borderRadius: '4px',
          border: '1px solid #ddd',
          marginBottom: '8px'
        }}
      />
      <div>
        {result.comment ?
          result.comment.split(';').map((_comment, index) => (
            <div key={index} style={{
              padding: '2px 0'
            }}>* {_comment.trim()}</div>
          ))
          : ""
        }
      </div>
    </div>
  );
};

// Main component with improved overall styling
const QuestionAnswerPanel = memo(function QuestionAnswerPanel({
    example,
    result,
    elapsedTime,
    formatTimestamp,
    isTimerRunning,
    startTimer,
    stopTimer,
    resetTimer,
    handleFileChange,
    handleIdxChange,
    handleQuestionCheck,
    handleAnswerCheck,
    handleCommentChange,
    totalExamples,
    currentIndex,
    currentFilename,
    ids,
    filenames,
}) {
    if (!example || !result) {
        return (
            <div className="card d-flex flex-column" style={{ height: "100%" }}>
                <div className="card-header">Loading...</div>
                <div className="card-body">Loading question and answer data...</div>
            </div>
        );
    }

    return (
        <div className="card d-flex flex-column" style={{ height: "100%" }}>
            <PanelHeader
                currentIndex={currentIndex}
                currentFilename={currentFilename}
                totalExamples={totalExamples}
                ids={ids}
                filenames={filenames}
                formatTimestamp={formatTimestamp}
                elapsedTime={elapsedTime}
                isTimerRunning={isTimerRunning}
                startTimer={startTimer}
                stopTimer={stopTimer}
                resetTimer={resetTimer}
                handleFileChange={handleFileChange}
                handleIdxChange={handleIdxChange}
            />

            <div className="card-body d-flex flex-column p-2 overflow-auto">
                {/* Question Section */}
                <div style={{
                  padding: '8px',
                  marginBottom: '10px',
                  borderLeft: '3px solid #3498db'
                }}>
                    <div style={{ fontWeight: 'bold', marginBottom: '5px' }}>Question</div>
                    {example.question && <p style={{ margin: 0 }}>{example.question}</p>}
                </div>

                {/* Question Validation Section */}
                {result.question && (
                    <QuestionValidation
                      result={result}
                      handleQuestionCheck={handleQuestionCheck}
                    />
                )}

                {/* Answer Section */}
                <AnswerSection
                  example={example}
                  result={result}
                  handleAnswerCheck={handleAnswerCheck}
                />

                {/* Additional Comments Section */}
                <CommentsSection
                  result={result}
                  handleCommentChange={handleCommentChange}
                />
            </div>
        </div>
    );
});

export default QuestionAnswerPanel;
