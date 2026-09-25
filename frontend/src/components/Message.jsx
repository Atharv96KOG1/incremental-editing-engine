import Timeline from './Timeline.jsx'
import SummaryCard from './SummaryCard.jsx'
import OperationBadges from './OperationBadges.jsx'
import DiffView from './DiffView.jsx'
import CodeBlock from './CodeBlock.jsx'
import RetrievalResults from './RetrievalResults.jsx'
import CandidatePicker from './CandidatePicker.jsx'
import AnswerCard from './AnswerCard.jsx'

export default function Message({ message, onSelectCandidate, onConfirmRun }) {
  if (message.role === 'user') {
    return (
      <div className="message-row message-row-user">
        <div className="user-bubble">{message.text}</div>
      </div>
    )
  }

  const status = message.metadata?.result?.status
  const needsSelection = status === 'needs_selection'
  const awaitingConfirmation = status === 'awaiting_confirmation'
  const answered = status === 'answered'

  const finalState =
    needsSelection || awaitingConfirmation
      ? 'pending'
      : message.metadata
        ? status === 'success' || status === 'rejected' || answered
          ? 'ok'
          : 'error'
        : message.error
          ? 'error'
          : null

  const isRetrieval = message.metadata?.strategy === 'HYBRID_RETRIEVAL'

  // Show the live preview (arrives right after APPLY/GENERATE) until the
  // final metadata (after TEST/COMMIT) lands, then switch to it -- both
  // carry the same operations/diff/new_file_content shape.
  const display = message.metadata || message.preview
  const isLivePreview = !message.metadata && !!message.preview

  return (
    <div className="message-row message-row-assistant">
      <div className="avatar avatar-assistant">E</div>
      <div className="assistant-content">
        <Timeline steps={message.steps} finalState={finalState} />
        {isLivePreview && <div className="live-preview-badge">● live preview — validating…</div>}
        {needsSelection ? (
          <CandidatePicker
            candidates={message.metadata.candidates}
            disabled={message.selectionPending}
            onSelect={(candidate) => onSelectCandidate(message.id, candidate, message.metadata)}
          />
        ) : isRetrieval ? (
          <RetrievalResults metadata={message.metadata} />
        ) : answered ? (
          <AnswerCard metadata={message.metadata} />
        ) : (
          <>
            {display?.operations?.length > 0 && <OperationBadges operations={display.operations} />}
            {display?.diff && <DiffView diff={display.diff} />}
            {display?.new_file_content && <CodeBlock code={display.new_file_content} label="Generated file" />}
            {awaitingConfirmation && (
              <div className="chat-confirm-bar">
                <span className="chat-confirm-bar-label">
                  {message.metadata.file
                    ? 'review the change in the code panel above, or accept/reject here'
                    : 'review before creating — no text preview available (binary file or plain folder)'}
                </span>
                <div className="chat-confirm-bar-actions">
                  {message.confirmError && <span className="chat-confirm-bar-error">{message.confirmError}</span>}
                  <button
                    className="confirm-btn confirm-btn-reject"
                    disabled={message.confirming}
                    onClick={() => onConfirmRun(message.metadata.run_id, message.id, false)}
                  >
                    Reject
                  </button>
                  <button
                    className="confirm-btn confirm-btn-accept"
                    disabled={message.confirming}
                    onClick={() => onConfirmRun(message.metadata.run_id, message.id, true)}
                  >
                    {message.confirming ? 'Applying…' : 'Accept'}
                  </button>
                </div>
              </div>
            )}
            {message.metadata && !awaitingConfirmation && <SummaryCard metadata={message.metadata} />}
          </>
        )}
        {message.error && !message.metadata && (
          <div className="summary-card summary-failed">
            <div className="summary-row">
              <div className="summary-key">error</div>
              <div className="summary-val">{message.error}</div>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}
