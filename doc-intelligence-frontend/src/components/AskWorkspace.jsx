import { useEffect, useMemo, useRef, useState } from "react"
import { apiErrorDetails, askDocument } from "../api"
import Icon from "./Icons"
import { EmptyState, Spinner, StatusBadge } from "./Primitives"
import RetrievalInspector from "./RetrievalInspector"

function pageLabel(item) {
  if (!item.page_start) return "Page unavailable"
  return item.page_end && item.page_end !== item.page_start
    ? `Pages ${item.page_start}–${item.page_end}`
    : `Page ${item.page_start}`
}

function uniqueCitations(citations = []) {
  return Array.from(new Map(citations.map((citation) => [citation.source_id, citation])).values())
}

function AskWorkspace({
  apiKey,
  documents,
  selectedDocumentId,
  onSelectDocument,
  onOpenLibrary,
  conversations,
  setConversations,
}) {
  const [question, setQuestion] = useState("")
  const [debugEnabled, setDebugEnabled] = useState(false)
  const [focusedTurnId, setFocusedTurnId] = useState(null)
  const [focusedSourceId, setFocusedSourceId] = useState(null)
  const [sourcesOpen, setSourcesOpen] = useState(false)
  const [inspectorResponse, setInspectorResponse] = useState(null)
  const composerRef = useRef(null)
  const conversationRef = useRef(null)
  const readyDocuments = documents.filter((document) => document.status === "ready")
  const selectedDocument = documents.find((document) => document.document_id === selectedDocumentId)
  const turns = conversations[selectedDocumentId] || []
  const pending = turns.some((turn) => turn.state === "pending")
  const latestAnsweredTurn = [...turns].reverse().find((turn) => turn.response)
  const focusedTurn = turns.find((turn) => turn.id === focusedTurnId && turn.response) || latestAnsweredTurn
  const citations = uniqueCitations(focusedTurn?.response?.citations)
  const canSubmit = Boolean(selectedDocument?.status === "ready" && question.trim() && !pending)

  useEffect(() => {
    const conversation = conversationRef.current
    const latestTurn = conversation?.querySelector(".research-turn:last-child")
    latestTurn?.scrollIntoView({ block: "start" })
  }, [selectedDocumentId, turns.length])

  const composerHint = useMemo(() => {
    if (readyDocuments.length === 0) return "Upload and index a document before asking a question."
    if (!selectedDocumentId) return "Choose a ready document to enable questions."
    if (selectedDocument?.status !== "ready") return "This document is not ready to query."
    if (pending) return "Wait for the current answer before asking another question."
    return "Enter to ask · Shift + Enter for a new line"
  }, [pending, readyDocuments.length, selectedDocument, selectedDocumentId])

  const focusSource = (turnId, sourceId) => {
    setFocusedTurnId(turnId)
    setFocusedSourceId(sourceId)
    setSourcesOpen(true)
  }

  const handleSubmit = async () => {
    const cleanedQuestion = question.trim()
    if (!canSubmit || !cleanedQuestion) return
    const turnId = window.crypto?.randomUUID?.() || `${Date.now()}-${Math.random()}`
    const documentId = selectedDocumentId
    const turn = { id: turnId, question: cleanedQuestion, state: "pending", debugRequested: debugEnabled }
    setQuestion("")
    setConversations((previous) => ({
      ...previous,
      [documentId]: [...(previous[documentId] || []), turn],
    }))

    try {
      const response = await askDocument(apiKey, documentId, cleanedQuestion, debugEnabled)
      setConversations((previous) => ({
        ...previous,
        [documentId]: (previous[documentId] || []).map((item) => item.id === turnId
          ? { ...item, state: "complete", response }
          : item),
      }))
      setFocusedTurnId(turnId)
      setFocusedSourceId(response.citations?.[0]?.source_id || null)
    } catch (error) {
      const details = apiErrorDetails(error, "Lexis could not answer this question. Try again.")
      setConversations((previous) => ({
        ...previous,
        [documentId]: (previous[documentId] || []).map((item) => item.id === turnId
          ? { ...item, state: "error", error: details.message }
          : item),
      }))
    } finally {
      window.requestAnimationFrame(() => composerRef.current?.focus())
    }
  }

  return (
    <main className="ask-page" id="main-content">
      <header className="workspace-header">
        <div className="workspace-header__title">
          <p className="eyebrow">Research workspace</p>
          <h1>Ask your document</h1>
        </div>
        <div className="workspace-controls">
          <label className="document-select">
            <span>Active document</span>
            <div>
              <Icon name="document" size={18} />
              <select value={selectedDocumentId} onChange={(event) => { onSelectDocument(event.target.value); setFocusedTurnId(null); setFocusedSourceId(null); setSourcesOpen(false) }}>
                <option value="">Choose a ready document</option>
                {readyDocuments.map((document) => <option key={document.document_id} value={document.document_id}>{document.filename}</option>)}
              </select>
              <Icon name="chevronDown" size={16} />
            </div>
          </label>
          <label className="learning-toggle">
            <input type="checkbox" checked={debugEnabled} onChange={(event) => setDebugEnabled(event.target.checked)} />
            <span className="learning-toggle__control" aria-hidden="true" />
            <span className="learning-toggle__copy"><strong>Learning mode</strong><small>Include retrieval diagnostics</small></span>
            <span className="learning-toggle__mobile-label">Debug</span>
          </label>
        </div>
      </header>

      {selectedDocument && (
        <div className="active-document-bar">
          <Icon name="document" size={18} />
          <strong title={selectedDocument.filename}>{selectedDocument.filename}</strong>
          <StatusBadge status={selectedDocument.status} />
          <span>{selectedDocument.pages ?? "—"} pages</span>
          <span>{selectedDocument.chunks_stored ?? "—"} passages</span>
        </div>
      )}

      <div className={`workspace-layout${sourcesOpen ? " sources-open" : ""}`}>
        <section ref={conversationRef} className="conversation" aria-label="Questions and answers">
          {readyDocuments.length === 0 ? (
            <EmptyState
              icon="document"
              title="No document is ready yet"
              description="Upload a text-based PDF and wait for indexing to finish before asking questions."
              action={<button type="button" className="button button--primary" onClick={onOpenLibrary}>Open document library<Icon name="arrowRight" /></button>}
            />
          ) : !selectedDocumentId ? (
            <EmptyState
              icon="search"
              title="Choose a document to begin"
              description="Select one ready PDF above. Each answer will stay connected to the pages and passages that support it."
            />
          ) : turns.length === 0 ? (
            <WorkspaceWelcome document={selectedDocument} onPrompt={(prompt) => { setQuestion(prompt); composerRef.current?.focus() }} />
          ) : (
            <div className="turn-list" aria-live="polite">
              {turns.map((turn) => (
                <ResearchTurn
                  key={turn.id}
                  turn={turn}
                  document={selectedDocument}
                  onSource={(sourceId) => focusSource(turn.id, sourceId)}
                  onShowSources={() => { setFocusedTurnId(turn.id); setSourcesOpen(true) }}
                  onInspect={() => setInspectorResponse(turn.response)}
                />
              ))}
            </div>
          )}

          <div className="composer-wrap">
            <div className="composer">
              <label htmlFor="question" className="sr-only">Question about the active document</label>
              <textarea
                ref={composerRef}
                id="question"
                rows="2"
                maxLength="2000"
                value={question}
                disabled={!selectedDocumentId || selectedDocument?.status !== "ready" || pending}
                placeholder={selectedDocumentId ? "Ask a specific question about this document…" : "Choose a document to start asking questions"}
                onChange={(event) => setQuestion(event.target.value)}
                onKeyDown={(event) => {
                  if (event.key === "Enter" && !event.shiftKey && !event.nativeEvent.isComposing) {
                    event.preventDefault()
                    void handleSubmit()
                  }
                }}
              />
              <button type="button" className="composer__send" aria-label="Ask question" onClick={() => void handleSubmit()} disabled={!canSubmit}>
                {pending ? <Spinner size={18} /> : <Icon name="send" size={19} />}
              </button>
            </div>
            <div className="composer-meta">
              <span>{composerHint}</span>
              {question.length > 1800 && <span>{question.length}/2000</span>}
            </div>
          </div>
        </section>

        <SourcesPanel
          turn={focusedTurn}
          citations={citations}
          focusedSourceId={focusedSourceId}
          open={sourcesOpen}
          onSelectSource={setFocusedSourceId}
          onClose={() => setSourcesOpen(false)}
        />
      </div>

      <RetrievalInspector response={inspectorResponse} onClose={() => setInspectorResponse(null)} />
    </main>
  )
}

function WorkspaceWelcome({ document, onPrompt }) {
  const prompts = [
    "What are the main conclusions?",
    "Summarize the key points and supporting evidence.",
    "What limitations or risks does the document mention?",
  ]
  return (
    <div className="workspace-welcome">
      <span className="workspace-welcome__icon"><Icon name="bookOpen" size={28} /></span>
      <p className="eyebrow">Document ready</p>
      <h2 title={document?.filename}>What would you like to understand?</h2>
      <p>Ask about details, compare ideas, or request a grounded summary. Lexis will only answer from evidence it can cite.</p>
      <div className="prompt-list">
        {prompts.map((prompt) => <button type="button" key={prompt} onClick={() => onPrompt(prompt)}>{prompt}<Icon name="arrowRight" size={17} /></button>)}
      </div>
    </div>
  )
}

function ResearchTurn({ turn, document, onSource, onShowSources, onInspect }) {
  if (turn.state === "pending") return <PendingTurn turn={turn} />
  if (turn.state === "error") {
    return (
      <article className="research-turn">
        <QuestionBlock question={turn.question} document={document} />
        <div className="answer-state answer-state--error" role="alert">
          <Icon name="warning" /><div><strong>Lexis couldn’t complete this request</strong><p>{turn.error}</p></div>
        </div>
      </article>
    )
  }

  const response = turn.response
  const refused = response.status === "insufficient_evidence" || (!response.answer && Boolean(response.refusal_reason))
  const partial = response.status === "partially_answered"
  const answer = response.answer || response.refusal_reason || "Lexis could not produce a grounded answer."
  const citations = uniqueCitations(response.citations)

  return (
    <article className="research-turn">
      <QuestionBlock question={turn.question} document={document} />
      <section className={refused ? "answer-block answer-block--refused" : partial ? "answer-block answer-block--partial" : "answer-block"} aria-label={refused ? "Grounded refusal" : partial ? "Grounded answer with limitations" : "Grounded answer"}>
        <header className="answer-block__header">
          <span className="answer-block__mark"><Icon name={refused || partial ? "info" : "bookOpen"} size={18} /></span>
          <div><strong>{refused ? "Grounded refusal" : partial ? "Grounded answer with limitations" : "Grounded answer"}</strong><span>{refused ? "The available evidence did not support an answer" : partial ? `Some parts could not be supported · ${citations.length} ${citations.length === 1 ? "source" : "sources"}` : `Based on ${citations.length} ${citations.length === 1 ? "source" : "sources"}`}</span></div>
        </header>
        <AnswerText text={answer} citations={citations} onCitation={onSource} />
        <footer className="answer-block__footer">
          <button type="button" className="text-button" onClick={onShowSources} disabled={citations.length === 0}>
            <Icon name="bookOpen" size={17} />{citations.length === 0 ? "No cited sources" : `View ${citations.length} ${citations.length === 1 ? "source" : "sources"}`}
          </button>
          {turn.debugRequested && response.retrieval_candidates && (
            <button type="button" className="text-button" onClick={onInspect}><Icon name="gear" size={17} />Inspect retrieval</button>
          )}
        </footer>
      </section>
    </article>
  )
}

function QuestionBlock({ question, document }) {
  return (
    <div className="question-block">
      <div><span>You asked</span><span className="question-block__document" title={document?.filename}>{document?.filename}</span></div>
      <p>{question}</p>
    </div>
  )
}

function PendingTurn({ turn }) {
  return (
    <article className="research-turn research-turn--pending">
      <div className="question-block"><div><span>You asked</span></div><p>{turn.question}</p></div>
      <div className="answer-loading" role="status">
        <span className="answer-loading__pulse"><Icon name="search" /></span>
        <div>
          <strong>Building a grounded answer</strong>
          <div className="answer-loading__stages">
            <span>Retrieving relevant passages</span>
            <span>Reranking candidate evidence</span>
            <span>Grounding the response</span>
          </div>
        </div>
      </div>
    </article>
  )
}

function AnswerText({ text, citations, onCitation }) {
  const sourceIds = citations.map((citation) => citation.source_id).filter(Boolean)
  const escaped = sourceIds.map((id) => id.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"))
  const markerPattern = escaped.length ? new RegExp(`(\\[(?:${escaped.join("|")})\\])`, "g") : null
  const lines = String(text).split(/\r?\n/)

  const renderInline = (line, lineIndex) => {
    if (!markerPattern) return line
    return line.split(markerPattern).map((part, index) => {
      const sourceId = part.startsWith("[") && part.endsWith("]") ? part.slice(1, -1) : null
      if (!sourceIds.includes(sourceId)) return part
      return <button type="button" className="citation-marker" key={`${lineIndex}-${index}`} onClick={() => onCitation(sourceId)} aria-label={`View source ${sourceId}`}>{sourceId}</button>
    })
  }

  const content = []
  let bullets = []
  const flushBullets = () => {
    if (bullets.length) {
      content.push(<ul key={`list-${content.length}`}>{bullets.map((bullet) => <li key={bullet.index}>{renderInline(bullet.text, bullet.index)}</li>)}</ul>)
      bullets = []
    }
  }
  lines.forEach((line, index) => {
    const bullet = line.match(/^\s*[-*]\s+(.+)/)
    if (bullet) {
      bullets.push({ text: bullet[1], index })
    } else {
      flushBullets()
      if (line.trim()) content.push(<p key={`line-${index}`}>{renderInline(line, index)}</p>)
    }
  })
  flushBullets()
  return <div className="answer-content">{content}</div>
}

function SourcesPanel({ turn, citations, focusedSourceId, open, onSelectSource, onClose }) {
  const selected = citations.find((citation) => citation.source_id === focusedSourceId) || citations[0]
  return (
    <aside className={`sources-panel${open ? " is-open" : ""}`} aria-label="Answer sources">
      <header className="sources-panel__header">
        <div><p className="eyebrow">Evidence</p><h2>Answer sources</h2></div>
        <button type="button" className="icon-button sources-panel__close" aria-label="Close sources" onClick={onClose}><Icon name="close" /></button>
      </header>
      {!turn ? (
        <div className="sources-panel__empty"><Icon name="bookOpen" /><p>Sources for the latest answer will appear here.</p></div>
      ) : citations.length === 0 ? (
        <div className="sources-panel__empty"><Icon name="info" /><p>This response has no cited sources. Grounded refusals intentionally contain no unsupported evidence.</p></div>
      ) : (
        <>
          <div className="source-tabs" aria-label="Sources used in this answer">
            {citations.map((citation) => (
              <button type="button" key={citation.source_id} className={selected?.source_id === citation.source_id ? "is-active" : ""} onClick={() => onSelectSource(citation.source_id)} aria-pressed={selected?.source_id === citation.source_id}>
                {citation.source_id}<span>{pageLabel(citation)}</span>
              </button>
            ))}
          </div>
          {selected && <EvidenceCard citation={selected} />}
        </>
      )}
    </aside>
  )
}

function EvidenceCard({ citation }) {
  return (
    <article className="evidence-card">
      <div className="evidence-card__identity"><span>{citation.source_id}</span><strong title={citation.filename}>{citation.filename}</strong></div>
      <dl className="evidence-card__meta">
        <div><dt>Location</dt><dd>{pageLabel(citation)}</dd></div>
        {citation.section_title && <div><dt>Section</dt><dd>{citation.section_title}</dd></div>}
      </dl>
      <div className="evidence-card__passage"><span>Evidence preview</span><blockquote>{citation.preview}</blockquote></div>
      {citation.supporting_quotes?.length > 0 && (
        <details className="evidence-details">
          <summary>Supporting excerpts ({citation.supporting_quotes.length})</summary>
          {citation.supporting_quotes.map((quote, index) => <blockquote key={`${citation.source_id}-quote-${index}`}>{quote}</blockquote>)}
        </details>
      )}
      {citation.supported_claims?.length > 0 && (
        <details className="evidence-details">
          <summary>Supported claims ({citation.supported_claims.length})</summary>
          <ul>{citation.supported_claims.map((claim, index) => <li key={`${citation.source_id}-claim-${index}`}>{claim}</li>)}</ul>
        </details>
      )}
      <p className="evidence-card__note"><Icon name="info" size={15} />Lexis records the page location but the current backend does not expose the PDF for direct page navigation.</p>
    </article>
  )
}

export default AskWorkspace
