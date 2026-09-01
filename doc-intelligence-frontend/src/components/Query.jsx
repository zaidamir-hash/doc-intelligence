import { useEffect, useRef, useState } from "react"
import { apiErrorMessage, askDocument } from "../api"

function pageLabel(item) {
  if (!item.page_start) return "Page unavailable"
  return item.page_end && item.page_end !== item.page_start
    ? `Pages ${item.page_start}–${item.page_end}`
    : `Page ${item.page_start}`
}

function QueryPanel({
  apiKey,
  authenticated,
  uploadedDocs,
  selectedDocumentId,
  setSelectedDocumentId,
  setQueryCount,
  configuration,
}) {
  const [question, setQuestion] = useState("")
  const [messages, setMessages] = useState([])
  const [loading, setLoading] = useState(false)
  const [debugEnabled, setDebugEnabled] = useState(false)
  const messagesEndRef = useRef(null)
  const readyDocuments = uploadedDocs.filter((document) => document.status === "ready")

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" })
  }, [messages])

  const handleQuery = async () => {
    const cleanedQuestion = question.trim()
    if (!cleanedQuestion) return
    if (!authenticated) {
      setMessages((previous) => [...previous, { type: "error", content: "Connect with a validated API key first." }])
      return
    }
    if (!selectedDocumentId) {
      setMessages((previous) => [...previous, { type: "error", content: "Select an indexed document first." }])
      return
    }

    setQuestion("")
    setMessages((previous) => [...previous, { type: "user", content: cleanedQuestion }])
    setLoading(true)
    try {
      const response = await askDocument(
        apiKey,
        selectedDocumentId,
        cleanedQuestion,
        debugEnabled,
      )
      setMessages((previous) => [...previous, { type: "answer", response }])
      setQueryCount((count) => count + 1)
    } catch (error) {
      setMessages((previous) => [...previous, {
        type: "error",
        content: apiErrorMessage(error, "Query failed. Please try again."),
      }])
    } finally {
      setLoading(false)
    }
  }

  return (
    <div style={{ flex: 1, display: "flex", flexDirection: "column", height: "100vh", backgroundColor: "#0A0F1E" }}>
      <header style={{ padding: "20px 28px", borderBottom: "1px solid #1E2D4A", display: "flex", justifyContent: "space-between", alignItems: "center", gap: "18px" }}>
        <div>
          <div style={{ fontSize: "18px", fontWeight: "600", color: "#F9FAFB" }}>Ask your document</div>
          <div style={{ fontSize: "12px", color: "#9CA3AF", marginTop: "4px" }}>
            Answers remain traceable to selected page-aware evidence.
          </div>
        </div>
        <div style={{ display: "flex", alignItems: "center", gap: "12px" }}>
          <select
            aria-label="Document to query"
            value={selectedDocumentId}
            onChange={(event) => setSelectedDocumentId(event.target.value)}
            style={controlStyle}
          >
            <option value="">Select an indexed document...</option>
            {readyDocuments.map((document) => (
              <option key={document.document_id} value={document.document_id}>
                {document.filename}
              </option>
            ))}
          </select>
          <label style={{ color: "#9CA3AF", fontSize: "12px", display: "flex", gap: "7px", alignItems: "center" }}>
            <input type="checkbox" checked={debugEnabled} onChange={(event) => setDebugEnabled(event.target.checked)} />
            Learning debug
          </label>
        </div>
      </header>

      {debugEnabled && configuration?.retrieval && (
        <div style={{ padding: "9px 28px", borderBottom: "1px solid #1E2D4A", color: "#93C5FD", fontSize: "11px", fontFamily: "JetBrains Mono, monospace" }}>
          Active: {configuration.retrieval.active_mode} · dense {configuration.retrieval.dense_candidate_k} · lexical {configuration.retrieval.lexical_candidate_k} · rerank {configuration.retrieval.rerank_k} · evidence {configuration.retrieval.final_evidence_k} · cutoff ≥ {configuration.retrieval.relevance_cutoff}
        </div>
      )}

      <main style={{ flex: 1, overflowY: "auto", padding: "24px 32px", display: "flex", flexDirection: "column", gap: "20px" }}>
        {messages.length === 0 && (
          <div style={{ margin: "auto", textAlign: "center", color: "#6B7280" }}>
            <div style={{ fontSize: "44px" }}>🔍</div>
            <p>Select a document and ask a question.</p>
            <small>Enable Learning debug when you want to inspect retrieval—not for normal use.</small>
          </div>
        )}

        {messages.map((message, index) => (
          <div key={`${message.type}-${index}`}>
            {message.type === "user" && <UserMessage content={message.content} />}
            {message.type === "error" && <ErrorMessage content={message.content} />}
            {message.type === "answer" && <AnswerMessage response={message.response} />}
          </div>
        ))}
        {loading && <div style={{ color: "#93C5FD", fontSize: "13px" }}>Retrieving, reranking, and grounding the answer...</div>}
        <div ref={messagesEndRef} />
      </main>

      <footer style={{ padding: "18px 28px", borderTop: "1px solid #1E2D4A" }}>
        <div style={{ display: "flex", gap: "10px", alignItems: "flex-end" }}>
          <textarea
            aria-label="Question"
            value={question}
            onChange={(event) => setQuestion(event.target.value)}
            onKeyDown={(event) => {
              if (event.key === "Enter" && !event.shiftKey) {
                event.preventDefault()
                void handleQuery()
              }
            }}
            placeholder="Ask a question about the selected document..."
            rows={2}
            maxLength={2000}
            style={{ ...controlStyle, flex: 1, resize: "none" }}
          />
          <button type="button" onClick={() => void handleQuery()} disabled={loading || !question.trim()} style={{
            padding: "13px 20px",
            border: "none",
            borderRadius: "8px",
            backgroundColor: loading || !question.trim() ? "#1E2D4A" : "#3B82F6",
            color: loading || !question.trim() ? "#6B7280" : "white",
            cursor: loading ? "wait" : "pointer",
            fontWeight: "600",
          }}>
            {loading ? "Working..." : "Ask →"}
          </button>
        </div>
      </footer>
    </div>
  )
}

function UserMessage({ content }) {
  return <div style={{ display: "flex", justifyContent: "flex-end" }}><div style={{ ...messageCard, backgroundColor: "#2563EB" }}>{content}</div></div>
}

function ErrorMessage({ content }) {
  return <div role="alert" style={{ ...messageCard, color: "#F87171", borderColor: "rgba(239,68,68,0.3)" }}>{content}</div>
}

function AnswerMessage({ response }) {
  const answer = response.answer || response.refusal_reason || "Lexis could not produce a grounded answer."
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "12px", maxWidth: "950px" }}>
      <div style={messageCard}>
        {answer}
        {response.status === "refused" && <div style={{ color: "#F59E0B", fontSize: "11px", marginTop: "8px" }}>Grounded refusal</div>}
      </div>

      {response.citations?.length > 0 && (
        <section>
          <SectionLabel text="Citations" />
          {response.citations.map((citation) => (
            <div key={citation.source_id} style={evidenceCard}>
              <strong style={{ color: "#60A5FA" }}>[{citation.source_id}] {citation.filename}</strong>
              <span style={{ color: "#9CA3AF" }}> · {pageLabel(citation)}</span>
              {citation.section_title && <span style={{ color: "#9CA3AF" }}> · {citation.section_title}</span>}
              <div style={{ color: "#D1D5DB", marginTop: "6px" }}>{citation.preview}</div>
            </div>
          ))}
        </section>
      )}

      {response.evidence?.length > 0 && (
        <details style={detailsStyle}>
          <summary style={{ cursor: "pointer", color: "#93C5FD" }}>Selected evidence sent to the answer model ({response.evidence.length})</summary>
          {response.evidence.map((item) => (
            <div key={item.source_id} style={{ ...evidenceCard, marginTop: "8px" }}>
              <strong>{item.source_id}</strong> · {pageLabel(item)} · reranker {item.reranker_score ?? "fallback"}/3
              <div style={{ color: "#9CA3AF", marginTop: "5px" }}>{item.preview}</div>
            </div>
          ))}
        </details>
      )}

      {response.retrieval_candidates && <DebugPanel response={response} />}
    </div>
  )
}

function DebugPanel({ response }) {
  const expansion = response.diagnostics?.query_expansion
  return (
    <details style={detailsStyle} open>
      <summary style={{ cursor: "pointer", color: "#C4B5FD", fontWeight: "600" }}>
        Retrieval path · {response.retrieval_mode}
      </summary>
      {expansion && (
        <div style={{ color: "#9CA3AF", fontSize: "12px", margin: "10px 0" }}>
          Original: “{expansion.original_query}”<br />
          Expansion: {expansion.used_expansion ? `“${expansion.generated_query}”` : `not used (${expansion.fallback_reason || "disabled"})`}
        </div>
      )}
      <div style={{ overflowX: "auto" }}>
        <table style={{ width: "100%", borderCollapse: "collapse", fontSize: "11px" }}>
          <thead><tr>{["Chunk", "Dense", "Lexical", "Fused", "Reranked", "Score", "Decision"].map((heading) => <th key={heading} style={debugCell}>{heading}</th>)}</tr></thead>
          <tbody>
            {response.retrieval_candidates.map((candidate) => (
              <tr key={candidate.chunk_id}>
                <td style={debugCell}>#{candidate.chunk_index} · {pageLabel(candidate)}</td>
                <td style={debugCell}>{candidate.dense_rank ?? "—"}</td>
                <td style={debugCell}>{candidate.lexical_rank ?? "—"}</td>
                <td style={debugCell}>{candidate.fused_rank}</td>
                <td style={debugCell}>{candidate.reranked_rank}</td>
                <td style={debugCell}>{candidate.reranker_score ?? "fallback"}</td>
                <td style={debugCell}>{candidate.relevance_decision}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </details>
  )
}

function SectionLabel({ text }) {
  return <div style={{ color: "#6B7280", fontSize: "10px", fontWeight: "700", textTransform: "uppercase", letterSpacing: "1px", marginBottom: "6px" }}>{text}</div>
}

const controlStyle = { padding: "10px 12px", backgroundColor: "#111827", border: "1px solid #1E2D4A", borderRadius: "8px", color: "#F9FAFB", fontFamily: "Inter, sans-serif" }
const messageCard = { maxWidth: "850px", padding: "14px 18px", backgroundColor: "#111827", border: "1px solid #1E2D4A", borderRadius: "10px", color: "#F9FAFB", lineHeight: "1.65", fontSize: "14px" }
const evidenceCard = { padding: "9px 12px", marginBottom: "6px", backgroundColor: "rgba(59,130,246,0.05)", border: "1px solid rgba(59,130,246,0.15)", borderRadius: "6px", fontSize: "11px" }
const detailsStyle = { padding: "12px", backgroundColor: "#111827", border: "1px solid #1E2D4A", borderRadius: "8px", color: "#D1D5DB" }
const debugCell = { padding: "7px", borderBottom: "1px solid #1E2D4A", textAlign: "left", color: "#9CA3AF", whiteSpace: "nowrap" }

export default QueryPanel
