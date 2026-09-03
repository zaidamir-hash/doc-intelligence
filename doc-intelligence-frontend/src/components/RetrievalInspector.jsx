import { useEffect, useRef, useState } from "react"
import Icon from "./Icons"

const modeLabels = {
  dense: "Dense retrieval",
  lexical: "Lexical retrieval",
  hybrid_rrf: "Hybrid retrieval",
  hybrid_rrf_rerank: "Hybrid + reranking",
  hybrid_rrf_rerank_expansion: "Hybrid + reranking + expansion",
}

const decisionLabels = {
  selected_evidence: "Used for answer",
  unscored_fallback: "Unscored fallback",
  below_relevance_cutoff: "Below relevance cutoff",
  eligible_not_selected: "Eligible, not selected",
}

function valueOrDash(value) {
  return value ?? "—"
}

function pageLabel(item) {
  if (!item.page_start) return "Page unavailable"
  return item.page_end && item.page_end !== item.page_start
    ? `Pages ${item.page_start}–${item.page_end}`
    : `Page ${item.page_start}`
}

function RetrievalInspector({ response, onClose }) {
  const dialogRef = useRef(null)
  const [activeTab, setActiveTab] = useState("overview")
  const diagnostics = response?.diagnostics
  const expansion = diagnostics?.query_expansion
  const candidates = response?.retrieval_candidates || []
  const suppressed = diagnostics?.suppressed_evidence || []
  const config = response?.retrieval_configuration || {}

  useEffect(() => {
    const dialog = dialogRef.current
    if (!dialog) return
    if (response && !dialog.open) dialog.showModal()
    if (!response && dialog.open) dialog.close()
  }, [response])

  if (!response) return null

  return (
    <dialog
      ref={dialogRef}
      className="inspector-dialog"
      aria-labelledby="inspector-title"
      onCancel={(event) => { event.preventDefault(); onClose() }}
      onClose={onClose}
    >
      <header className="inspector-header">
        <div>
          <p className="eyebrow">Learning mode</p>
          <h2 id="inspector-title">Retrieval inspector</h2>
          <p>See how Lexis found and selected evidence for this answer.</p>
        </div>
        <button type="button" className="icon-button" aria-label="Close retrieval inspector" onClick={onClose}>
          <Icon name="close" />
        </button>
      </header>

      <div className="inspector-tabs" role="tablist" aria-label="Retrieval diagnostics">
        <button type="button" role="tab" aria-selected={activeTab === "overview"} onClick={() => setActiveTab("overview")}>Overview</button>
        <button type="button" role="tab" aria-selected={activeTab === "candidates"} onClick={() => setActiveTab("candidates")}>Candidates <span>{candidates.length}</span></button>
        <button type="button" role="tab" aria-selected={activeTab === "suppression"} onClick={() => setActiveTab("suppression")}>Suppression <span>{suppressed.length}</span></button>
      </div>

      <div className="inspector-content">
        {activeTab === "overview" && (
          <div role="tabpanel" className="inspector-overview">
            <section className="inspector-section">
              <h3>Active pipeline</h3>
              <div className="pipeline-mode">
                <span className="pipeline-mode__icon"><Icon name="search" /></span>
                <div>
                  <strong>{modeLabels[response.retrieval_mode] || response.retrieval_mode}</strong>
                  <span>The backend’s active retrieval configuration for this request.</span>
                </div>
              </div>
              <dl className="configuration-grid">
                <div><dt>Dense pool</dt><dd>{valueOrDash(config.dense_candidate_k)}</dd></div>
                <div><dt>Lexical pool</dt><dd>{valueOrDash(config.lexical_candidate_k)}</dd></div>
                <div><dt>Rerank pool</dt><dd>{valueOrDash(config.rerank_k)}</dd></div>
                <div><dt>Final evidence</dt><dd>{valueOrDash(config.final_evidence_k)}</dd></div>
                <div><dt>Relevance cutoff</dt><dd>{valueOrDash(config.relevance_cutoff)}</dd></div>
                <div><dt>RRF constant</dt><dd>{valueOrDash(config.rrf_k)}</dd></div>
              </dl>
            </section>

            <section className="inspector-section">
              <h3>Query expansion</h3>
              <dl className="query-expansion">
                <div><dt>Original question</dt><dd>{expansion?.original_query || response.question}</dd></div>
                <div>
                  <dt>Generated expansion</dt>
                  <dd>{expansion?.generated_query || "No expansion was generated."}</dd>
                </div>
                <div>
                  <dt>Outcome</dt>
                  <dd>{expansion?.used_expansion ? "The expansion was used alongside the original question." : expansion?.fallback_reason || "The original question was used without expansion."}</dd>
                </div>
              </dl>
            </section>

            {(diagnostics?.reranker_fallback_error || diagnostics?.generation_fallback_error) && (
              <section className="inspector-section inspector-section--warning">
                <h3>Fallbacks</h3>
                {diagnostics.reranker_fallback_error && <p>{diagnostics.reranker_fallback_error}</p>}
                {diagnostics.generation_fallback_error && <p>{diagnostics.generation_fallback_error}</p>}
              </section>
            )}
          </div>
        )}

        {activeTab === "candidates" && (
          <div role="tabpanel">
            <p className="inspector-explanation">Ranks show where each passage appeared at each retrieval stage. A dash means that retrieval method did not return the passage.</p>
            {candidates.length === 0 ? <InspectorEmpty text="No retrieval candidates were returned." /> : (
              <div className="diagnostics-table-wrap">
                <table className="diagnostics-table">
                  <thead><tr><th scope="col">Passage</th><th scope="col">Dense</th><th scope="col">Lexical</th><th scope="col">Fused</th><th scope="col">Reranked</th><th scope="col">Score</th><th scope="col">Decision</th></tr></thead>
                  <tbody>
                    {candidates.map((candidate) => (
                      <tr key={candidate.chunk_id} className={candidate.selected_for_generation ? "is-selected" : ""}>
                        <td><strong>#{candidate.chunk_index}</strong><span>{pageLabel(candidate)}</span>{candidate.section_title && <span title={candidate.section_title}>{candidate.section_title}</span>}</td>
                        <td>{valueOrDash(candidate.dense_rank)}</td>
                        <td>{valueOrDash(candidate.lexical_rank)}</td>
                        <td>{valueOrDash(candidate.fused_rank)}</td>
                        <td>{valueOrDash(candidate.reranked_rank)}</td>
                        <td>{candidate.reranker_score ?? "Fallback"}</td>
                        <td><span className={`decision decision--${candidate.relevance_decision}`}>{decisionLabels[candidate.relevance_decision] || candidate.relevance_decision}</span></td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </div>
        )}

        {activeTab === "suppression" && (
          <div role="tabpanel">
            <p className="inspector-explanation">Suppression prevents overlapping or duplicate passages from crowding the final answer context.</p>
            {suppressed.length === 0 ? <InspectorEmpty text="No evidence passages were suppressed." /> : (
              <div className="suppression-list">
                {suppressed.map((item) => (
                  <article key={item.chunk_id}>
                    <div><strong>Chunk {item.chunk_id}</strong><span>{item.reason}</span></div>
                    {item.duplicate_of_chunk_id && <p>Duplicates chunk {item.duplicate_of_chunk_id}</p>}
                    {item.text_similarity != null && <p>Text similarity: {item.text_similarity}</p>}
                  </article>
                ))}
              </div>
            )}
          </div>
        )}
      </div>
    </dialog>
  )
}

function InspectorEmpty({ text }) {
  return <div className="inspector-empty"><Icon name="info" /><p>{text}</p></div>
}

export default RetrievalInspector
