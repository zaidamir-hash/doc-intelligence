import { useState, useRef, useEffect } from "react"
import axios from "axios"

const API_URL = "http://127.0.0.1:8000"

function QueryPanel({ apiKey, uploadedDocs }) {
  const [question, setQuestion] = useState("")
  const [selectedDoc, setSelectedDoc] = useState("")
  const [messages, setMessages] = useState([])
  const [loading, setLoading] = useState(false)
  const messagesEndRef = useRef(null)

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" })
  }, [messages])

  const handleQuery = async () => {
    if (!question.trim()) return
    if (!apiKey.trim()) {
      setMessages(prev => [...prev, {
        type: "error",
        content: "Please enter your API key in the sidebar first"
      }])
      return
    }
    if (!selectedDoc) {
      setMessages(prev => [...prev, {
        type: "error",
        content: "Please select a document to query first"
      }])
      return
    }

    const userQuestion = question
    setQuestion("")
    setMessages(prev => [...prev, { type: "user", content: userQuestion }])
    setLoading(true)

    try {
      const response = await axios.post(`${API_URL}/query`, {
        question: userQuestion,
        filename: selectedDoc
      }, {
        headers: { "X-API-Key": apiKey }
      })

      setMessages(prev => [...prev, {
        type: "answer",
        content: response.data.answer,
        sources: response.data.sources
      }])
    } catch (err) {
      setMessages(prev => [...prev, {
        type: "error",
        content: err.response?.data?.detail || "Query failed. Please try again."
      }])
    } finally {
      setLoading(false)
    }
  }

  const handleKeyDown = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault()
      handleQuery()
    }
  }

  return (
    <div style={{
      flex: 1,
      display: "flex",
      flexDirection: "column",
      height: "100vh",
      backgroundColor: "#0A0F1E"
    }}>
      {/* Header */}
      <div style={{
        padding: "24px 32px",
        borderBottom: "1px solid #1E2D4A",
        display: "flex",
        alignItems: "center",
        justifyContent: "space-between",
        gap: "12px"
      }}>
        <div>
          <div style={{
            fontSize: "18px",
            fontWeight: "600",
            color: "#F9FAFB"
          }}>
            Ask your documents
          </div>
          <div style={{
            fontSize: "13px",
            color: "#9CA3AF",
            marginTop: "2px"
          }}>
            Upload a PDF and start querying
          </div>
        </div>

        <select
          value={selectedDoc}
          onChange={(e) => setSelectedDoc(e.target.value)}
          style={{
            padding: "10px 14px",
            backgroundColor: "#111827",
            border: "1px solid #1E2D4A",
            borderRadius: "8px",
            color: "#F9FAFB",
            fontSize: "13px",
            fontFamily: "Inter, sans-serif",
            outline: "none",
            minWidth: "220px"
          }}
        >
          <option value="">Select a document...</option>
          {uploadedDocs.map((doc, i) => (
            <option key={i} value={doc.filename}>{doc.filename}</option>
          ))}
        </select>
      </div>

      {/* Messages Area */}
      <div style={{
        flex: 1,
        overflowY: "auto",
        padding: "24px 32px",
        display: "flex",
        flexDirection: "column",
        gap: "24px"
      }}>
        {messages.length === 0 && (
          <div style={{
            flex: 1,
            display: "flex",
            flexDirection: "column",
            alignItems: "center",
            justifyContent: "center",
            gap: "16px",
            color: "#4B5563",
            marginTop: "80px"
          }}>
            <div style={{ fontSize: "48px" }}>🔍</div>
            <div style={{
              fontSize: "16px",
              fontWeight: "500",
              color: "#6B7280"
            }}>
              No questions yet
            </div>
            <div style={{ fontSize: "13px", color: "#4B5563", textAlign: "center" }}>
              Select a document above and ask anything about it
            </div>
          </div>
        )}

        {messages.map((msg, index) => (
          <div key={index}>
            {msg.type === "user" && (
              <div style={{
                display: "flex",
                justifyContent: "flex-end"
              }}>
                <div style={{
                  maxWidth: "70%",
                  padding: "12px 16px",
                  backgroundColor: "#3B82F6",
                  borderRadius: "12px 12px 2px 12px",
                  fontSize: "14px",
                  color: "#F9FAFB",
                  lineHeight: "1.5"
                }}>
                  {msg.content}
                </div>
              </div>
            )}

            {msg.type === "answer" && (
              <div style={{ display: "flex", flexDirection: "column", gap: "12px" }}>
                <div style={{
                  maxWidth: "80%",
                  padding: "16px 20px",
                  backgroundColor: "#111827",
                  borderRadius: "2px 12px 12px 12px",
                  border: "1px solid #1E2D4A",
                  fontSize: "14px",
                  color: "#F9FAFB",
                  lineHeight: "1.7"
                }}>
                  {msg.content}
                </div>

                {msg.sources && msg.sources.length > 0 && (
                  <div style={{
                    display: "flex",
                    flexDirection: "column",
                    gap: "6px",
                    maxWidth: "80%"
                  }}>
                    <div style={{
                      fontSize: "11px",
                      fontWeight: "600",
                      color: "#4B5563",
                      textTransform: "uppercase",
                      letterSpacing: "0.8px"
                    }}>
                      Sources
                    </div>
                    {msg.sources.map((source, i) => (
                      <div key={i} style={{
                        padding: "8px 12px",
                        backgroundColor: "rgba(59,130,246,0.05)",
                        border: "1px solid rgba(59,130,246,0.15)",
                        borderRadius: "6px",
                        fontSize: "11px",
                        fontFamily: "JetBrains Mono, monospace"
                      }}>
                        <span style={{ color: "#3B82F6", fontWeight: "600" }}>
                          {source.filename}
                        </span>
                        <span style={{ color: "#4B5563" }}> · chunk {source.chunk_index}</span>
                        <div style={{
                          color: "#6B7280",
                          marginTop: "4px",
                          lineHeight: "1.5"
                        }}>
                          {source.preview}...
                        </div>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            )}

            {msg.type === "error" && (
              <div style={{
                padding: "10px 14px",
                backgroundColor: "rgba(239,68,68,0.1)",
                border: "1px solid rgba(239,68,68,0.2)",
                borderRadius: "8px",
                fontSize: "13px",
                color: "#F87171"
              }}>
                {msg.content}
              </div>
            )}
          </div>
        ))}

        {loading && (
          <div style={{ display: "flex", gap: "6px", alignItems: "center" }}>
            {[0, 1, 2].map(i => (
              <div key={i} style={{
                width: "6px",
                height: "6px",
                backgroundColor: "#3B82F6",
                borderRadius: "50%",
                animation: `bounce 1.2s ease-in-out ${i * 0.2}s infinite`
              }} />
            ))}
          </div>
        )}

        <div ref={messagesEndRef} />
      </div>

      {/* Input Area */}
      <div style={{
        padding: "20px 32px",
        borderTop: "1px solid #1E2D4A",
        backgroundColor: "#0A0F1E"
      }}>
        <div style={{
          display: "flex",
          gap: "12px",
          alignItems: "flex-end"
        }}>
          <textarea
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Ask anything about your documents... (Enter to send)"
            rows={1}
            style={{
              flex: 1,
              padding: "14px 16px",
              backgroundColor: "#111827",
              border: "1px solid #1E2D4A",
              borderRadius: "10px",
              color: "#F9FAFB",
              fontSize: "14px",
              outline: "none",
              resize: "none",
              fontFamily: "Inter, sans-serif",
              lineHeight: "1.5",
              transition: "border-color 0.2s ease"
            }}
            onFocus={(e) => e.target.style.borderColor = "#3B82F6"}
            onBlur={(e) => e.target.style.borderColor = "#1E2D4A"}
          />
          <button
            onClick={handleQuery}
            disabled={loading || !question.trim()}
            style={{
              padding: "14px 20px",
              backgroundColor: loading || !question.trim() ? "#1E2D4A" : "#3B82F6",
              border: "none",
              borderRadius: "10px",
              color: loading || !question.trim() ? "#4B5563" : "#F9FAFB",
              fontSize: "14px",
              fontWeight: "600",
              cursor: loading || !question.trim() ? "not-allowed" : "pointer",
              transition: "all 0.2s ease",
              whiteSpace: "nowrap",
              fontFamily: "Inter, sans-serif"
            }}
          >
            {loading ? "Thinking..." : "Ask →"}
          </button>
        </div>
        <div style={{
          fontSize: "11px",
          color: "#4B5563",
          marginTop: "8px"
        }}>
          Press Enter to send · Shift+Enter for new line
        </div>
      </div>

      <style>{`
        @keyframes bounce {
          0%, 60%, 100% { transform: translateY(0); }
          30% { transform: translateY(-8px); }
        }
      `}</style>
    </div>
  )
}

export default QueryPanel