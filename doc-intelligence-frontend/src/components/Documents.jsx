import { useState, useCallback } from "react"
import { useDropzone } from "react-dropzone"
import axios from "axios"
import { theme } from "../styles/theme"

const API_URL = "http://127.0.0.1:8000"

function Documents({ apiKey, uploadedDocs, setUploadedDocs }) {
  const [uploading, setUploading] = useState(false)
  const [uploadError, setUploadError] = useState("")
  const [uploadSuccess, setUploadSuccess] = useState("")

  const onDrop = useCallback(async (acceptedFiles) => {
    const file = acceptedFiles[0]
    if (!file) return

    if (!apiKey.trim()) {
      setUploadError("Please enter your API key in the sidebar first")
      return
    }

    setUploading(true)
    setUploadError("")
    setUploadSuccess("")

    const formData = new FormData()
    formData.append("file", file)

    try {
      const response = await axios.post(`${API_URL}/upload`, formData, {
        headers: {
          "X-API-Key": apiKey,
          "Content-Type": "multipart/form-data"
        }
      })

      setUploadedDocs(prev => [...prev, {
        filename: response.data.filename,
        pages: response.data.pages,
        chunks: response.data.chunks_stored
      }])
      setUploadSuccess(`Successfully indexed ${response.data.filename} — ${response.data.chunks_stored} chunks created`)
    } catch (err) {
      setUploadError(err.response?.data?.detail || "Upload failed")
    } finally {
      setUploading(false)
    }
  }, [apiKey, setUploadedDocs])

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: { "application/pdf": [".pdf"] },
    maxFiles: 1
  })

  return (
    <div style={{
      flex: 1,
      overflowY: "auto",
      padding: "32px 40px",
      backgroundColor: theme.colors.bg,
    }}>
      {/* Header */}
      <div style={{ marginBottom: "32px" }}>
        <div style={{
          fontSize: "11px",
          fontWeight: "600",
          color: theme.colors.accentBlue,
          textTransform: "uppercase",
          letterSpacing: "1.5px",
          marginBottom: "8px",
        }}>
          Knowledge Base
        </div>
        <h1 style={{
          fontSize: "28px",
          fontWeight: "700",
          color: theme.colors.textPrimary,
          letterSpacing: "-0.8px",
          margin: 0,
        }}>
          Documents
        </h1>
        <p style={{
          fontSize: "14px",
          color: theme.colors.textSecondary,
          marginTop: "6px",
        }}>
          Upload and manage your document knowledge base
        </p>
      </div>

      {/* Upload Zone */}
      <div
        {...getRootProps()}
        style={{
          border: `2px dashed ${isDragActive ? theme.colors.accentBlue : theme.colors.border}`,
          borderRadius: theme.radius.xl,
          padding: "48px 32px",
          textAlign: "center",
          cursor: "pointer",
          backgroundColor: isDragActive ? theme.colors.accentGlow : theme.colors.surface,
          transition: "all 0.2s ease",
          marginBottom: "24px",
          boxShadow: isDragActive ? theme.shadows.glow : "none",
        }}
      >
        <input {...getInputProps()} />
        <div style={{ fontSize: "40px", marginBottom: "16px" }}>
          {uploading ? "⏳" : isDragActive ? "📂" : "📄"}
        </div>
        <div style={{
          fontSize: "16px",
          fontWeight: "600",
          color: isDragActive ? theme.colors.accentBlue : theme.colors.textPrimary,
          marginBottom: "8px",
        }}>
          {uploading
            ? "Processing document..."
            : isDragActive
            ? "Release to upload"
            : "Drop your PDF here"}
        </div>
        <div style={{
          fontSize: "13px",
          color: theme.colors.textSecondary,
          marginBottom: "20px",
        }}>
          {uploading
            ? "Extracting text, generating embeddings and indexing chunks"
            : "or click to browse files"}
        </div>
        {!uploading && (
          <div style={{
            display: "inline-flex",
            alignItems: "center",
            gap: "8px",
            padding: "8px 20px",
            backgroundColor: theme.colors.accentGlow,
            border: `1px solid ${theme.colors.accentBlue}`,
            borderRadius: theme.radius.md,
            fontSize: "13px",
            fontWeight: "600",
            color: theme.colors.accentBlue,
          }}>
            Browse Files
          </div>
        )}

        {uploading && (
          <div style={{
            display: "flex",
            justifyContent: "center",
            gap: "6px",
            marginTop: "8px",
          }}>
            {[0, 1, 2].map(i => (
              <div key={i} style={{
                width: "8px",
                height: "8px",
                backgroundColor: theme.colors.accentBlue,
                borderRadius: "50%",
                animation: `bounce 1.2s ease-in-out ${i * 0.2}s infinite`
              }} />
            ))}
          </div>
        )}
      </div>

      {/* Error / Success Messages */}
      {uploadError && (
        <div style={{
          padding: "12px 16px",
          backgroundColor: "rgba(239,68,68,0.08)",
          border: "1px solid rgba(239,68,68,0.2)",
          borderRadius: theme.radius.md,
          fontSize: "13px",
          color: "#F87171",
          marginBottom: "16px",
          display: "flex",
          alignItems: "center",
          gap: "8px",
        }}>
          ⚠️ {uploadError}
        </div>
      )}

      {uploadSuccess && (
        <div style={{
          padding: "12px 16px",
          backgroundColor: "rgba(16,185,129,0.08)",
          border: "1px solid rgba(16,185,129,0.2)",
          borderRadius: theme.radius.md,
          fontSize: "13px",
          color: theme.colors.success,
          marginBottom: "16px",
          display: "flex",
          alignItems: "center",
          gap: "8px",
        }}>
          ✅ {uploadSuccess}
        </div>
      )}

      {/* Documents Table */}
      <div style={{
        backgroundColor: theme.colors.surface,
        border: `1px solid ${theme.colors.border}`,
        borderRadius: theme.radius.lg,
        overflow: "hidden",
      }}>
        <div style={{
          padding: "20px 24px",
          borderBottom: `1px solid ${theme.colors.border}`,
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
        }}>
          <div style={{
            fontSize: "14px",
            fontWeight: "600",
            color: theme.colors.textPrimary,
          }}>
            Indexed Documents
          </div>
          <div style={{
            padding: "4px 12px",
            backgroundColor: theme.colors.accentGlow,
            border: `1px solid ${theme.colors.accentBlue}`,
            borderRadius: "20px",
            fontSize: "12px",
            fontWeight: "600",
            color: theme.colors.accentBlue,
          }}>
            {uploadedDocs.length} documents
          </div>
        </div>

        {uploadedDocs.length === 0 ? (
          <div style={{
            padding: "64px 24px",
            textAlign: "center",
          }}>
            <div style={{ fontSize: "40px", marginBottom: "12px" }}>📭</div>
            <div style={{
              fontSize: "15px",
              fontWeight: "600",
              color: theme.colors.textSecondary,
              marginBottom: "6px",
            }}>
              No documents indexed yet
            </div>
            <div style={{
              fontSize: "13px",
              color: theme.colors.textMuted,
            }}>
              Upload a PDF above to get started
            </div>
          </div>
        ) : (
          <table style={{ width: "100%", borderCollapse: "collapse" }}>
            <thead>
              <tr style={{ backgroundColor: theme.colors.surfaceElevated }}>
                {["Document", "Pages", "Chunks", "Status", "Action"].map(h => (
                  <th key={h} style={{
                    padding: "12px 24px",
                    textAlign: "left",
                    fontSize: "11px",
                    fontWeight: "600",
                    color: theme.colors.textMuted,
                    textTransform: "uppercase",
                    letterSpacing: "0.8px",
                  }}>
                    {h}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {uploadedDocs.map((doc, i) => (
                <tr key={i} style={{
                  borderTop: `1px solid ${theme.colors.border}`,
                }}>
                  <td style={{
                    padding: "16px 24px",
                    fontSize: "13px",
                    color: theme.colors.textPrimary,
                    fontWeight: "500",
                    fontFamily: theme.fonts.mono,
                  }}>
                    📄 {doc.filename}
                  </td>
                  <td style={{
                    padding: "16px 24px",
                    fontSize: "13px",
                    color: theme.colors.textSecondary,
                  }}>
                    {doc.pages}
                  </td>
                  <td style={{
                    padding: "16px 24px",
                    fontSize: "13px",
                    color: theme.colors.textSecondary,
                  }}>
                    {doc.chunks}
                  </td>
                  <td style={{
                    padding: "16px 24px",
                  }}>
                    <span style={{
                      display: "inline-flex",
                      alignItems: "center",
                      gap: "6px",
                      padding: "3px 10px",
                      borderRadius: "20px",
                      backgroundColor: "rgba(16,185,129,0.1)",
                      border: "1px solid rgba(16,185,129,0.2)",
                      fontSize: "11px",
                      fontWeight: "600",
                      color: theme.colors.success,
                    }}>
                      <div style={{
                        width: "5px",
                        height: "5px",
                        borderRadius: "50%",
                        backgroundColor: theme.colors.success,
                        boxShadow: `0 0 4px ${theme.colors.success}`,
                      }} />
                      Indexed
                    </span>
                  </td>
                  <td style={{
                    padding: "16px 24px",
                  }}>
                    <button style={{
                      padding: "5px 14px",
                      backgroundColor: theme.colors.accentGlow,
                      border: `1px solid ${theme.colors.accentBlue}`,
                      borderRadius: theme.radius.sm,
                      fontSize: "12px",
                      fontWeight: "600",
                      color: theme.colors.accentBlue,
                      cursor: "pointer",
                      fontFamily: theme.fonts.sans,
                    }}>
                      Query →
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
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

export default Documents