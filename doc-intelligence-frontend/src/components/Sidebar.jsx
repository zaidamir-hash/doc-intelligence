import { useState, useCallback } from "react"
import { useDropzone } from "react-dropzone"
import axios from "axios"

const API_URL = "http://127.0.0.1:8000"

function Sidebar({ uploadedDocs, setUploadedDocs, apiKey, setApiKey }) {
  const [uploading, setUploading] = useState(false)
  const [uploadError, setUploadError] = useState("")
  const [uploadSuccess, setUploadSuccess] = useState("")

  const onDrop = useCallback(async (acceptedFiles) => {
    const file = acceptedFiles[0]
    if (!file) return

    if (!apiKey.trim()) {
      setUploadError("Please enter your API key first")
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
      setUploadSuccess(`Successfully uploaded ${response.data.filename}`)
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
      width: "320px",
      minWidth: "320px",
      height: "100vh",
      backgroundColor: "#111827",
      borderRight: "1px solid #1E2D4A",
      display: "flex",
      flexDirection: "column",
      padding: "24px 20px",
      overflowY: "auto"
    }}>
      {/* Logo */}
      <div style={{ marginBottom: "32px" }}>
        <div style={{
          fontSize: "20px",
          fontWeight: "700",
          color: "#F9FAFB",
          letterSpacing: "-0.5px"
        }}>
          Doc<span style={{ color: "#3B82F6" }}>Intel</span>
        </div>
        <div style={{
          fontSize: "12px",
          color: "#9CA3AF",
          marginTop: "4px"
        }}>
          Document Intelligence Platform
        </div>
      </div>

      {/* API Key Input */}
      <div style={{ marginBottom: "24px" }}>
        <label style={{
          fontSize: "11px",
          fontWeight: "600",
          color: "#9CA3AF",
          textTransform: "uppercase",
          letterSpacing: "0.8px",
          display: "block",
          marginBottom: "8px"
        }}>
          API Key
        </label>
        <input
          type="password"
          placeholder="Enter your API key"
          value={apiKey}
          onChange={(e) => setApiKey(e.target.value)}
          style={{
            width: "100%",
            padding: "10px 12px",
            backgroundColor: "#0A0F1E",
            border: "1px solid #1E2D4A",
            borderRadius: "8px",
            color: "#F9FAFB",
            fontSize: "13px",
            outline: "none",
            fontFamily: "JetBrains Mono, monospace"
          }}
        />
      </div>

      {/* Upload Zone */}
      <div style={{ marginBottom: "24px" }}>
        <label style={{
          fontSize: "11px",
          fontWeight: "600",
          color: "#9CA3AF",
          textTransform: "uppercase",
          letterSpacing: "0.8px",
          display: "block",
          marginBottom: "8px"
        }}>
          Upload Document
        </label>
        <div
          {...getRootProps()}
          style={{
            border: `2px dashed ${isDragActive ? "#3B82F6" : "#1E2D4A"}`,
            borderRadius: "10px",
            padding: "24px 16px",
            textAlign: "center",
            cursor: "pointer",
            backgroundColor: isDragActive ? "rgba(59,130,246,0.05)" : "transparent",
            transition: "all 0.2s ease"
          }}
        >
          <input {...getInputProps()} />
          <div style={{ fontSize: "24px", marginBottom: "8px" }}>📄</div>
          <div style={{
            fontSize: "13px",
            color: isDragActive ? "#3B82F6" : "#9CA3AF",
            fontWeight: "500"
          }}>
            {uploading
              ? "Uploading..."
              : isDragActive
              ? "Drop PDF here"
              : "Drag & drop PDF or click"}
          </div>
          <div style={{
            fontSize: "11px",
            color: "#4B5563",
            marginTop: "4px"
          }}>
            PDF files only
          </div>
        </div>

        {uploadError && (
          <div style={{
            marginTop: "8px",
            padding: "8px 12px",
            backgroundColor: "rgba(239,68,68,0.1)",
            border: "1px solid rgba(239,68,68,0.2)",
            borderRadius: "6px",
            fontSize: "12px",
            color: "#F87171"
          }}>
            {uploadError}
          </div>
        )}

        {uploadSuccess && (
          <div style={{
            marginTop: "8px",
            padding: "8px 12px",
            backgroundColor: "rgba(16,185,129,0.1)",
            border: "1px solid rgba(16,185,129,0.2)",
            borderRadius: "6px",
            fontSize: "12px",
            color: "#34D399"
          }}>
            {uploadSuccess}
          </div>
        )}
      </div>

      {/* Uploaded Documents List */}
      {uploadedDocs.length > 0 && (
        <div>
          <label style={{
            fontSize: "11px",
            fontWeight: "600",
            color: "#9CA3AF",
            textTransform: "uppercase",
            letterSpacing: "0.8px",
            display: "block",
            marginBottom: "8px"
          }}>
            Documents ({uploadedDocs.length})
          </label>
          <div style={{ display: "flex", flexDirection: "column", gap: "8px" }}>
            {uploadedDocs.map((doc, index) => (
              <div key={index} style={{
                padding: "10px 12px",
                backgroundColor: "#0A0F1E",
                borderRadius: "8px",
                border: "1px solid #1E2D4A"
              }}>
                <div style={{
                  fontSize: "12px",
                  fontWeight: "600",
                  color: "#F9FAFB",
                  marginBottom: "4px",
                  wordBreak: "break-all"
                }}>
                  {doc.filename}
                </div>
                <div style={{
                  fontSize: "11px",
                  color: "#9CA3AF"
                }}>
                  {doc.pages} pages · {doc.chunks} chunks
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Footer */}
      <div style={{
        marginTop: "auto",
        paddingTop: "24px",
        borderTop: "1px solid #1E2D4A",
        fontSize: "11px",
        color: "#4B5563"
      }}>
        Powered by RAG + pgvector
      </div>
    </div>
  )
}

export default Sidebar