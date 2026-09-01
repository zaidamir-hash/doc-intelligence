import { useCallback, useEffect, useState } from "react"
import { useDropzone } from "react-dropzone"
import {
  apiErrorMessage,
  deleteIndexedDocument,
  uploadPdf,
} from "../api"
import { theme } from "../styles/theme"

const statusStyles = {
  ready: { label: "Indexed", color: "#10B981", background: "rgba(16,185,129,0.1)" },
  processing: { label: "Processing", color: "#F59E0B", background: "rgba(245,158,11,0.1)" },
  failed: { label: "Failed", color: "#F87171", background: "rgba(239,68,68,0.1)" },
}

function Documents({
  apiKey,
  authenticated,
  uploadedDocs,
  documentsLoading,
  documentsError,
  onRefresh,
  onQuery,
}) {
  const [uploading, setUploading] = useState(false)
  const [uploadError, setUploadError] = useState("")
  const [uploadSuccess, setUploadSuccess] = useState("")
  const [pendingDeleteId, setPendingDeleteId] = useState("")
  const [deleteError, setDeleteError] = useState("")

  useEffect(() => {
    if (!uploading) return undefined
    const timer = window.setInterval(() => {
      void onRefresh().catch(() => {})
    }, 1500)
    return () => window.clearInterval(timer)
  }, [onRefresh, uploading])

  const onDrop = useCallback(async (acceptedFiles) => {
    const file = acceptedFiles[0]
    if (!file) return
    if (!authenticated) {
      setUploadError("Connect with a valid backend API key first.")
      return
    }

    setUploading(true)
    setUploadError("")
    setUploadSuccess("")
    try {
      const result = await uploadPdf(apiKey, file)
      setUploadSuccess(
        `${result.filename} is indexed with ${result.chunks_stored} chunks.`,
      )
    } catch (error) {
      setUploadError(apiErrorMessage(error, "Upload failed."))
    } finally {
      setUploading(false)
      try {
        await onRefresh()
      } catch {
        // The shared document-list error already gives the actionable message.
      }
    }
  }, [apiKey, authenticated, onRefresh])

  const handleDelete = async (documentId) => {
    setDeleteError("")
    try {
      await deleteIndexedDocument(apiKey, documentId)
      setPendingDeleteId("")
      await onRefresh()
    } catch (error) {
      setDeleteError(apiErrorMessage(error, "Document deletion failed."))
    }
  }

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: { "application/pdf": [".pdf"] },
    maxFiles: 1,
    disabled: uploading,
  })

  return (
    <div style={{ flex: 1, overflowY: "auto", padding: "32px 40px", backgroundColor: theme.colors.bg }}>
      <div style={{ marginBottom: "24px" }}>
        <div style={{ fontSize: "11px", color: theme.colors.accentBlue, textTransform: "uppercase", letterSpacing: "1.5px" }}>
          Knowledge Base
        </div>
        <h1 style={{ color: theme.colors.textPrimary, margin: "8px 0 6px" }}>Documents</h1>
        <p style={{ color: theme.colors.textSecondary, margin: 0 }}>
          The backend database is the source of truth for every row below.
        </p>
      </div>

      <div {...getRootProps()} style={{
        border: `2px dashed ${isDragActive ? theme.colors.accentBlue : theme.colors.border}`,
        borderRadius: theme.radius.xl,
        padding: "38px 24px",
        textAlign: "center",
        cursor: uploading ? "wait" : "pointer",
        backgroundColor: isDragActive ? theme.colors.accentGlow : theme.colors.surface,
        marginBottom: "18px",
      }}>
        <input {...getInputProps()} />
        <div style={{ fontSize: "34px", marginBottom: "10px" }}>{uploading ? "⏳" : "📄"}</div>
        <div style={{ color: theme.colors.textPrimary, fontWeight: "600" }}>
          {uploading ? "Extracting, chunking, embedding, and indexing..." : "Drop one PDF here or click to browse"}
        </div>
        <div style={{ color: theme.colors.textMuted, fontSize: "12px", marginTop: "7px" }}>
          Scanned/image-only PDFs need OCR, which is not enabled.
        </div>
      </div>

      {(uploadError || uploadSuccess || deleteError || documentsError) && (
        <div role="alert" style={{
          padding: "11px 14px",
          borderRadius: theme.radius.md,
          marginBottom: "16px",
          backgroundColor: uploadSuccess ? "rgba(16,185,129,0.08)" : "rgba(239,68,68,0.08)",
          color: uploadSuccess ? theme.colors.success : "#F87171",
          fontSize: "13px",
        }}>
          {uploadError || deleteError || documentsError || uploadSuccess}
        </div>
      )}

      <section style={{ backgroundColor: theme.colors.surface, border: `1px solid ${theme.colors.border}`, borderRadius: theme.radius.lg, overflow: "hidden" }}>
        <div style={{ padding: "18px 22px", display: "flex", justifyContent: "space-between", borderBottom: `1px solid ${theme.colors.border}` }}>
          <strong style={{ color: theme.colors.textPrimary }}>Current documents</strong>
          <button type="button" onClick={() => void onRefresh()} disabled={!authenticated || documentsLoading} style={{
            border: `1px solid ${theme.colors.border}`,
            borderRadius: theme.radius.sm,
            background: theme.colors.surfaceElevated,
            color: theme.colors.textSecondary,
            padding: "6px 10px",
            cursor: "pointer",
          }}>
            {documentsLoading ? "Refreshing..." : `Refresh · ${uploadedDocs.length}`}
          </button>
        </div>

        {!authenticated ? (
          <EmptyState text="Connect with a validated API key to restore indexed documents." />
        ) : uploadedDocs.length === 0 ? (
          <EmptyState text={documentsLoading ? "Loading documents..." : "No current documents are indexed."} />
        ) : (
          <div style={{ overflowX: "auto" }}>
            <table style={{ width: "100%", borderCollapse: "collapse" }}>
              <thead>
                <tr style={{ backgroundColor: theme.colors.surfaceElevated }}>
                  {["Document", "Pages", "Chunks", "Status", "Actions"].map((heading) => (
                    <th key={heading} style={{ padding: "11px 18px", textAlign: "left", fontSize: "11px", color: theme.colors.textMuted }}>
                      {heading}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {uploadedDocs.map((document) => {
                  const status = statusStyles[document.status] || statusStyles.failed
                  return (
                    <tr key={document.document_id} style={{ borderTop: `1px solid ${theme.colors.border}` }}>
                      <td style={{ padding: "14px 18px", color: theme.colors.textPrimary, fontFamily: theme.fonts.mono, fontSize: "12px" }}>
                        {document.filename}
                        {document.processing_error && (
                          <div style={{ color: "#F87171", fontFamily: theme.fonts.sans, marginTop: "5px" }}>
                            {document.processing_error}
                          </div>
                        )}
                      </td>
                      <td style={cellStyle}>{document.pages}</td>
                      <td style={cellStyle}>{document.chunks_stored}</td>
                      <td style={cellStyle}>
                        <span style={{ padding: "4px 9px", borderRadius: "20px", color: status.color, backgroundColor: status.background, fontWeight: "600", fontSize: "11px" }}>
                          {status.label}
                        </span>
                      </td>
                      <td style={{ ...cellStyle, whiteSpace: "nowrap" }}>
                        <button type="button" disabled={document.status !== "ready"} onClick={() => onQuery(document.document_id)} style={actionStyle(document.status === "ready")}>
                          Query →
                        </button>
                        {pendingDeleteId === document.document_id ? (
                          <>
                            <button type="button" onClick={() => void handleDelete(document.document_id)} style={dangerStyle}>Confirm delete</button>
                            <button type="button" onClick={() => setPendingDeleteId("")} style={plainStyle}>Cancel</button>
                          </>
                        ) : (
                          <button type="button" disabled={document.status === "processing"} onClick={() => setPendingDeleteId(document.document_id)} style={plainStyle}>
                            Delete
                          </button>
                        )}
                      </td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          </div>
        )}
      </section>
    </div>
  )
}

function EmptyState({ text }) {
  return <div style={{ padding: "48px", textAlign: "center", color: theme.colors.textMuted }}>{text}</div>
}

const cellStyle = { padding: "14px 18px", color: "#9CA3AF", fontSize: "13px" }
const plainStyle = { marginLeft: "7px", padding: "5px 9px", border: "1px solid #1E2D4A", borderRadius: "5px", color: "#9CA3AF", background: "transparent", cursor: "pointer" }
const dangerStyle = { ...plainStyle, color: "#F87171", borderColor: "rgba(239,68,68,0.5)" }
const actionStyle = (enabled) => ({
  padding: "5px 10px",
  border: "1px solid #3B82F6",
  borderRadius: "5px",
  color: enabled ? "#3B82F6" : "#4B5563",
  background: "rgba(59,130,246,0.08)",
  cursor: enabled ? "pointer" : "not-allowed",
})

export default Documents
