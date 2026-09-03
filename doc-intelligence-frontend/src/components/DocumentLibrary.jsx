import { useCallback, useEffect, useMemo, useState } from "react"
import { useDropzone } from "react-dropzone"
import { apiErrorDetails, deleteIndexedDocument, uploadPdf } from "../api"
import ConfirmDialog from "./ConfirmDialog"
import Icon from "./Icons"
import { EmptyState, Notice, Spinner, StatusBadge } from "./Primitives"

function formatFileSize(bytes) {
  if (!Number.isFinite(bytes)) return ""
  if (bytes < 1024 * 1024) return `${Math.max(1, Math.round(bytes / 1024))} KB`
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`
}

function formatDate(value) {
  if (!value) return "Not available"
  const date = new Date(value)
  if (Number.isNaN(date.getTime())) return "Not available"
  return new Intl.DateTimeFormat(undefined, {
    dateStyle: "medium",
    timeStyle: "short",
  }).format(date)
}

function DocumentLibrary({
  apiKey,
  documents,
  documentsLoading,
  documentsError,
  configuration,
  onRefresh,
  onAsk,
}) {
  const [searchTerm, setSearchTerm] = useState("")
  const [pendingFile, setPendingFile] = useState(null)
  const [uploading, setUploading] = useState(false)
  const [uploadError, setUploadError] = useState("")
  const [uploadSuccess, setUploadSuccess] = useState(null)
  const [documentToDelete, setDocumentToDelete] = useState(null)
  const [deleting, setDeleting] = useState(false)
  const [deleteError, setDeleteError] = useState("")
  const maxUploadBytes = configuration?.api?.max_upload_bytes

  useEffect(() => {
    if (!uploading) return undefined
    const timer = window.setInterval(() => {
      void onRefresh().catch(() => {})
    }, 1600)
    return () => window.clearInterval(timer)
  }, [onRefresh, uploading])

  const chooseFile = useCallback((acceptedFiles, rejectedFiles) => {
    setUploadError("")
    setUploadSuccess(null)
    const rejection = rejectedFiles[0]
    if (rejection) {
      const code = rejection.errors[0]?.code
      if (code === "file-too-large") {
        setUploadError(`This PDF is larger than the ${formatFileSize(maxUploadBytes)} upload limit.`)
      } else if (code === "too-many-files") {
        setUploadError("Choose one PDF at a time.")
      } else {
        setUploadError("Choose a valid PDF file. Other file types aren’t supported.")
      }
      setPendingFile(null)
      return
    }
    const file = acceptedFiles[0]
    if (file) setPendingFile(file)
  }, [maxUploadBytes])

  const { getRootProps, getInputProps, isDragActive, open } = useDropzone({
    onDrop: chooseFile,
    accept: { "application/pdf": [".pdf"] },
    maxFiles: 1,
    maxSize: maxUploadBytes,
    disabled: uploading,
    noClick: true,
    noKeyboard: true,
  })

  const handleUpload = async () => {
    if (!pendingFile || uploading) return
    setUploading(true)
    setUploadError("")
    setUploadSuccess(null)
    try {
      const result = await uploadPdf(apiKey, pendingFile)
      setUploadSuccess({
        documentId: result.document_id,
        filename: result.filename,
        reused: result.reused_existing_index,
      })
      setPendingFile(null)
      try {
        await onRefresh()
      } catch {
        // The upload response is authoritative for completion; the shared
        // document-list notice separately explains a refresh failure.
      }
    } catch (error) {
      const details = apiErrorDetails(error, "The PDF could not be uploaded. Try again.")
      setUploadError(details.message)
    } finally {
      setUploading(false)
    }
  }

  const handleDelete = async () => {
    if (!documentToDelete || deleting) return
    setDeleting(true)
    setDeleteError("")
    try {
      await deleteIndexedDocument(apiKey, documentToDelete.document_id)
      setDocumentToDelete(null)
      await onRefresh()
    } catch (error) {
      setDeleteError(apiErrorDetails(error, "The document could not be deleted.").message)
    } finally {
      setDeleting(false)
    }
  }

  const filteredDocuments = useMemo(() => {
    const query = searchTerm.trim().toLocaleLowerCase()
    if (!query) return documents
    return documents.filter((document) => document.filename.toLocaleLowerCase().includes(query))
  }, [documents, searchTerm])

  return (
    <main className="page library-page" id="main-content">
      <header className="page-header">
        <div>
          <p className="eyebrow">Workspace</p>
          <h1>Document library</h1>
          <p>Upload, monitor, and choose the source you want to research.</p>
        </div>
        <button type="button" className="button button--primary" onClick={open} disabled={uploading}>
          <Icon name="plus" />Upload PDF
        </button>
      </header>

      <section
        {...getRootProps({
          className: `upload-zone${isDragActive ? " is-drag-active" : ""}${uploading ? " is-busy" : ""}`,
        })}
        aria-label="PDF upload area"
      >
        <input {...getInputProps({ "aria-label": "Choose a PDF to upload" })} />
        <span className="upload-zone__icon"><Icon name="upload" size={24} /></span>
        <div className="upload-zone__copy">
          <strong>{isDragActive ? "Drop your PDF here" : "Drag and drop a PDF"}</strong>
          <span>Text-based PDF{maxUploadBytes ? ` · up to ${formatFileSize(maxUploadBytes)}` : ""} · one file at a time</span>
        </div>
        <button type="button" className="button button--secondary" onClick={open} disabled={uploading}>
          Browse files
        </button>
      </section>

      {pendingFile && (
        <section className="selected-upload" aria-labelledby="selected-upload-title">
          <span className="selected-upload__file-icon"><Icon name="document" /></span>
          <div className="selected-upload__details">
            <strong id="selected-upload-title" title={pendingFile.name}>{pendingFile.name}</strong>
            <span>{formatFileSize(pendingFile.size)} · Ready to upload</span>
          </div>
          <div className="selected-upload__actions">
            <button type="button" className="button button--ghost" onClick={() => setPendingFile(null)} disabled={uploading}>Remove</button>
            <button type="button" className="button button--primary" onClick={handleUpload} disabled={uploading}>
              {uploading ? <><Spinner />Indexing PDF…</> : <>Upload and index<Icon name="arrowRight" /></>}
            </button>
          </div>
        </section>
      )}

      {uploading && (
        <Notice title="Indexing your document">
          Extracting text, creating searchable passages, and preparing document retrieval. Keep this tab open while the synchronous upload completes.
        </Notice>
      )}
      {uploadError && (
        <Notice kind="error" title={uploadError.includes("OCR") || uploadError.includes("scanned") ? "This PDF needs OCR" : "Upload unsuccessful"}>
          {uploadError}
        </Notice>
      )}
      {uploadSuccess && (
        <Notice
          kind="success"
          title={uploadSuccess.reused ? "Document already indexed" : "Document ready"}
          actions={<button type="button" className="button button--success" onClick={() => onAsk(uploadSuccess.documentId)}>Ask a question<Icon name="arrowRight" /></button>}
        >
          <span className="truncate-inline" title={uploadSuccess.filename}>{uploadSuccess.filename}</span> is ready to research.
        </Notice>
      )}
      {documentsError && <Notice kind="error" title="Couldn’t refresh the library">{documentsError}</Notice>}

      <section className="library-section" aria-labelledby="library-heading">
        <div className="library-toolbar">
          <div>
            <h2 id="library-heading">Your documents</h2>
            <span>{documents.length} {documents.length === 1 ? "document" : "documents"}</span>
          </div>
          <div className="library-toolbar__actions">
            <label className="search-control">
              <span className="sr-only">Search documents</span>
              <Icon name="search" size={18} />
              <input type="search" value={searchTerm} onChange={(event) => setSearchTerm(event.target.value)} placeholder="Search documents" />
            </label>
            <button type="button" className="icon-button" aria-label="Refresh documents" title="Refresh documents" onClick={() => void onRefresh()} disabled={documentsLoading}>
              <Icon name="refresh" className={documentsLoading ? "is-spinning" : ""} />
            </button>
          </div>
        </div>

        {documentsLoading && documents.length === 0 ? (
          <DocumentSkeleton />
        ) : documents.length === 0 ? (
          <EmptyState
            icon="document"
            title="Bring your first document into focus"
            description="Upload a text-based PDF. Lexis will index its pages so every answer can point back to supporting evidence."
            action={<button type="button" className="button button--primary" onClick={open}><Icon name="upload" />Choose a PDF</button>}
          />
        ) : filteredDocuments.length === 0 ? (
          <EmptyState
            icon="search"
            title="No documents match"
            description={`Try a different search for “${searchTerm.trim()}”.`}
            action={<button type="button" className="button button--secondary" onClick={() => setSearchTerm("")}>Clear search</button>}
          />
        ) : (
          <div className="document-table-wrap">
            <table className="document-table">
              <thead>
                <tr>
                  <th scope="col">Document</th>
                  <th scope="col">Pages</th>
                  <th scope="col">Passages</th>
                  <th scope="col">Updated</th>
                  <th scope="col">Status</th>
                  <th scope="col"><span className="sr-only">Actions</span></th>
                </tr>
              </thead>
              <tbody>
                {filteredDocuments.map((document) => (
                  <tr key={document.document_id} className={document.status === "failed" ? "document-row document-row--failed" : "document-row"}>
                    <td data-label="Document">
                      <div className="document-name">
                        <span className="document-name__icon"><Icon name="document" /></span>
                        <div>
                          <strong title={document.filename}>{document.filename}</strong>
                          {document.processing_error && <span className="document-name__error">{document.processing_error}</span>}
                        </div>
                      </div>
                    </td>
                    <td data-label="Pages">{document.pages ?? "—"}</td>
                    <td data-label="Passages">{document.chunks_stored ?? "—"}</td>
                    <td data-label="Updated"><time dateTime={document.updated_at || undefined}>{formatDate(document.updated_at)}</time></td>
                    <td data-label="Status"><StatusBadge status={document.status} /></td>
                    <td className="document-actions">
                      <button type="button" className="button button--small button--secondary" disabled={document.status !== "ready"} onClick={() => onAsk(document.document_id)}>
                        Ask<Icon name="arrowRight" size={16} />
                      </button>
                      <button type="button" className="icon-button icon-button--danger" aria-label={`Delete ${document.filename}`} title="Delete document" disabled={document.status === "processing"} onClick={() => { setDeleteError(""); setDocumentToDelete(document) }}>
                        <Icon name="trash" size={18} />
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </section>

      <ConfirmDialog
        open={Boolean(documentToDelete)}
        title="Delete this document?"
        description={documentToDelete ? `“${documentToDelete.filename}” and its indexed passages will be permanently removed from the Lexis database.` : ""}
        confirmLabel="Delete document"
        busy={deleting}
        error={deleteError}
        onConfirm={handleDelete}
        onClose={() => { setDocumentToDelete(null); setDeleteError("") }}
      />
    </main>
  )
}

function DocumentSkeleton() {
  return (
    <div className="document-skeleton" role="status" aria-label="Loading documents">
      {[1, 2, 3].map((row) => <div key={row}><span /><span /><span /></div>)}
    </div>
  )
}

export default DocumentLibrary
