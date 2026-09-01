import { useCallback, useEffect, useRef, useState } from "react"
import Navigation from "./components/Navigation"
import Dashboard from "./components/Dashboard"
import Documents from "./components/Documents"
import Query from "./components/Query"
import {
  apiErrorMessage,
  fetchConfiguration,
  fetchDocuments,
  validateApiKey,
} from "./api"
import { theme } from "./styles/theme"

const SESSION_KEY = "lexis-api-key"

function App() {
  const [activePage, setActivePage] = useState("dashboard")
  const [apiKey, setApiKey] = useState(
    () => window.sessionStorage.getItem(SESSION_KEY) || "",
  )
  const initialApiKey = useRef(apiKey)
  const [authStatus, setAuthStatus] = useState(
    apiKey ? "checking" : "disconnected",
  )
  const [authMessage, setAuthMessage] = useState("")
  const [uploadedDocs, setUploadedDocs] = useState([])
  const [documentsLoading, setDocumentsLoading] = useState(false)
  const [documentsError, setDocumentsError] = useState("")
  const [selectedDocumentId, setSelectedDocumentId] = useState("")
  const [queryCount, setQueryCount] = useState(0)
  const [configuration, setConfiguration] = useState(null)

  const loadDocuments = useCallback(async (key) => {
    if (!key) return
    setDocumentsLoading(true)
    setDocumentsError("")
    try {
      const documents = await fetchDocuments(key)
      setUploadedDocs(documents)
      setSelectedDocumentId((currentId) => {
        const selected = documents.find(
          (document) => document.document_id === currentId,
        )
        return selected?.status === "ready" ? currentId : ""
      })
    } catch (error) {
      setDocumentsError(apiErrorMessage(error, "Could not load documents."))
      throw error
    } finally {
      setDocumentsLoading(false)
    }
  }, [])

  const connect = useCallback(async (key) => {
    const candidate = key.trim()
    if (!candidate) {
      setAuthStatus("disconnected")
      setAuthMessage("Enter the backend API key.")
      return
    }
    setAuthStatus("checking")
    setAuthMessage("Checking with the backend...")
    try {
      await validateApiKey(candidate)
      const [, activeConfiguration] = await Promise.all([
        loadDocuments(candidate),
        fetchConfiguration(candidate),
      ])
      window.sessionStorage.setItem(SESSION_KEY, candidate)
      setConfiguration(activeConfiguration)
      setAuthStatus("authenticated")
      setAuthMessage("Backend verified")
    } catch (error) {
      window.sessionStorage.removeItem(SESSION_KEY)
      setUploadedDocs([])
      setConfiguration(null)
      setAuthStatus("invalid")
      setAuthMessage(apiErrorMessage(error, "The backend rejected this key."))
    }
  }, [loadDocuments])

  useEffect(() => {
    if (initialApiKey.current) void connect(initialApiKey.current)
  }, [connect])

  const handleApiKeyChange = (value) => {
    setApiKey(value)
    window.sessionStorage.removeItem(SESSION_KEY)
    setAuthStatus(value.trim() ? "unvalidated" : "disconnected")
    setAuthMessage(value.trim() ? "Press Connect to validate" : "")
    setUploadedDocs([])
    setSelectedDocumentId("")
    setConfiguration(null)
  }

  const handleDisconnect = () => {
    window.sessionStorage.removeItem(SESSION_KEY)
    setApiKey("")
    setAuthStatus("disconnected")
    setAuthMessage("")
    setUploadedDocs([])
    setSelectedDocumentId("")
    setConfiguration(null)
  }

  const handleQueryDocument = (documentId) => {
    setSelectedDocumentId(documentId)
    setActivePage("query")
  }

  return (
    <div style={{
      display: "flex",
      height: "100vh",
      overflow: "hidden",
      backgroundColor: theme.colors.bg,
      fontFamily: theme.fonts.sans,
    }}>
      <Navigation
        activePage={activePage}
        setActivePage={setActivePage}
        apiKey={apiKey}
        setApiKey={handleApiKeyChange}
        authStatus={authStatus}
        authMessage={authMessage}
        onConnect={() => connect(apiKey)}
        onDisconnect={handleDisconnect}
      />
      <main style={{ flex: 1, overflow: "hidden", display: "flex" }}>
        {activePage === "dashboard" && (
          <Dashboard uploadedDocs={uploadedDocs} queryCount={queryCount} />
        )}
        {activePage === "documents" && (
          <Documents
            apiKey={apiKey}
            authenticated={authStatus === "authenticated"}
            uploadedDocs={uploadedDocs}
            documentsLoading={documentsLoading}
            documentsError={documentsError}
            onRefresh={() => loadDocuments(apiKey)}
            onQuery={handleQueryDocument}
          />
        )}
        {activePage === "query" && (
          <Query
            apiKey={apiKey}
            authenticated={authStatus === "authenticated"}
            uploadedDocs={uploadedDocs}
            selectedDocumentId={selectedDocumentId}
            setSelectedDocumentId={setSelectedDocumentId}
            setQueryCount={setQueryCount}
            configuration={configuration}
          />
        )}
      </main>
    </div>
  )
}

export default App
