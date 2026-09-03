import { useCallback, useEffect, useRef, useState } from "react"
import { apiErrorDetails, fetchConfiguration, fetchDocuments, validateApiKey } from "./api"
import AppShell from "./components/AppShell"
import AskWorkspace from "./components/AskWorkspace"
import ConnectionScreen from "./components/ConnectionScreen"
import DocumentLibrary from "./components/DocumentLibrary"

const SESSION_KEY = "lexis-api-key"

function App() {
  const [activePage, setActivePage] = useState("library")
  const [apiKey, setApiKey] = useState(
    () => window.sessionStorage.getItem(SESSION_KEY) || "",
  )
  const initialApiKey = useRef(apiKey)
  const [authStatus, setAuthStatus] = useState(
    apiKey ? "checking" : "disconnected",
  )
  const [authMessage, setAuthMessage] = useState("")
  const [documents, setDocuments] = useState([])
  const [documentsLoading, setDocumentsLoading] = useState(false)
  const [documentsError, setDocumentsError] = useState("")
  const [selectedDocumentId, setSelectedDocumentId] = useState("")
  const [configuration, setConfiguration] = useState(null)
  const [conversations, setConversations] = useState({})

  const loadDocuments = useCallback(async (key) => {
    if (!key) return []
    setDocumentsLoading(true)
    setDocumentsError("")
    try {
      const nextDocuments = await fetchDocuments(key)
      setDocuments(nextDocuments)
      setSelectedDocumentId((currentId) => {
        const selected = nextDocuments.find(
          (document) => document.document_id === currentId,
        )
        return selected?.status === "ready" ? currentId : ""
      })
      return nextDocuments
    } catch (error) {
      setDocumentsError(
        apiErrorDetails(error, "We couldn’t refresh your documents.").message,
      )
      throw error
    } finally {
      setDocumentsLoading(false)
    }
  }, [])

  const connect = useCallback(async (key) => {
    const candidate = key.trim()
    if (!candidate) {
      setAuthStatus("disconnected")
      setAuthMessage("Enter the API key provided by your Lexis backend.")
      return
    }

    setAuthStatus("checking")
    setAuthMessage("Verifying a secure connection to Lexis…")
    try {
      await validateApiKey(candidate)
      const [, activeConfiguration] = await Promise.all([
        loadDocuments(candidate),
        fetchConfiguration(candidate),
      ])
      window.sessionStorage.setItem(SESSION_KEY, candidate)
      setApiKey(candidate)
      setConfiguration(activeConfiguration)
      setAuthStatus("authenticated")
      setAuthMessage("")
    } catch (error) {
      const details = apiErrorDetails(error, "Lexis could not validate this connection.")
      window.sessionStorage.removeItem(SESSION_KEY)
      setDocuments([])
      setConfiguration(null)
      setAuthStatus(details.kind === "unreachable" ? "unreachable" : "invalid")
      setAuthMessage(details.message)
    }
  }, [loadDocuments])

  useEffect(() => {
    if (initialApiKey.current) void connect(initialApiKey.current)
  }, [connect])

  const handleApiKeyChange = (value) => {
    setApiKey(value)
    setAuthMessage("")
    if (authStatus !== "authenticated") {
      setAuthStatus(value.trim() ? "unvalidated" : "disconnected")
    }
  }

  const handleDisconnect = () => {
    window.sessionStorage.removeItem(SESSION_KEY)
    setApiKey("")
    setAuthStatus("disconnected")
    setAuthMessage("")
    setDocuments([])
    setDocumentsError("")
    setSelectedDocumentId("")
    setConfiguration(null)
    setConversations({})
    setActivePage("library")
  }

  const openDocumentWorkspace = (documentId) => {
    setSelectedDocumentId(documentId)
    setActivePage("ask")
  }

  if (authStatus !== "authenticated") {
    return (
      <ConnectionScreen
        apiKey={apiKey}
        authStatus={authStatus}
        message={authMessage}
        onApiKeyChange={handleApiKeyChange}
        onConnect={() => connect(apiKey)}
      />
    )
  }

  return (
    <AppShell
      activePage={activePage}
      onNavigate={setActivePage}
      onDisconnect={handleDisconnect}
      documents={documents}
    >
      {activePage === "library" ? (
        <DocumentLibrary
          apiKey={apiKey}
          documents={documents}
          documentsLoading={documentsLoading}
          documentsError={documentsError}
          configuration={configuration}
          onRefresh={() => loadDocuments(apiKey)}
          onAsk={openDocumentWorkspace}
        />
      ) : (
        <AskWorkspace
          apiKey={apiKey}
          documents={documents}
          selectedDocumentId={selectedDocumentId}
          onSelectDocument={setSelectedDocumentId}
          onOpenLibrary={() => setActivePage("library")}
          conversations={conversations}
          setConversations={setConversations}
        />
      )}
    </AppShell>
  )
}

export default App
