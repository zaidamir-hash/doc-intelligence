import { useState } from "react"
import Navigation from "./components/Navigation"
import Dashboard from "./components/Dashboard"
import Documents from "./components/Documents"
import Query from "./components/Query"
import { theme } from "./styles/theme"

function App() {
  const [activePage, setActivePage] = useState("dashboard")
  const [apiKey, setApiKey] = useState("")
  const [uploadedDocs, setUploadedDocs] = useState([])
  const [queryCount, setQueryCount] = useState(0)

  return (
    <div style={{
      display: "flex",
      height: "100vh",
      overflow: "hidden",
      backgroundColor: theme.colors.bg,
      fontFamily: theme.fonts.sans
    }}>
      <Navigation 
        activePage={activePage} 
        setActivePage={setActivePage}
        apiKey={apiKey}
        setApiKey={setApiKey}
      />
      <main style={{
        flex: 1,
        overflow: "hidden",
        display: "flex",
        flexDirection: "column"
      }}>
        <div style={{ display: activePage === "dashboard" ? "flex" : "none", flex: 1, flexDirection: "column", overflow: "hidden" }}>
          <Dashboard uploadedDocs={uploadedDocs} queryCount={queryCount} />
        </div>
        <div style={{ display: activePage === "documents" ? "flex" : "none", flex: 1, flexDirection: "column", overflow: "hidden" }}>
          <Documents apiKey={apiKey} uploadedDocs={uploadedDocs} setUploadedDocs={setUploadedDocs} />
        </div>
        <div style={{ display: activePage === "query" ? "flex" : "none", flex: 1, flexDirection: "column", overflow: "hidden" }}>
          <Query apiKey={apiKey} setQueryCount={setQueryCount} uploadedDocs={uploadedDocs} />
        </div>
      </main>
    </div>
  )
}

export default App