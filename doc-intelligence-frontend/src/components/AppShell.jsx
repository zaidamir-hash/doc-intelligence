import { useEffect, useRef, useState } from "react"
import { API_URL } from "../api"
import Icon from "./Icons"

const navigation = [
  { id: "library", label: "Library", icon: "library" },
  { id: "ask", label: "Ask", icon: "search" },
]

function AppShell({ activePage, onNavigate, onDisconnect, documents, children }) {
  const [connectionOpen, setConnectionOpen] = useState(false)
  const connectionRef = useRef(null)
  const readyCount = documents.filter((document) => document.status === "ready").length

  useEffect(() => {
    if (!connectionOpen) return undefined
    const closeOnOutsideClick = (event) => {
      if (!connectionRef.current?.contains(event.target)) setConnectionOpen(false)
    }
    const closeOnEscape = (event) => {
      if (event.key === "Escape") setConnectionOpen(false)
    }
    document.addEventListener("pointerdown", closeOnOutsideClick)
    document.addEventListener("keydown", closeOnEscape)
    return () => {
      document.removeEventListener("pointerdown", closeOnOutsideClick)
      document.removeEventListener("keydown", closeOnEscape)
    }
  }, [connectionOpen])

  return (
    <div className="app-shell">
      <header className="topbar">
        <a className="brand" href="/" aria-label="Lexis home">
          <span className="brand__mark" aria-hidden="true">L</span>
          <span>Lexis</span>
        </a>

        <nav className="primary-nav" aria-label="Primary navigation">
          {navigation.map((item) => (
            <button
              key={item.id}
              type="button"
              className={activePage === item.id ? "primary-nav__item is-active" : "primary-nav__item"}
              aria-current={activePage === item.id ? "page" : undefined}
              onClick={() => onNavigate(item.id)}
            >
              <Icon name={item.icon} />
              <span>{item.label}</span>
            </button>
          ))}
        </nav>

        <div className="connection-menu" ref={connectionRef}>
          <button
            type="button"
            className="connection-trigger"
            aria-expanded={connectionOpen}
            aria-haspopup="menu"
            onClick={() => setConnectionOpen((open) => !open)}
          >
            <span className="connection-trigger__dot" aria-hidden="true" />
            <span className="connection-trigger__label">Connected</span>
            <Icon name="chevronDown" size={16} />
          </button>
          {connectionOpen && (
            <div className="connection-popover" role="menu">
              <div className="connection-popover__status">
                <span className="connection-trigger__dot" aria-hidden="true" />
                <div><strong>Backend connected</strong><span>{readyCount} ready {readyCount === 1 ? "document" : "documents"}</span></div>
              </div>
              <div className="connection-popover__endpoint"><span>Endpoint</span><code>{API_URL}</code></div>
              <button type="button" className="connection-popover__disconnect" role="menuitem" onClick={onDisconnect}>
                Disconnect
              </button>
            </div>
          )}
        </div>
      </header>

      <div className="app-content">{children}</div>
    </div>
  )
}

export default AppShell
