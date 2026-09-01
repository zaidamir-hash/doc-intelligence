import { theme } from "../styles/theme"

const navItems = [
  { id: "dashboard", label: "Dashboard", icon: "⬛" },
  { id: "documents", label: "Documents", icon: "📄" },
  { id: "query", label: "Query", icon: "💬" },
]

const statusPresentation = {
  authenticated: { label: "Connected", color: theme.colors.success },
  checking: { label: "Checking...", color: "#F59E0B" },
  invalid: { label: "Not connected", color: "#F87171" },
  unvalidated: { label: "Not validated", color: "#F59E0B" },
  disconnected: { label: "Not connected", color: theme.colors.textMuted },
}

function Navigation({
  activePage,
  setActivePage,
  apiKey,
  setApiKey,
  authStatus,
  authMessage,
  onConnect,
  onDisconnect,
}) {
  const status = statusPresentation[authStatus] || statusPresentation.disconnected

  return (
    <aside style={{
      width: "240px",
      minWidth: "240px",
      height: "100vh",
      backgroundColor: theme.colors.surface,
      borderRight: `1px solid ${theme.colors.border}`,
      display: "flex",
      flexDirection: "column",
    }}>
      <div style={{ padding: "28px 24px", borderBottom: `1px solid ${theme.colors.border}` }}>
        <div style={{ fontSize: "22px", fontWeight: "700", color: theme.colors.textPrimary }}>
          Lex<span style={{ color: theme.colors.accentBlue }}>is</span>
        </div>
        <div style={{ fontSize: "11px", color: theme.colors.textMuted, marginTop: "4px" }}>
          Document Intelligence
        </div>
      </div>

      <nav style={{ padding: "16px 12px", flex: 1 }}>
        {navItems.map((item) => (
          <button
            type="button"
            key={item.id}
            onClick={() => setActivePage(item.id)}
            style={{
              display: "flex",
              gap: "12px",
              alignItems: "center",
              width: "100%",
              padding: "10px 12px",
              marginBottom: "4px",
              borderRadius: theme.radius.md,
              border: "none",
              borderLeft: activePage === item.id
                ? `2px solid ${theme.colors.accentBlue}`
                : "2px solid transparent",
              backgroundColor: activePage === item.id
                ? theme.colors.accentGlow
                : "transparent",
              color: activePage === item.id
                ? theme.colors.accentBlue
                : theme.colors.textSecondary,
              cursor: "pointer",
              fontFamily: theme.fonts.sans,
              fontWeight: activePage === item.id ? "600" : "400",
            }}
          >
            <span>{item.icon}</span>{item.label}
          </button>
        ))}
      </nav>

      <div style={{ padding: "16px", borderTop: `1px solid ${theme.colors.border}` }}>
        <label htmlFor="api-key" style={{
          display: "block",
          fontSize: "10px",
          fontWeight: "600",
          color: theme.colors.textMuted,
          textTransform: "uppercase",
          marginBottom: "8px",
        }}>
          Backend API Key
        </label>
        <input
          id="api-key"
          type="password"
          autoComplete="off"
          placeholder="Enter API key"
          value={apiKey}
          onChange={(event) => setApiKey(event.target.value)}
          onKeyDown={(event) => {
            if (event.key === "Enter") onConnect()
          }}
          style={{
            width: "100%",
            boxSizing: "border-box",
            padding: "9px 12px",
            backgroundColor: theme.colors.bg,
            border: `1px solid ${theme.colors.border}`,
            borderRadius: theme.radius.sm,
            color: theme.colors.textPrimary,
          }}
        />
        <button
          type="button"
          onClick={authStatus === "authenticated" ? onDisconnect : onConnect}
          disabled={authStatus === "checking"}
          style={{
            width: "100%",
            marginTop: "8px",
            padding: "8px",
            borderRadius: theme.radius.sm,
            border: `1px solid ${theme.colors.accentBlue}`,
            backgroundColor: theme.colors.accentGlow,
            color: theme.colors.accentBlue,
            cursor: authStatus === "checking" ? "wait" : "pointer",
            fontWeight: "600",
          }}
        >
          {authStatus === "authenticated" ? "Disconnect" : "Connect"}
        </button>
        <div style={{ display: "flex", alignItems: "center", gap: "6px", marginTop: "9px" }}>
          <span style={{ width: "6px", height: "6px", borderRadius: "50%", backgroundColor: status.color }} />
          <span style={{ fontSize: "11px", color: status.color }}>{status.label}</span>
        </div>
        {authMessage && (
          <div style={{ fontSize: "10px", color: theme.colors.textMuted, marginTop: "5px", lineHeight: "1.4" }}>
            {authMessage}
          </div>
        )}
      </div>
    </aside>
  )
}

export default Navigation
