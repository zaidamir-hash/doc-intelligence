import { theme } from "../styles/theme"

const navItems = [
  { id: "dashboard", label: "Dashboard", icon: "⬛" },
  { id: "documents", label: "Documents", icon: "📄" },
  { id: "query", label: "Query", icon: "💬" },
]

function Navigation({ activePage, setActivePage, apiKey, setApiKey }) {
  return (
    <div style={{
      width: "240px",
      minWidth: "240px",
      height: "100vh",
      backgroundColor: theme.colors.surface,
      borderRight: `1px solid ${theme.colors.border}`,
      display: "flex",
      flexDirection: "column",
      padding: "0",
    }}>
      {/* Logo */}
      <div style={{
        padding: "28px 24px",
        borderBottom: `1px solid ${theme.colors.border}`,
      }}>
        <div style={{
          fontSize: "22px",
          fontWeight: "700",
          letterSpacing: "-0.8px",
          color: theme.colors.textPrimary,
        }}>
          Lex<span style={{
            color: theme.colors.accentBlue,
            position: "relative",
          }}>is</span>
        </div>
        <div style={{
          fontSize: "11px",
          color: theme.colors.textMuted,
          marginTop: "4px",
          letterSpacing: "0.5px",
        }}>
          Document Intelligence
        </div>
      </div>

      {/* Nav Items */}
      <nav style={{
        padding: "16px 12px",
        flex: 1,
        display: "flex",
        flexDirection: "column",
        gap: "4px",
      }}>
        {navItems.map(item => (
          <button
            key={item.id}
            onClick={() => setActivePage(item.id)}
            style={{
              display: "flex",
              alignItems: "center",
              gap: "12px",
              padding: "10px 12px",
              borderRadius: theme.radius.md,
              border: "none",
              cursor: "pointer",
              backgroundColor: activePage === item.id
                ? theme.colors.accentGlow
                : "transparent",
              color: activePage === item.id
                ? theme.colors.accentBlue
                : theme.colors.textSecondary,
              fontSize: "14px",
              fontWeight: activePage === item.id ? "600" : "400",
              fontFamily: theme.fonts.sans,
              textAlign: "left",
              width: "100%",
              transition: "all 0.15s ease",
              borderLeft: activePage === item.id
                ? `2px solid ${theme.colors.accentBlue}`
                : "2px solid transparent",
            }}
          >
            <span style={{ fontSize: "16px" }}>{item.icon}</span>
            {item.label}
          </button>
        ))}
      </nav>

      {/* API Key Section */}
      <div style={{
        padding: "16px",
        borderTop: `1px solid ${theme.colors.border}`,
      }}>
        <div style={{
          fontSize: "10px",
          fontWeight: "600",
          color: theme.colors.textMuted,
          textTransform: "uppercase",
          letterSpacing: "1px",
          marginBottom: "8px",
        }}>
          API Key
        </div>
        <input
          type="password"
          placeholder="Enter API key"
          value={apiKey}
          onChange={(e) => setApiKey(e.target.value)}
          style={{
            width: "100%",
            padding: "9px 12px",
            backgroundColor: theme.colors.bg,
            border: `1px solid ${theme.colors.border}`,
            borderRadius: theme.radius.sm,
            color: theme.colors.textPrimary,
            fontSize: "12px",
            outline: "none",
            fontFamily: theme.fonts.mono,
            boxSizing: "border-box",
          }}
        />
        {/* Status indicator */}
        <div style={{
          display: "flex",
          alignItems: "center",
          gap: "6px",
          marginTop: "8px",
        }}>
          <div style={{
            width: "6px",
            height: "6px",
            borderRadius: "50%",
            backgroundColor: apiKey.trim() ? theme.colors.success : theme.colors.textMuted,
            boxShadow: apiKey.trim() ? `0 0 6px ${theme.colors.success}` : "none",
            transition: "all 0.3s ease",
          }} />
          <span style={{
            fontSize: "11px",
            color: apiKey.trim() ? theme.colors.success : theme.colors.textMuted,
          }}>
            {apiKey.trim() ? "Connected" : "Not connected"}
          </span>
        </div>
      </div>

      {/* Footer */}
      <div style={{
        padding: "12px 16px",
        borderTop: `1px solid ${theme.colors.border}`,
        fontSize: "10px",
        color: theme.colors.textMuted,
        display: "flex",
        alignItems: "center",
        gap: "6px",
      }}>
        <div style={{
          width: "6px",
          height: "6px",
          borderRadius: "50%",
          backgroundColor: theme.colors.success,
          boxShadow: `0 0 6px ${theme.colors.success}`,
        }} />
        System operational
      </div>
    </div>
  )
}

export default Navigation