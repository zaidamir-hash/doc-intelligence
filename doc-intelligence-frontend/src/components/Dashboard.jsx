import { theme } from "../styles/theme"

function StatCard({ label, value, icon, color, glow }) {
  return (
    <div style={{
      backgroundColor: theme.colors.surface,
      border: `1px solid ${theme.colors.border}`,
      borderRadius: theme.radius.lg,
      padding: "24px",
      display: "flex",
      flexDirection: "column",
      gap: "12px",
      boxShadow: glow ? theme.shadows.glow : theme.shadows.card,
      position: "relative",
      overflow: "hidden",
      transition: "all 0.2s ease",
    }}>
      {/* Background accent */}
      <div style={{
        position: "absolute",
        top: 0,
        right: 0,
        width: "80px",
        height: "80px",
        borderRadius: "0 14px 0 80px",
        backgroundColor: `${color}10`,
      }} />

      <div style={{
        fontSize: "24px",
      }}>
        {icon}
      </div>

      <div>
        <div style={{
          fontSize: "32px",
          fontWeight: "700",
          color: color,
          letterSpacing: "-1px",
          lineHeight: "1",
        }}>
          {value}
        </div>
        <div style={{
          fontSize: "13px",
          color: theme.colors.textSecondary,
          marginTop: "6px",
          fontWeight: "500",
        }}>
          {label}
        </div>
      </div>
    </div>
  )
}

function Dashboard({ uploadedDocs, queryCount }) {
  const readyDocuments = uploadedDocs.filter((doc) => doc.status === "ready")
  const totalChunks = readyDocuments.reduce((sum, doc) => sum + (doc.chunks_stored || 0), 0)
  const totalPages = uploadedDocs.reduce((sum, doc) => sum + (doc.pages || 0), 0)

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
          Overview
        </div>
        <h1 style={{
          fontSize: "28px",
          fontWeight: "700",
          color: theme.colors.textPrimary,
          letterSpacing: "-0.8px",
          margin: 0,
        }}>
          Dashboard
        </h1>
        <p style={{
          fontSize: "14px",
          color: theme.colors.textSecondary,
          marginTop: "6px",
        }}>
          Your document intelligence at a glance
        </p>
      </div>

      {/* Stat Cards */}
      <div style={{
        display: "grid",
        gridTemplateColumns: "repeat(4, 1fr)",
        gap: "16px",
        marginBottom: "32px",
      }}>
        <StatCard
          label="Documents Uploaded"
          value={readyDocuments.length}
          icon="📁"
          color={theme.colors.accentBlue}
          glow={true}
        />
        <StatCard
          label="Queries Made"
          value={queryCount}
          icon="💬"
          color={theme.colors.accentPurple}
        />
        <StatCard
          label="Pages Processed"
          value={totalPages}
          icon="📄"
          color={theme.colors.success}
        />
        <StatCard
          label="Chunks Stored"
          value={totalChunks}
          icon="🧩"
          color="#F59E0B"
        />
      </div>

      {/* Documents Table */}
      <div style={{
        backgroundColor: theme.colors.surface,
        border: `1px solid ${theme.colors.border}`,
        borderRadius: theme.radius.lg,
        overflow: "hidden",
        marginBottom: "24px",
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
            Recent Documents
          </div>
          <div style={{
            fontSize: "12px",
            color: theme.colors.textMuted,
          }}>
            {uploadedDocs.length} total
          </div>
        </div>

        {uploadedDocs.length === 0 ? (
          <div style={{
            padding: "48px 24px",
            textAlign: "center",
            color: theme.colors.textMuted,
          }}>
            <div style={{ fontSize: "32px", marginBottom: "12px" }}>📭</div>
            <div style={{ fontSize: "14px" }}>No documents uploaded yet</div>
            <div style={{ fontSize: "12px", marginTop: "4px" }}>
              Go to Documents to upload your first PDF
            </div>
          </div>
        ) : (
          <table style={{
            width: "100%",
            borderCollapse: "collapse",
          }}>
            <thead>
              <tr style={{
                backgroundColor: theme.colors.surfaceElevated,
              }}>
                {["Filename", "Pages", "Chunks", "Status"].map(h => (
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
              {uploadedDocs.map((doc) => (
                <tr key={doc.document_id} style={{
                  borderTop: `1px solid ${theme.colors.border}`,
                  transition: "background 0.15s ease",
                }}>
                  <td style={{
                    padding: "14px 24px",
                    fontSize: "13px",
                    color: theme.colors.textPrimary,
                    fontWeight: "500",
                    fontFamily: theme.fonts.mono,
                  }}>
                    {doc.filename}
                  </td>
                  <td style={{
                    padding: "14px 24px",
                    fontSize: "13px",
                    color: theme.colors.textSecondary,
                  }}>
                    {doc.pages}
                  </td>
                  <td style={{
                    padding: "14px 24px",
                    fontSize: "13px",
                    color: theme.colors.textSecondary,
                  }}>
                    {doc.chunks_stored}
                  </td>
                  <td style={{
                    padding: "14px 24px",
                  }}>
                    <span style={{
                      display: "inline-flex",
                      alignItems: "center",
                      gap: "6px",
                      padding: "3px 10px",
                      borderRadius: "20px",
                      backgroundColor: doc.status === "ready"
                        ? "rgba(16,185,129,0.1)"
                        : doc.status === "processing"
                          ? "rgba(245,158,11,0.1)"
                          : "rgba(239,68,68,0.1)",
                      border: `1px solid ${doc.status === "ready" ? "rgba(16,185,129,0.2)" : "rgba(239,68,68,0.2)"}`,
                      fontSize: "11px",
                      fontWeight: "600",
                      color: doc.status === "ready"
                        ? theme.colors.success
                        : doc.status === "processing" ? "#F59E0B" : "#F87171",
                    }}>
                      <div style={{
                        width: "5px",
                        height: "5px",
                        borderRadius: "50%",
                        backgroundColor: doc.status === "ready"
                          ? theme.colors.success
                          : doc.status === "processing" ? "#F59E0B" : "#F87171",
                      }} />
                      {doc.status === "ready" ? "Indexed" : doc.status}
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>

      {/* Quick Actions */}
      <div style={{
        backgroundColor: theme.colors.surface,
        border: `1px solid ${theme.colors.border}`,
        borderRadius: theme.radius.lg,
        padding: "24px",
      }}>
        <div style={{
          fontSize: "14px",
          fontWeight: "600",
          color: theme.colors.textPrimary,
          marginBottom: "16px",
        }}>
          Getting Started
        </div>
        <div style={{
          display: "flex",
          flexDirection: "column",
          gap: "12px",
        }}>
          {[
            { step: "1", text: "Enter your API key in the sidebar", done: false },
            { step: "2", text: "Upload a PDF document in the Documents tab", done: false },
            { step: "3", text: "Ask questions about your document in the Query tab", done: false },
          ].map(item => (
            <div key={item.step} style={{
              display: "flex",
              alignItems: "center",
              gap: "14px",
              padding: "12px 16px",
              backgroundColor: theme.colors.surfaceElevated,
              borderRadius: theme.radius.md,
              border: `1px solid ${theme.colors.border}`,
            }}>
              <div style={{
                width: "26px",
                height: "26px",
                borderRadius: "50%",
                backgroundColor: theme.colors.accentGlow,
                border: `1px solid ${theme.colors.accentBlue}`,
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                fontSize: "12px",
                fontWeight: "700",
                color: theme.colors.accentBlue,
                flexShrink: 0,
              }}>
                {item.step}
              </div>
              <span style={{
                fontSize: "13px",
                color: theme.colors.textSecondary,
              }}>
                {item.text}
              </span>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}

export default Dashboard
