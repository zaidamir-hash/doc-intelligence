import Icon from "./Icons"

const statusLabels = {
  ready: "Ready",
  processing: "Processing",
  failed: "Failed",
}

export function StatusBadge({ status }) {
  return (
    <span className={`status-badge status-badge--${status}`}>
      <span className="status-badge__dot" aria-hidden="true" />
      {statusLabels[status] || status}
    </span>
  )
}

export function Notice({ kind = "info", title, children, actions }) {
  const icon = kind === "error" ? "warning" : kind === "success" ? "check" : "info"
  return (
    <div className={`notice notice--${kind}`} role={kind === "error" ? "alert" : "status"}>
      <Icon name={icon} size={18} />
      <div className="notice__content">
        {title && <strong>{title}</strong>}
        {children && <div>{children}</div>}
      </div>
      {actions && <div className="notice__actions">{actions}</div>}
    </div>
  )
}

export function Spinner({ size = 18 }) {
  return <span className="spinner" style={{ width: size, height: size }} aria-hidden="true" />
}

export function EmptyState({ icon = "document", title, description, action }) {
  return (
    <div className="empty-state">
      <span className="empty-state__icon"><Icon name={icon} size={26} /></span>
      <h2>{title}</h2>
      <p>{description}</p>
      {action}
    </div>
  )
}
