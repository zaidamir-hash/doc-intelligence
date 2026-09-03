import { useEffect, useRef } from "react"
import Icon from "./Icons"
import { Spinner } from "./Primitives"

function ConfirmDialog({ open, title, description, confirmLabel, busy, error, onConfirm, onClose }) {
  const dialogRef = useRef(null)

  useEffect(() => {
    const dialog = dialogRef.current
    if (open && !dialog.open) dialog.showModal()
    if (!open && dialog.open) dialog.close()
  }, [open])

  return (
    <dialog
      ref={dialogRef}
      className="dialog"
      onCancel={(event) => {
        event.preventDefault()
        if (!busy) onClose()
      }}
      onClick={(event) => {
        if (event.target === dialogRef.current && !busy) onClose()
      }}
    >
      <div className="dialog__body">
        <button className="icon-button dialog__close" type="button" aria-label="Close dialog" onClick={onClose} disabled={busy}>
          <Icon name="close" />
        </button>
        <span className="dialog__icon"><Icon name="trash" size={22} /></span>
        <h2>{title}</h2>
        <p>{description}</p>
        {error && <p className="field-error" role="alert">{error}</p>}
      </div>
      <div className="dialog__actions">
        <button type="button" className="button button--secondary" onClick={onClose} disabled={busy}>Cancel</button>
        <button type="button" className="button button--danger" onClick={onConfirm} disabled={busy}>
          {busy ? <><Spinner size={16} />Deleting…</> : confirmLabel}
        </button>
      </div>
    </dialog>
  )
}

export default ConfirmDialog
