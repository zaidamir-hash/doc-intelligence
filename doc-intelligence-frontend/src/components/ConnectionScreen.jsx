import { useId } from "react"
import { API_URL } from "../api"
import Icon from "./Icons"
import { Notice, Spinner } from "./Primitives"

function ConnectionScreen({ apiKey, authStatus, message, onApiKeyChange, onConnect }) {
  const errorId = useId()
  const checking = authStatus === "checking"
  const hasError = authStatus === "invalid" || authStatus === "unreachable"

  const submit = (event) => {
    event.preventDefault()
    if (!checking && apiKey.trim()) onConnect()
  }

  return (
    <main className="connection-page">
      <section className="connection-intro" aria-labelledby="connection-title">
        <a className="brand brand--connection" href="/" aria-label="Lexis home">
          <span className="brand__mark" aria-hidden="true">L</span>
          <span>Lexis</span>
        </a>
        <div className="connection-intro__content">
          <p className="eyebrow">Document intelligence</p>
          <h1 id="connection-title">Answers you can trace back to the page.</h1>
          <p>
            Upload text-based PDFs, ask precise questions, and inspect the passages
            Lexis used to ground each response.
          </p>
          <ul className="feature-list">
            <li><Icon name="document" />Persistent document library</li>
            <li><Icon name="search" />Hybrid evidence retrieval</li>
            <li><Icon name="bookOpen" />Page-aware sources and passages</li>
          </ul>
        </div>
        <p className="connection-intro__note">Built for focused document research.</p>
      </section>

      <section className="connection-panel" aria-label="Connect to Lexis">
        <form className="connection-form" onSubmit={submit}>
          <div className="connection-form__icon"><Icon name="shield" size={24} /></div>
          <p className="eyebrow">Secure connection</p>
          <h2>Connect to your backend</h2>
          <p className="connection-form__description">
            Enter the API key configured by the Lexis server. This validates access;
            it is not a user-account login.
          </p>

          <label htmlFor="api-key">API key</label>
          <input
            id="api-key"
            type="password"
            autoComplete="off"
            autoFocus={!checking}
            value={apiKey}
            onChange={(event) => onApiKeyChange(event.target.value)}
            aria-describedby={hasError ? errorId : "connection-storage-note"}
            aria-invalid={hasError}
            placeholder="Enter your API key"
            disabled={checking}
          />

          {message && (
            <div id={errorId} className="connection-message" aria-live="polite">
              {hasError ? <Notice kind="error">{message}</Notice> : (
                <span><Spinner size={16} />{message}</span>
              )}
            </div>
          )}

          <button className="button button--primary button--large" type="submit" disabled={checking || !apiKey.trim()}>
            {checking ? <><Spinner />Connecting…</> : <>Connect securely<Icon name="arrowRight" /></>}
          </button>

          <p id="connection-storage-note" className="connection-form__storage-note">
            Your key stays in this browser tab only and is cleared when you disconnect
            or close the tab.
          </p>
          <div className="connection-form__endpoint">
            <span>Backend</span>
            <code>{API_URL}</code>
          </div>
        </form>
      </section>
    </main>
  )
}

export default ConnectionScreen
