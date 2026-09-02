# Lexis frontend

This Vite/React application is the user interface for the Lexis API. The root
[`README.md`](../README.md) is the authoritative setup and operations guide.

The frontend deliberately contains no retrieval implementation. It validates a
backend API key, renders the backend-authoritative document lifecycle, sends a
stable document UUID with each question, and displays grounded citations plus
optional retrieval diagnostics.

```powershell
npm ci
npm run dev
```

Optional `.env.local` values:

```dotenv
VITE_API_URL=http://127.0.0.1:8000
VITE_API_KEY_HEADER=X-API-Key
```

Quality checks:

```powershell
npm run lint
npm run build
```
