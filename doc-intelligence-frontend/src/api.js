import axios from "axios"

export const API_URL = import.meta.env.VITE_API_URL || "http://127.0.0.1:8000"
const API_KEY_HEADER = import.meta.env.VITE_API_KEY_HEADER || "X-API-Key"

const client = axios.create({ baseURL: API_URL, timeout: 120000 })

function authHeaders(apiKey) {
  return { headers: { [API_KEY_HEADER]: apiKey } }
}

export function apiErrorMessage(error, fallback) {
  const detail = error.response?.data?.detail
  if (typeof detail === "string") return detail
  if (detail && typeof detail.message === "string") return detail.message
  return fallback
}

export async function validateApiKey(apiKey) {
  const response = await client.get("/auth/validate", authHeaders(apiKey))
  return response.data
}

export async function fetchConfiguration(apiKey) {
  const response = await client.get("/configuration", authHeaders(apiKey))
  return response.data
}

export async function fetchDocuments(apiKey) {
  const response = await client.get("/documents", authHeaders(apiKey))
  return response.data.documents
}

export async function uploadPdf(apiKey, file) {
  const formData = new FormData()
  formData.append("file", file)
  const response = await client.post("/upload", formData, authHeaders(apiKey))
  return response.data
}

export async function deleteIndexedDocument(apiKey, documentId) {
  const response = await client.delete(
    `/documents/${documentId}`,
    authHeaders(apiKey),
  )
  return response.data
}

export async function askDocument(apiKey, documentId, question, debug) {
  const response = await client.post(
    "/query",
    { document_id: documentId, question, debug },
    authHeaders(apiKey),
  )
  return response.data
}
