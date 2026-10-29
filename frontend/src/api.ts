const API = import.meta.env.VITE_API_URL || "http://127.0.0.1:8000";
export async function listDocuments() {
  const r = await fetch(`${API}/documents`); if (!r.ok) throw new Error(await r.text()); return r.json();
}
export async function uploadDocument(file: File) {
  const fd = new FormData(); fd.append("file", file);
  const r = await fetch(`${API}/documents/upload`, { method: "POST", body: fd });
  if (!r.ok) throw new Error((await r.json()).detail || "Upload failed"); return r.json();
}
export async function askQuestion(document_id: string, question: string) {
  const r = await fetch(`${API}/chat`, { method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify({document_id,question}) });
  if (!r.ok) throw new Error((await r.json()).detail || "Question failed"); return r.json();
}
export async function getReport(documentId: string, kind: "summary"|"risk-review") {
  const r = await fetch(`${API}/documents/${documentId}/${kind}`, { method:"POST" });
  if (!r.ok) throw new Error((await r.json()).detail || "Report failed"); return r.json();
}

export async function classifyClause(text: string) {
  const r = await fetch(`${API}/classify-clause`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text }),
  });
  const data = await r.json();
  if (!r.ok) throw new Error(data.detail || "Clause classification failed");
  return data;
}


export async function runAgent(document_id: string, request: string) {
  const r = await fetch(`${API}/agent`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ document_id, request }),
  });

  const data = await r.json();

  if (!r.ok) {
    throw new Error(data.detail || "Agent request failed");
  }

  return data;
}

