
import { useEffect, useState } from 'react';
import { FileText, Upload, Send, ShieldCheck, BookOpen, RefreshCw, AlertCircle, Scale, Brain } from 'lucide-react';
import { askQuestion, getReport, listDocuments, uploadDocument, classifyClause, runAgent } from './api';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';


type Doc = { id:string; filename:string; pages:number; chunks:number; created_at?:string };
type Message = { role:'user'|'assistant'; text:string; sources?:string[] };

type Classification = {
  status: string;
  prediction: string | null;
  decision_score?: number | null;
  top_categories?: { category: string; decision_score: number }[];
  disclaimer?: string;
  message?: string;
};

async function handleClassifyClause(text: string): Promise<Classification> {
  const response = await fetch('http://127.0.0.1:8000/classify-clause', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ text }),
  });

  const data = await response.json();

  if (!response.ok) {
    throw new Error(data.detail || 'Clause classification failed.');
  }

  return data;
}

export default function App() {
  const [docs,setDocs] = useState<Doc[]>([]);
  const [selected,setSelected] = useState('');
  const [file,setFile] = useState<File|null>(null);
  const [question,setQuestion] = useState('');
  const [messages,setMessages] = useState<Message[]>([]);
  const [busy,setBusy] = useState(false);
  const [error,setError] = useState('');
  const [report,setReport] = useState('');
  const [tab, setTab] = useState<'chat'|'summary'|'risk'|'classify'|'agent'>('chat');
  const [clauseText, setClauseText] = useState('');
  const [classification, setClassification] = useState<any>(null);
  const [agentRequest, setAgentRequest] = useState('');
  const [agentResult, setAgentResult] = useState<any>(null);

  async function refresh() { try { const d=await listDocuments(); setDocs(d.documents); if (!selected && d.documents.length) setSelected(d.documents[0].id); } catch(e) { setError(String(e)); } }
  useEffect(()=>{ void refresh(); },[]);
  async function doUpload() {
    if(!file) return; setBusy(true); setError('');
    try { const d=await uploadDocument(file); await refresh(); setSelected(d.document_id); setMessages([]); setReport(''); setFile(null); }
    catch(e){setError(String(e));} finally{setBusy(false);}
  }
  async function send() {
    if(!selected || !question.trim()) return; const q=question.trim(); setQuestion(''); setMessages(m=>[...m,{role:'user',text:q}]); setBusy(true); setError('');
    try { const a=await askQuestion(selected,q); setMessages(m=>[...m,{role:'assistant',text:a.answer,sources:a.sources||[]}]); }
    catch(e){setError(String(e));} finally{setBusy(false);}
  }
  async function makeReport(kind:'summary'|'risk-review') {
    if(!selected) return; setBusy(true); setError(''); setReport('');
    try { const r=await getReport(selected,kind); setReport(kind==='summary'?r.summary:r.risk_review); }
    catch(e){setError(String(e));} finally{setBusy(false);}
  }
async function doClassify() {
  if (!clauseText.trim()) return;

  setBusy(true);
  setError('');
  setClassification(null);

  try {
    const result = await classifyClause(clauseText.trim());
    setClassification(result);
  } catch (e) {
    setError(String(e));
  } finally {
    setBusy(false);
  }
}
  async function runClassification() {
    if (!clauseText.trim() || busy) return;

    setBusy(true);
    setError('');
    setClassification(null);
  
    try {
      const result = await classifyClause(clauseText.trim());

      if (result.status !== 'ok') {
        throw new Error(result.message || 'The classifier could not process this clause.');
      }

      setClassification(result);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  }

async function doRunAgent() {
  if (!selected || !agentRequest.trim() || busy) return;

  setBusy(true);
  setError('');
  setAgentResult(null);

  try {
    const result = await runAgent(selected, agentRequest.trim());
    setAgentResult(result);
  } catch (e) {
    setError(e instanceof Error ? e.message : String(e));
  } finally {
    setBusy(false);
  }
}

  return <div className="shell">
    <aside className="sidebar">
      <div className="brand"><div className="brandIcon"><Scale size={22}/></div><div><strong>LegalLens AI</strong><small>Legal document intelligence</small></div></div>
      <div className="sideLabel">WORKSPACE</div>
      <label className="uploadBox"><Upload size={18}/><span>{file?file.name:'Choose a PDF'}</span><input type="file" accept="application/pdf,.pdf" onChange={e=>setFile(e.target.files?.[0]||null)}/></label>
      <button className="primary full" disabled={!file||busy} onClick={()=>void doUpload()}>{busy?'Working…':'Upload & index PDF'}</button>
      <div className="sideLabel">YOUR DOCUMENTS <button className="iconButton" onClick={()=>void refresh()} title="Refresh"><RefreshCw size={14}/></button></div>
      <div className="docList">{docs.map(d=><button key={d.id} className={`docItem ${selected===d.id?'active':''}`} onClick={()=>{setSelected(d.id);setMessages([]);setReport('')}}><FileText size={17}/><span>{d.filename}<small>{d.pages} pages · {d.chunks} chunks</small></span></button>)}</div>
      <div className="sidebarFoot"><ShieldCheck size={16}/> Local document storage</div>
    </aside>
    <main className="main">
      <header><div><span className="eyebrow">LEGAL AI WORKSPACE</span><h1>Understand your documents.</h1><p>Ask questions, summarize agreements, and flag clauses worth reviewing.</p></div><div className="status"><span/> API-connected</div></header>
      <nav className="tabs">
        <button className={tab==='chat'?'selected':''} onClick={()=>{setTab('chat');setReport('')}}><BookOpen size={16}/> Ask a question</button>
        <button className={tab==='summary'?'selected':''} onClick={()=>{setTab('summary');setReport('')}}><FileText size={16}/> Summary</button>
        <button className={tab==='risk'?'selected':''} onClick={()=>{setTab('risk');setReport('')}}><ShieldCheck size={16}/> Risk review</button>
<button
  className={tab === 'agent' ? 'selected' : ''}
  onClick={() => {
    setTab('agent');
    setReport('');
    setError('');
    setAgentResult(null);
  }}
>
  <Brain size={16} /> Legal AI Agent
</button>
       
        <button className={tab === 'classify' ? 'selected' : ''} onClick={() => {setTab('classify');setReport('');setError('');}}><Brain size={16} /> Classify clause</button>
      </nav>
      {error && <div className="error"><AlertCircle size={17}/>{error}</div>}
     {!selected ? (
  <section className="empty">
    <div className="emptyIcon"><FileText size={30}/></div>
    <h2>Start with a document</h2>
    <p>Upload a text-based PDF to index it and begin asking questions.</p>
  </section>

) : tab === 'agent' ? (
  <section className="reportPanel">
    <div className="reportIntro">
      <div className="emptyIcon"><Brain size={26} /></div>
      <h2>Legal AI Agent</h2>
      <p>
        Ask a question, summarize your contract, or review potential risks.
        LangGraph routes your request to the appropriate workflow.
      </p>
    </div>

    <form
      className="classifierForm"
      onSubmit={e => {
        e.preventDefault();
        void doRunAgent();
      }}
    >
      <label htmlFor="agentRequest">What would you like the agent to do?</label>
      <textarea
        id="agentRequest"
        value={agentRequest}
        onChange={e => setAgentRequest(e.target.value)}
        placeholder="e.g. Review this contract for potential risks"
        rows={4}
        disabled={busy}
      />
      <button
        className="primary"
        type="submit"
        disabled={busy || !agentRequest.trim()}
      >
        {busy ? 'Working…' : 'Run Legal AI Agent'}
      </button>
    </form>

    {agentResult && (
      <article className="reportText markdown-content">
        <h2>Agent result</h2>
        <p><strong>Selected workflow:</strong> {agentResult.action}</p>
        <ReactMarkdown remarkPlugins={[remarkGfm]}>
          {agentResult.result}
        </ReactMarkdown>

        {agentResult.sources?.length > 0 && (
          <details>
            <summary>Retrieved evidence ({agentResult.sources.length})</summary>
            {agentResult.sources.map((source: string, i: number) => (
              <blockquote key={i}>{source}</blockquote>
            ))}
          </details>
        )}
      </article>
    )}
  </section>
) : tab === 'classify' ? (

  <section className="reportPanel">
    <div className="reportIntro">
      <div className="emptyIcon"><Brain size={26}/></div>
      <h2>Classify a legal clause</h2>
      <p>Predict a general contract-clause category using your trained classifier.</p>
    </div>
    <div className="classifierForm">
      <label htmlFor="clauseText">Clause text</label>
      <textarea
        id="clauseText"
        value={clauseText}
        onChange={e => setClauseText(e.target.value)}
        placeholder="Paste a legal clause here..."
        rows={7}
        disabled={busy}
      />
      <button
        className="primary"
        disabled={busy || !clauseText.trim()}
        onClick={() => void runClassification()}
      >
        {busy ? 'Classifying…' : 'Classify clause'}
      </button>
    </div>
    {classification && (
      <article className="reportText">
        <h2>Prediction</h2>
        <p><strong>{classification.prediction}</strong></p>
        <h3>Top categories</h3>
        <ol>
          {classification.top_categories?.map((item: { category: string; decision_score: number }) => (
            <li key={item.category}>
              <strong>{item.category}</strong>
              {' — decision score: '}
              {item.decision_score.toFixed(3)}
            </li>
          ))}
        </ol>
        <p className="classifierDisclaimer">{classification.disclaimer}</p>
      </article>
    )}
  </section>
) : tab === 'chat' ? (
  <section className="chatPanel">
    {messages.length === 0 ? (
      <div className="empty">
        <div className="emptyIcon"><BookOpen size={30}/></div>
        <h2>Ask your document anything</h2>
        <p>Answers use retrieved passages and page citations where available.</p>
        <div className="suggestions">
          {['What is the notice period?', 'Summarize the main obligations', 'What terms are not specified?'].map(q => (
            <button key={q} onClick={() => setQuestion(q)}>{q}</button>
          ))}
        </div>
      </div>
    ) : (
      <div className="messages">
        {messages.map((m, i) => (
          <article key={i} className={`message ${m.role}`}>
            <div className="role">{m.role === 'user' ? 'YOU' : 'LEGALLENS AI'}</div>
            <div className="messageText markdown-content">
              <ReactMarkdown remarkPlugins={[remarkGfm]}>{m.text}</ReactMarkdown>
            </div>
            {m.sources?.length ? (
              <details>
                <summary>Retrieved evidence ({m.sources.length})</summary>
                {m.sources.map((s, j) => <blockquote key={j}>{s}</blockquote>)}
              </details>
            ) : null}
          </article>
        ))}
      </div>
    )}
    <form className="composer" onSubmit={e => { e.preventDefault(); void send(); }}>
      <input
        value={question}
        onChange={e => setQuestion(e.target.value)}
        placeholder="Ask about salary, notice period, confidentiality…"
        disabled={busy}
      />
      <button className="primary" disabled={busy || !question.trim()}>
        <Send size={17}/><span>Send</span>
      </button>
    </form>
  </section>
) : (
  <section className="reportPanel">
    <div className="reportIntro">
      <div className="emptyIcon">
        {tab === 'summary' ? <BookOpen size={26}/> : <ShieldCheck size={26}/>}
      </div>
      <h2>{tab === 'summary' ? 'Document summary' : 'Contract risk review'}</h2>
      <p>
        {tab === 'summary'
          ? 'Extract key terms and obligations with evidence.'
          : 'Identify unclear, one-sided, or missing terms to discuss with a qualified professional.'}
      </p>
      <button
        className="primary"
        disabled={busy}
        onClick={() => void makeReport(tab === 'summary' ? 'summary' : 'risk-review')}
      >
        {busy ? 'Analyzing…' : tab === 'summary' ? 'Generate summary' : 'Review potential risks'}
      </button>
    </div>
    {report && (
      <article className="reportText markdown-content">
        <ReactMarkdown remarkPlugins={[remarkGfm]}>{report}</ReactMarkdown>
      </article>
    )}
  </section>
)}

      <footer><span><ShieldCheck size={14}/> Educational use only — not legal advice.</span><span>LegalLens AI · Capstone build</span></footer>
    </main>
  </div>;
}
