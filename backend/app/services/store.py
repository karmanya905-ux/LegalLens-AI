from pathlib import Path
from sqlalchemy import create_engine, text


class DocumentStore:
    def __init__(self, path: str):
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        self.engine = create_engine(f"sqlite:///{p.resolve().as_posix()}", future=True)
        with self.engine.begin() as conn:
            conn.execute(text("""CREATE TABLE IF NOT EXISTS documents (
                id TEXT PRIMARY KEY, filename TEXT NOT NULL, path TEXT NOT NULL,
                pages INTEGER NOT NULL, chunks INTEGER NOT NULL, created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )"""))
            conn.execute(text("""CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT, document_id TEXT NOT NULL,
                event_type TEXT NOT NULL, content TEXT NOT NULL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )"""))

    def add_document(self, document_id, filename, path, pages, chunks):
        with self.engine.begin() as conn:
            conn.execute(text("""INSERT INTO documents (id,filename,path,pages,chunks)
                VALUES (:id,:filename,:path,:pages,:chunks)"""),
                {"id": document_id, "filename": filename, "path": path, "pages": pages, "chunks": chunks})

    def get_document(self, document_id):
        with self.engine.connect() as conn:
            row = conn.execute(text("SELECT * FROM documents WHERE id=:id"), {"id": document_id}).mappings().first()
            return dict(row) if row else None

    def list_documents(self):
        with self.engine.connect() as conn:
            return [dict(r) for r in conn.execute(text(
                "SELECT id,filename,pages,chunks,created_at FROM documents ORDER BY created_at DESC"
            )).mappings().all()]

    def add_event(self, document_id, event_type, content):
        with self.engine.begin() as conn:
            conn.execute(text("""INSERT INTO events (document_id,event_type,content)
                VALUES (:document_id,:event_type,:content)"""),
                {"document_id": document_id, "event_type": event_type, "content": content})
