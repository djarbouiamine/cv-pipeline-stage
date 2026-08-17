import os
import json
import hashlib
from typing import Optional, Dict, Any, List, Set

# -------------------------------------------------------------
# Configuration – PostgreSQL support (optional)
# -------------------------------------------------------------
_POSTGRES_ENV = {
    "host": os.getenv("POSTGRES_HOST"),
    "port": os.getenv("POSTGRES_PORT"),
    "dbname": os.getenv("POSTGRES_DB"),
    "user": os.getenv("POSTGRES_USER"),
    "password": os.getenv("POSTGRES_PASSWORD"),
}

_USE_POSTGRES = all(_POSTGRES_ENV.values())

if _USE_POSTGRES:
    try:
        import psycopg2
        from psycopg2.extras import Json
    except Exception as e:
        print("⚠️ PostgreSQL driver not available – falling back to JSON cache", e)
        _USE_POSTGRES = False

# -------------------------------------------------------------
# Helper – SHA‑256 of a file (raw bytes)
# -------------------------------------------------------------
def file_sha256(path: str) -> str:
    """Return the SHA‑256 hex digest of *path*'s binary contents."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()

# -------------------------------------------------------------
# CVCache – thin abstraction over PostgreSQL or JSON fallback
# -------------------------------------------------------------
class CVCache:
    """Persist extracted CV data.

    Each record is identified by the SHA‑256 hash of the original PDF file.
    Stored fields:
        - hash (primary key)
        - email (lower‑cased string or empty)
        - phone (digits‑only string or empty)
        - data (full JSON payload from cv_extractor)
        - embedding (optional, binary for Postgres, omitted for JSON fallback)
        - source_path (original filename for traceability)
    """

    def __init__(self, json_path: str = "output/cv_cache.json"):
        self.json_path = json_path
        if _USE_POSTGRES:
            self._init_pg()
        else:
            self._load_json()

    # -----------------------------------------------------------------
    # PostgreSQL implementation
    # -----------------------------------------------------------------
    def _init_pg(self):
        conn_str = (
            f"host={_POSTGRES_ENV['host']} "
            f"port={_POSTGRES_ENV['port']} "
            f"dbname={_POSTGRES_ENV['dbname']} "
            f"user={_POSTGRES_ENV['user']} "
            f"password={_POSTGRES_ENV['password']}"
        )
        self.pg_conn = psycopg2.connect(conn_str)
        self.pg_conn.autocommit = True
        with self.pg_conn.cursor() as cur:
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS cv_cache (
                    hash TEXT PRIMARY KEY,
                    email TEXT,
                    phone TEXT,
                    data JSONB,
                    embedding BYTEA,
                    source_path TEXT,
                    processed_at TIMESTAMPTZ DEFAULT now()
                );
                """
            )

    def _pg_get(self, hash_val: str) -> Optional[Dict[str, Any]]:
        with self.pg_conn.cursor() as cur:
            cur.execute("SELECT data FROM cv_cache WHERE hash = %s;", (hash_val,))
            row = cur.fetchone()
            return row[0] if row else None

    def _pg_insert(self, entry: Dict[str, Any]):
        with self.pg_conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO cv_cache (hash, email, phone, data, embedding, source_path)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (hash) DO UPDATE SET
                    email = EXCLUDED.email,
                    phone = EXCLUDED.phone,
                    data = EXCLUDED.data,
                    embedding = EXCLUDED.embedding,
                    source_path = EXCLUDED.source_path,
                    processed_at = now();
                """,
                (
                    entry.get("hash"),
                    entry.get("email"),
                    entry.get("phone"),
                    Json(entry.get("data")),
                    entry.get("embedding"),
                    entry.get("source_path"),
                ),
            )

    def _pg_all_hashes(self) -> Set[str]:
        with self.pg_conn.cursor() as cur:
            cur.execute("SELECT hash FROM cv_cache;")
            return {row[0] for row in cur.fetchall()}

    def _pg_all_entries(self) -> List[Dict[str, Any]]:
        with self.pg_conn.cursor() as cur:
            cur.execute("SELECT hash, email, phone, data, embedding FROM cv_cache;")
            rows = cur.fetchall()
            result = []
            for h, e, p, d, emb in rows:
                result.append({"hash": h, "email": e, "phone": p, "data": d, "embedding": emb})
            return result

    # -----------------------------------------------------------------
    # JSON fallback implementation
    # -----------------------------------------------------------------
    def _load_json(self):
        os.makedirs(os.path.dirname(self.json_path) or ".", exist_ok=True)
        if os.path.exists(self.json_path):
            with open(self.json_path, "r", encoding="utf-8") as f:
                try:
                    self._json_cache: List[Dict[str, Any]] = json.load(f)
                except json.JSONDecodeError:
                    self._json_cache = []
        else:
            self._json_cache = []
        # fast lookup index
        self._hash_index: Dict[str, Dict[str, Any]] = {e["hash"]: e for e in self._json_cache}

    def _json_save(self):
        with open(self.json_path, "w", encoding="utf-8") as f:
            json.dump(self._json_cache, f, ensure_ascii=False, indent=2)

    def _json_get(self, hash_val: str) -> Optional[Dict[str, Any]]:
        return self._hash_index.get(hash_val)

    def _json_insert(self, entry: Dict[str, Any]):
        existing = self._hash_index.get(entry["hash"])
        if existing:
            self._json_cache.remove(existing)
        self._json_cache.append(entry)
        self._hash_index[entry["hash"]] = entry
        self._json_save()

    def _json_all_hashes(self) -> Set[str]:
        return set(self._hash_index.keys())

    def _json_all_entries(self) -> List[Dict[str, Any]]:
        return self._json_cache

    # -----------------------------------------------------------------
    # Public API – delegates to the chosen backend
    # -----------------------------------------------------------------
    def get_by_hash(self, hash_val: str) -> Optional[Dict[str, Any]]:
        return self._pg_get(hash_val) if _USE_POSTGRES else self._json_get(hash_val)

    def insert(self, entry: Dict[str, Any]):
        if _USE_POSTGRES:
            self._pg_insert(entry)
        else:
            self._json_insert(entry)

    def all_hashes(self) -> Set[str]:
        return self._pg_all_hashes() if _USE_POSTGRES else self._json_all_hashes()

    def all_entries(self) -> List[Dict[str, Any]]:
        return self._pg_all_entries() if _USE_POSTGRES else self._json_all_entries()

    def delete_by_hash(self, hash_val: str) -> bool:
        """Remove a cached entry by its file hash."""
        if _USE_POSTGRES:
            with self.pg_conn.cursor() as cur:
                cur.execute("DELETE FROM cv_cache WHERE hash = %s;", (hash_val,))
                return cur.rowcount > 0
        entry = self._hash_index.get(hash_val)
        if not entry:
            return False
        self._json_cache.remove(entry)
        del self._hash_index[hash_val]
        self._json_save()
        return True

    def delete_by_email(self, email: str) -> bool:
        """Remove cached entries matching an email (case-insensitive)."""
        email = (email or "").strip().lower()
        if not email:
            return False
        if _USE_POSTGRES:
            with self.pg_conn.cursor() as cur:
                cur.execute("DELETE FROM cv_cache WHERE lower(email) = %s;", (email,))
                return cur.rowcount > 0
        to_remove = [
            e for e in self._json_cache
            if (e.get("email") or "").strip().lower() == email
        ]
        if not to_remove:
            return False
        for entry in to_remove:
            self._json_cache.remove(entry)
            self._hash_index.pop(entry["hash"], None)
        self._json_save()
        return True

    def delete_by_filename(self, filename: str) -> bool:
        """Remove cached entries whose source_path or filename ends with *filename*."""
        filename = os.path.basename((filename or "").replace("\\", "/"))
        if not filename:
            return False
        if _USE_POSTGRES:
            with self.pg_conn.cursor() as cur:
                cur.execute(
                    "DELETE FROM cv_cache WHERE source_path LIKE %s;",
                    (f"%{filename}",),
                )
                return cur.rowcount > 0
        to_remove = []
        for entry in self._json_cache:
            source_path = (entry.get("source_path") or "").replace("\\", "/")
            if source_path.endswith(filename):
                to_remove.append(entry)
        if not to_remove:
            return False
        for entry in to_remove:
            self._json_cache.remove(entry)
            self._hash_index.pop(entry["hash"], None)
        self._json_save()
        return True

    def delete_matching(
        self,
        *,
        doc_hash: str = "",
        email: str = "",
        nom: str = "",
        phone: str = "",
        filename: str = "",
    ) -> List[Dict[str, Any]]:
        """Remove all cache entries matching any identifier. Returns removed entries."""
        doc_hash = (doc_hash or "").strip()
        email = (email or "").strip().lower()
        nom = (nom or "").strip().lower()
        phone = (phone or "").strip()
        filename = os.path.basename((filename or "").replace("\\", "/"))

        if _USE_POSTGRES:
            clauses = []
            params: List[Any] = []
            if doc_hash:
                clauses.append("hash = %s")
                params.append(doc_hash)
            if email:
                clauses.append("lower(email) = %s")
                params.append(email)
            if phone:
                clauses.append("phone = %s")
                params.append(phone)
            if filename:
                clauses.append("source_path LIKE %s")
                params.append(f"%{filename}")
            if not clauses:
                return []
            where = " OR ".join(clauses)
            with self.pg_conn.cursor() as cur:
                cur.execute(
                    f"DELETE FROM cv_cache WHERE {where} RETURNING hash, email, phone, source_path;",
                    tuple(params),
                )
                rows = cur.fetchall()
            return [
                {"hash": h, "email": e, "phone": p, "source_path": sp}
                for h, e, p, sp in rows
            ]

        to_remove: List[Dict[str, Any]] = []
        seen_hashes: Set[str] = set()
        for entry in list(self._json_cache):
            entry_hash = entry.get("hash") or ""
            entry_email = (entry.get("email") or "").strip().lower()
            entry_phone = (entry.get("phone") or "").strip()
            entry_nom = ((entry.get("data") or {}).get("nom") or "").strip().lower()
            source_path = (entry.get("source_path") or "").replace("\\", "/")
            entry_filename = os.path.basename(source_path)

            matched = False
            if doc_hash and entry_hash == doc_hash:
                matched = True
            elif email and entry_email == email:
                matched = True
            elif nom and entry_nom == nom:
                matched = True
            elif phone and entry_phone == phone:
                matched = True
            elif filename and (
                entry_filename == filename or source_path.endswith(filename)
            ):
                matched = True

            if matched and entry_hash not in seen_hashes:
                seen_hashes.add(entry_hash)
                to_remove.append(entry)

        if not to_remove:
            return []

        for entry in to_remove:
            self._json_cache.remove(entry)
            self._hash_index.pop(entry.get("hash"), None)
        self._json_save()
        return to_remove

    def prune(self, existing_hashes: Set[str]):
        """Delete cached rows whose hash is *not* in ``existing_hashes``.
        For JSON fallback the file is rewritten.
        """
        if _USE_POSTGRES:
            with self.pg_conn.cursor() as cur:
                cur.execute(
                    "DELETE FROM cv_cache WHERE hash NOT IN %s;",
                    (tuple(existing_hashes) or ("",),)
                )
        else:
            self._json_cache = [e for e in self._json_cache if e["hash"] in existing_hashes]
            self._hash_index = {e["hash"]: e for e in self._json_cache}
            self._json_save()
