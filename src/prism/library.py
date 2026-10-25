"""The library: folders and web pages read into passages, searched for each message.

A source (a folder, or a list of URLs) is used through a context item (#name). When a message
names it (or it is marked always), the passages that best match the message are added to the
room, with the file and page they came from, and the best PDFs/images are attached for the
bots that can open files. Bots search further with `ASK @docs: <keywords>`.

Search is BM25 over words (no service needed). When an embedding model is set (Settings →
Readers), passages also get vectors and results blend both (reciprocal rank fusion).
Scanning is incremental (path, size, modified time) and runs in the background.
"""

import asyncio
import hashlib
import logging
import math
import os
import re
import time
from collections import Counter
from dataclasses import dataclass, field
from operator import mul
from pathlib import Path

import httpx

from . import enrich, readers
from .db import Store

log = logging.getLogger(__name__)

SKIP_DIRS = {".git", ".hg", ".svn", "node_modules", "__pycache__", ".venv", "venv", ".tox",
             ".idea", ".vscode", ".cache", "$recycle.bin", "system volume information"}
MAX_FILES = 20_000
MAX_FILE_BYTES = 200 * 1024 * 1024     # bigger files are listed by name only
MAX_READ_BYTES = 60 * 1024 * 1024      # read whole into memory for text extraction
MAX_PAGE_BYTES = 15 * 1024 * 1024      # web pages
RESCAN_AFTER_S = 300                   # a folder in use is checked for changes this often
PASSAGES = 8                           # per message
PER_FILE = 3
BUDGET = 7000                          # characters of passages per message
ATTACH = 3                             # files uploaded per message (to bots that open files)
ATTACH_MAX_BYTES = 30 * 1024 * 1024
STOP = set("""a an and are as at be but by can could did do does for from had has have how i if
in into is it its me my of on or our so than that the their them then there these they this to
too was we were what when where which who whom why will with would you your about also any all
just more most not no only other some such very can't don't please tell show find give""".split())


def words(text: str) -> list[str]:
    out = []
    for w in re.findall(r"\w+", text.lower()):
        if len(w) < 2 or w in STOP:
            continue
        if len(w) > 4 and w.endswith("s") and not w.endswith("ss"):
            w = w[:-1]  # plans ~ plan
        out.append(w)
    return out


@dataclass
class _Index:
    version: int
    ids: list[int] = field(default_factory=list)
    tfs: list[Counter] = field(default_factory=list)
    lens: list[int] = field(default_factory=list)
    df: Counter = field(default_factory=Counter)
    rows: dict[int, dict] = field(default_factory=dict)
    vecs: dict[int, object] = field(default_factory=dict)

    @property
    def avg(self) -> float:
        return (sum(self.lens) / len(self.lens)) if self.lens else 1.0


class Library:
    def __init__(self, store: Store, home: Path, creds=lambda: {}, readers_config=lambda: {}):
        self.store, self.home = store, home
        self.creds = creds                    # enabled credentials with secrets ({} if locked)
        self.readers_config = readers_config  # meta `readers`
        self.jobs: dict[int, dict] = {}       # source id -> scan progress
        self._tasks: dict[int, asyncio.Task] = {}
        self._cache: dict[int, _Index] = {}

    # --- sources ------------------------------------------------------------------------

    def add(self, chat_id: int | None, name: str, location: str, always: bool, options: dict) -> int:
        location = location.strip()
        urls = [u.strip() for u in location.splitlines() if u.strip()]
        if urls and all(re.match(r"https?://", u, re.I) for u in urls):
            kind, location = "web", "\n".join(dict.fromkeys(urls))
        else:
            folder = Path(os.path.expandvars(os.path.expanduser(location.strip('"'))))
            if not folder.is_dir():
                raise ValueError(f"No folder at {location} - type the full path, e.g. D:\\Research"
                                 if os.name == "nt" else f"No folder at {location} - type the full path")
            kind, location = "folder", str(folder.resolve())
        source_id = self.store.add_source(kind, location, options)
        self.store.add_context(chat_id, name, self.summary(source_id), name=name, always=always,
                               library=source_id)
        return source_id

    def summary(self, source_id: int) -> str:
        src = self.store.get_source(source_id)
        if not src:
            return ""
        c = self.store.library_counts(source_id)
        where = (f"Folder {src['location']}" if src["kind"] == "folder"
                 else f"{len(src['location'].splitlines())} web page(s)")
        if not src["scanned_at"] and not c["files"]:
            return f"{where} - reading…"
        parts = [f"{c['files']} files", f"{c['chunks']} passages"]
        if c["skipped"]:
            parts.append(f"{c['skipped']} not readable")
        if c["waiting"]:
            parts.append(f"{c['waiting']} found by name only (images/scans/audio need a reader)")
        if c["vecs"]:
            parts.append("meaning search on")
        return f"{where} - " + ", ".join(parts)

    def status(self, source_id: int) -> dict:
        src = self.store.get_source(source_id) or {}
        job = self.jobs.get(source_id)
        problems = self.store.library_problems(source_id) if src else []
        return {"id": source_id, "kind": src.get("kind"), "location": src.get("location"),
                "options": src.get("options", {}), "scanned_at": src.get("scanned_at"),
                "busy": bool(job and not job["done"]), "job": job,
                **self.store.library_counts(source_id), "problems": problems}

    # --- scanning -----------------------------------------------------------------------

    def scan(self, source_id: int, full: bool = False) -> bool:
        """Start reading the source in the background (only what changed, unless full)."""
        if source_id in self._tasks and not self._tasks[source_id].done():
            return False
        self.jobs[source_id] = {"stage": "listing", "done": False, "files": 0, "total": 0,
                                "current": "", "error": None, "started": time.time()}
        task = asyncio.create_task(self._scan(source_id, full))
        self._tasks[source_id] = task
        task.add_done_callback(lambda _: self._tasks.pop(source_id, None))
        return True

    def scan_if_stale(self, source_id: int) -> None:
        src = self.store.get_source(source_id)
        if src and (not src["scanned_at"] or src["kind"] == "folder"
                    and time.time() - src["scanned_at"] > RESCAN_AFTER_S):
            self.scan(source_id)

    async def wait(self, source_id: int) -> None:
        if task := self._tasks.get(source_id):
            await asyncio.shield(task)

    async def _scan(self, source_id: int, full: bool) -> None:
        job = self.jobs[source_id]
        try:
            src = self.store.get_source(source_id)
            if not src:
                return
            known = {f["path"]: f for f in self.store.library_files(source_id)}
            if src["kind"] == "folder":
                listing = await asyncio.to_thread(self._list_folder, Path(src["location"]))
                job["total"] = len(listing)
                job["stage"] = "reading"
                changed = 0
                for rel, path, size, mtime in listing:
                    old = known.get(rel)
                    if not full and old and old["size"] == size and old["mtime"] == mtime:
                        continue
                    job["current"] = rel
                    await asyncio.to_thread(self._read_file, source_id, rel, path, size, mtime)
                    job["files"] += 1
                    changed += 1
                gone = [f["id"] for p, f in known.items() if p not in {r for r, *_ in listing}]
                if gone:
                    self.store.delete_library_files(gone)
                if changed or gone:
                    self.store.bump_source(source_id)
            else:
                urls = src["location"].splitlines()
                job["total"], job["stage"] = len(urls), "fetching"
                for url in urls:
                    if url in known and not full and not (known[url]["note"] or "").startswith("could not"):
                        continue
                    job["current"] = url
                    await self._fetch(source_id, url)
                    job["files"] += 1
                gone = [f["id"] for p, f in known.items() if p not in urls]
                if gone:
                    self.store.delete_library_files(gone)
                self.store.bump_source(source_id)
            await self._enrich(source_id, src["options"], full, job)
            await self._embed(source_id, job)
        except Exception as e:
            log.exception("scanning library source %s failed", source_id)
            job["error"] = str(e) or type(e).__name__
        finally:
            if self.store.get_source(source_id):
                self.store.update_source(source_id, scanned_at=time.time())
                self.store.set_context_content(source_id, self.summary(source_id))
            job.update(done=True, stage="done", current="")

    def _list_folder(self, root: Path) -> list[tuple[str, str, int, float]]:
        out = []
        for d, dirs, files in os.walk(root):
            dirs[:] = sorted(x for x in dirs if x.lower() not in SKIP_DIRS and not x.startswith("."))
            for name in sorted(files):
                if name.startswith(("~$", ".")) or not readers.supported(name):
                    continue
                path = os.path.join(d, name)
                try:
                    st = os.stat(path)
                except OSError:
                    continue
                out.append((os.path.relpath(path, root).replace("\\", "/"), path, st.st_size, st.st_mtime))
                if len(out) >= MAX_FILES:
                    return out
        return out

    def _read_file(self, source_id: int, rel: str, path: str, size: int, mtime: float) -> None:
        kind = readers.kind_of(rel)
        if size > MAX_FILE_BYTES or (size > MAX_READ_BYTES and kind not in ("image", "audio", "video")):
            doc = readers.Doc(kind, note=f"too big to read ({size // 1024 // 1024} MB)")
        elif kind in ("image", "audio", "video"):
            doc = readers.read(rel, b"")  # nothing to read as text: found by name, a reader may add more
        else:
            try:
                data = Path(path).read_bytes()
            except OSError as e:
                doc = readers.Doc(kind, note=f"could not open: {e.strerror or e}")
            else:
                doc = readers.read(rel, data)
        self._put(source_id, rel, path, doc, size, mtime)

    def _put(self, source_id: int, rel: str, local: str | None, doc: readers.Doc,
             size: int | None, mtime: float | None) -> None:
        name = readers.display_name(rel)
        # Every file is findable by its name and folder, even with no text.
        label = f"[{rel}]" + (f" ({name})" if name != Path(rel).name else "")
        passages = readers.chunks(doc.pages) or [(None, "")]
        chunks = [(page, f"{label}\n{text}".strip()) for page, text in passages]
        self.store.put_library_file(source_id, rel, chunks, local=local, kind=doc.kind,
                                    mtime=mtime, size=size, note=doc.note, needs=doc.needs, enriched=None)

    async def _fetch(self, source_id: int, url: str) -> None:
        cache = self.home / "library-cache"
        cache.mkdir(parents=True, exist_ok=True)
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(20, read=60), follow_redirects=True,
                                         headers={"User-Agent": "Mozilla/5.0 PRISM/0.7 (library)"}) as http:
                async with http.stream("GET", url) as r:
                    if r.status_code != 200:
                        raise ValueError(f"the site answered {r.status_code}")
                    data = b""
                    async for block in r.aiter_bytes():
                        data += block
                        if len(data) > MAX_PAGE_BYTES:
                            raise ValueError("page too big")
                    ctype = r.headers.get("content-type", "").split(";")[0].strip().lower()
        except Exception as e:
            self._put(source_id, url, None, readers.Doc("web", note=f"could not fetch: {str(e)[:160]}"), None, None)
            return
        ext = {"application/pdf": "pdf", "text/html": "html", "application/xhtml+xml": "html",
               "text/plain": "txt", "text/markdown": "md", "application/json": "json", "text/csv": "csv",
               "image/png": "png", "image/jpeg": "jpg", "image/webp": "webp", "image/gif": "gif"}.get(ctype)
        if not ext:
            ext = readers.ext_of(url.split("?")[0]) if readers.supported(url.split("?")[0]) else "html"
        local = cache / f"{hashlib.sha1(url.encode()).hexdigest()[:16]}.{ext}"
        local.write_bytes(data)
        doc = await asyncio.to_thread(readers.read, f"page.{ext}", data)
        self._put(source_id, url, str(local), doc, len(data), time.time())

    # --- readers: images, scans, audio, video; vectors -----------------------------------

    async def _enrich(self, source_id: int, options: dict, full: bool, job: dict) -> None:
        wants = {"vision": options.get("vision"), "transcribe": options.get("transcribe")}
        if not any(wants.values()):
            return
        todo = [f for f in self.store.library_files(source_id)
                if f["needs"] and wants.get(f["needs"]) and f["local"]
                and (f["enriched"] is None or (full and f["enriched"].startswith("failed")))]
        if not todo:
            return
        creds = self._creds()
        found = await enrich.detect(self.home, creds)
        cfg = enrich.load_config(self.readers_config())
        chosen = {job_: enrich.pick(cfg[job_], found[job_]) for job_ in ("vision", "transcribe")}
        job.update(stage="reading images, scans and recordings", files=0, total=len(todo))
        for f in todo:
            reader = chosen[f["needs"]]
            if not reader:
                continue  # stays "found by name only"; the summary says why
            job["current"] = f["path"]
            try:
                chunks = await self._run_reader(f, reader, creds)
                self.store.set_enriched(f["id"], f"done by {reader}", chunks)
            except Exception as e:
                self.store.set_enriched(f["id"], f"failed ({reader}): {str(e)[:200] or type(e).__name__}")
            job["files"] += 1
            self.store.bump_source(source_id)

    async def _run_reader(self, f: dict, reader: str, creds: dict) -> list[tuple[int | None, str]]:
        label = f"[{f['path']}]"
        if f["needs"] == "transcribe":
            text = await enrich.transcribe(reader, self.home, f["local"], creds)
            return [(p, f"{label} (transcript)\n{t}") for p, t in readers.chunks([(None, text)])]
        if f["kind"] == "pdf":
            pages = await asyncio.to_thread(enrich.pdf_page_images, f["local"])
            out = []
            for no, jpeg in pages:
                text = await enrich.describe_image(reader, jpeg, creds)
                out += [(no, f"{label} (page {no}, read from the scan)\n{t}") for _, t in readers.chunks([(no, text)])]
            return out
        jpeg = await asyncio.to_thread(enrich.to_jpeg, Path(f["local"]).read_bytes())
        text = await enrich.describe_image(reader, jpeg, creds)
        return [(p, f"{label} (what the image shows)\n{t}") for p, t in readers.chunks([(None, text)])]

    def embedder(self) -> str | None:
        return enrich.load_config(self.readers_config()).get("embed") or None

    async def _embed(self, source_id: int, job: dict) -> None:
        spec = self.embedder()
        if not spec:
            return
        job.update(stage="meaning search", files=0, total=self.store.library_counts(source_id)["chunks"])
        while rows := self.store.chunks_without_vec(source_id, 32):
            try:
                vecs = await enrich.embed(spec, [r["text"][:2000] for r in rows])
            except Exception as e:
                job["error"] = f"meaning search stopped: {e}"
                return
            self.store.set_vecs([(r["id"], v) for r, v in zip(rows, vecs)])
            job["files"] += len(rows)
        self.store.bump_source(source_id)

    def _creds(self) -> dict:
        try:
            return self.creds() or {}
        except Exception:
            return {}

    # --- search -------------------------------------------------------------------------

    def _index(self, source_id: int) -> _Index:
        src = self.store.get_source(source_id)
        version = src["version"] if src else -1
        idx = self._cache.get(source_id)
        if idx and idx.version == version:
            return idx
        idx = _Index(version)
        for r in self.store.library_chunks(source_id):
            tf = Counter(words(r["text"]))
            idx.ids.append(r["id"])
            idx.tfs.append(tf)
            idx.lens.append(sum(tf.values()) or 1)
            idx.df.update(tf.keys())
            idx.rows[r["id"]] = {"file_id": r["file_id"], "page": r["page"], "text": r["text"],
                                 "path": r["path"], "local": r["local"], "kind": r["kind"]}
            if r["vec"]:
                idx.vecs[r["id"]] = enrich.unpack(r["vec"])
        self._cache[source_id] = idx
        return idx

    def search(self, source_ids: list[int], query: str, k: int = PASSAGES, per_file: int = PER_FILE,
               qvec=None) -> list[dict]:
        """Best passages for the query across the sources: BM25, blended with meaning search
        when there are vectors."""
        terms = list(dict.fromkeys(words(query)))
        ranked: list[tuple[float, int, int]] = []  # (score, source, chunk)
        dense: list[tuple[float, int, int]] = []
        for sid in source_ids:
            idx = self._index(sid)
            n = len(idx.ids)
            if not n:
                continue
            idf = {t: math.log(1 + (n - idx.df[t] + 0.5) / (idx.df[t] + 0.5)) for t in terms if idx.df[t]}
            avg = idx.avg
            for cid, tf, ln in zip(idx.ids, idx.tfs, idx.lens):
                s = sum(idf[t] * tf[t] * 2.2 / (tf[t] + 1.2 * (0.25 + 0.75 * ln / avg)) for t in idf if t in tf)
                if s > 0:
                    ranked.append((s, sid, cid))
            if qvec is not None and idx.vecs:
                for cid, v in idx.vecs.items():
                    if len(v) == len(qvec):
                        dense.append((sum(map(mul, v, qvec)), sid, cid))
        ranked.sort(reverse=True)
        dense.sort(reverse=True)
        score: Counter = Counter()
        for lst in (ranked[:200], dense[:200]):
            for rank, (_, sid, cid) in enumerate(lst):
                score[(sid, cid)] += 1 / (60 + rank)
        out, per = [], Counter()
        for (sid, cid), s in score.most_common():
            row = self._cache[sid].rows[cid]
            if per[(sid, row["file_id"])] >= per_file:
                continue
            per[(sid, row["file_id"])] += 1
            out.append({**row, "source": sid, "chunk": cid, "score": s})
            if len(out) >= k:
                break
        return out

    async def query_vec(self, query: str):
        spec = self.embedder()
        if not spec:
            return None
        try:
            return enrich.unpack((await asyncio.wait_for(enrich.embed(spec, [query]), 15))[0])
        except Exception:
            return None  # words only this time

    async def retrieve(self, source_ids: list[int], query: str, k: int = PASSAGES) -> list[dict]:
        qvec = await self.query_vec(query)
        return await asyncio.to_thread(self.search, source_ids, query, k, PER_FILE, qvec)


def passages_text(hits: list[dict], budget: int = BUDGET) -> str:
    """Passages as the bots see them: numbered, with where each came from."""
    out, used = [], 0
    for i, h in enumerate(hits, start=1):
        text = h["text"]
        if text.startswith("["):  # the label line is shown in the citation instead
            text = text.split("\n", 1)[1] if "\n" in text else ""
        where = h["path"] + (f", p.{h['page']}" if h["page"] else "")
        piece = f"[{i}] {where}\n{text.strip() or '(no text - found by its name)'}"
        if used + len(piece) > budget and out:
            break
        out.append(piece[:budget])
        used += len(piece)
    return "\n\n".join(out)


def attachable(hits: list[dict], already: set[str]) -> list[str]:
    """The best files to upload along (PDFs and images every chat site opens), not ones this
    chat was already given."""
    out = []
    for h in hits:
        local = h.get("local")
        if not local or local in already or local in out:
            continue
        if readers.ext_of(local) not in readers.UPLOADABLE:
            continue
        try:
            if os.path.getsize(local) > ATTACH_MAX_BYTES:
                continue
        except OSError:
            continue
        out.append(local)
        if len(out) >= ATTACH:
            break
    return out
