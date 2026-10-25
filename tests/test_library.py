"""The library: reading files of every kind, searching them, and giving the room the passages."""

import asyncio
import io
import json
import mailbox
import threading
import time
import zipfile

import pytest
import uvicorn
from fastapi import FastAPI, Request

from prism import enrich, readers
from prism.db import Store
from prism.library import Library, attachable, passages_text, words


def docx(paragraphs: list[str]) -> bytes:
    body = "".join(f'<w:p><w:r><w:t>{p}</w:t></w:r></w:p>' for p in paragraphs)
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w") as z:
        z.writestr("word/document.xml", '<w:document xmlns:w="http://schemas.openxmlformats.org/'
                   f'wordprocessingml/2006/main"><w:body>{body}</w:body></w:document>')
    return out.getvalue()


def pptx(slides: list[str]) -> bytes:
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w") as z:
        for i, text in enumerate(slides, start=1):
            z.writestr(f"ppt/slides/slide{i}.xml", '<p:sld xmlns:p="p" xmlns:a="http://schemas.openxml'
                       f'formats.org/drawingml/2006/main"><a:p><a:r><a:t>{text}</a:t></a:r></a:p></p:sld>')
    return out.getvalue()


def xlsx(rows: list[list[str]]) -> bytes:
    ns = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
    shared = [v for r in rows for v in r]
    sheet = "".join("<row>" + "".join(f'<c t="s"><v>{shared.index(v)}</v></c>' for v in r) + "</row>" for r in rows)
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w") as z:
        z.writestr("xl/workbook.xml", f'<workbook xmlns="{ns}"><sheets><sheet name="Budget"/></sheets></workbook>')
        z.writestr("xl/sharedStrings.xml", f'<sst xmlns="{ns}">' + "".join(f"<si><t>{v}</t></si>" for v in shared) + "</sst>")
        z.writestr("xl/worksheets/sheet1.xml", f'<worksheet xmlns="{ns}"><sheetData>{sheet}</sheetData></worksheet>')
    return out.getvalue()


def pdf(pages: list[str]) -> bytes:
    """A minimal PDF with one line of text per page (Helvetica)."""
    objs = ["<< /Type /Catalog /Pages 2 0 R >>", None, "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"]
    kids = []
    for text in pages:
        stream = f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET".encode()
        objs.append(f"<< /Length {len(stream)} >>\nstream\n{stream.decode()}\nendstream")
        objs.append(f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents {len(objs)} 0 R"
                    " /Resources << /Font << /F1 3 0 R >> >> >>")
        kids.append(f"{len(objs)} 0 R")
    objs[1] = f"<< /Type /Pages /Kids [{' '.join(kids)}] /Count {len(kids)} >>"
    out, offsets = b"%PDF-1.4\n", []
    for i, o in enumerate(objs, start=1):
        offsets.append(len(out))
        out += f"{i} 0 obj\n{o}\nendobj\n".encode()
    xref = len(out)
    out += f"xref\n0 {len(objs) + 1}\n0000000000 65535 f \n".encode()
    out += "".join(f"{o:010d} 00000 n \n" for o in offsets).encode()
    out += f"trailer\n<< /Size {len(objs) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF".encode()
    return out


def png() -> bytes:
    from PIL import Image
    out = io.BytesIO()
    Image.new("RGB", (40, 30), "red").save(out, "PNG")
    return out.getvalue()


# --- readers -----------------------------------------------------------------------------------

def test_reads_office_pdf_mail_notebook_and_archives():
    assert "renewal clause" in readers.read("a.docx", docx(["Contract", "The renewal clause"])).pages[0][1]
    slides = readers.read("deck.pptx", pptx(["Roadmap Q3", "Hiring plan"])).pages
    assert slides == [(1, "Roadmap Q3"), (2, "Hiring plan")]
    sheet = readers.read("b.xlsx", xlsx([["item", "cost"], ["laptops", "4200"]])).pages[0][1]
    assert "Sheet: Budget" in sheet and "laptops | 4200" in sheet
    doc = readers.read("r.pdf", pdf(["Quarterly revenue grew strongly this year", "Churn fell to two percent overall"]))
    assert [p for p, _ in doc.pages] == [1, 2] and "Churn" in doc.pages[1][1] and not doc.needs
    mail = b"From: ana@x.com\nSubject: Launch date\nContent-Type: text/plain\n\nWe ship on 12 May."
    assert "Subject: Launch date" in readers.read("m.eml", mail).pages[0][1]
    nb = json.dumps({"cells": [{"cell_type": "markdown", "source": ["# Model"]},
                               {"cell_type": "code", "source": "fit(x)"}]}).encode()
    assert "fit(x)" in readers.read("n.ipynb", nb).pages[0][1]
    html = b"<html><head><title>Pricing</title><script>x=1</script></head><body><h1>Plans</h1><p>Pro is $20</p></body></html>"
    text = readers.read("p.html", html).pages[0][1]
    assert text.startswith("Pricing") and "Pro is $20" in text and "x=1" not in text

    z = io.BytesIO()  # a Notion export: ids in the names, markdown inside
    with zipfile.ZipFile(z, "w") as zf:
        zf.writestr("Export/Team wiki 0123456789abcdef0123456789abcdef.md", "# Onboarding\nAsk Priya for access")
        zf.writestr("Export/logo.png", png())
    doc = readers.read("export.zip", z.getvalue())
    joined = "\n".join(t for _, t in doc.pages)
    assert "[Export/Team wiki.md]" in joined and "Ask Priya" in joined and "[Export/logo.png]" in joined
    assert readers.read("photo.jpg", b"").needs == "vision"
    assert readers.read("talk.mp3", b"").needs == "transcribe"
    assert "eml" in readers.read("old.msg", b"x").note
    assert readers.read("bad.pdf", b"not a pdf").note.startswith("could not read")


def test_mbox_and_chunks():
    box = "From a@x Mon Jan 1 00:00:00 2024\nSubject: One\n\nfirst body\n\nFrom b@x Mon Jan 1 00:00:00 2024\nSubject: Two\n\nsecond body\n"
    pages = readers.read("mail.mbox", box.encode()).pages
    assert [p for p, _ in pages] == [1, 2] and "second body" in pages[1][1]
    long = " ".join(f"Sentence number {i} is here." for i in range(400))
    parts = readers.chunks([(3, long)])
    assert len(parts) > 5 and all(p == 3 and len(t) <= 1200 for p, t in parts)
    assert all(t.startswith("Sentence") or t.startswith("number") or t[0].isalnum() for _, t in parts)


# --- scanning and search -----------------------------------------------------------------------

@pytest.fixture
def lib(tmp_path):
    store = Store(tmp_path / "t.db")
    return Library(store, tmp_path / "home")


def make_folder(tmp_path):
    root = tmp_path / "Research"
    (root / "notes").mkdir(parents=True)
    (root / "node_modules").mkdir()
    (root / "notes" / "pricing.md").write_text("Our pricing: the Pro plan costs 20 dollars a month.")
    (root / "notes" / "team.txt").write_text("Priya leads design. Omar runs infrastructure and on-call.")
    (root / "report.pdf").write_bytes(pdf(["Revenue grew 40 percent in the third quarter", "Churn fell to two percent"]))
    (root / "contract.docx").write_bytes(docx(["Renewal happens every March"]))
    (root / "whiteboard.png").write_bytes(png())
    (root / "node_modules" / "junk.js").write_text("pricing pricing pricing")
    (root / "~$contract.docx").write_text("lock file")
    return root


async def test_scan_search_and_rescan(lib, tmp_path):
    root = make_folder(tmp_path)
    sid = lib.add(None, "research", str(root), False, {})
    lib.scan(sid)
    await lib.wait(sid)
    st = lib.status(sid)
    assert st["files"] == 5 and st["waiting"] == 1  # the image: by name only
    assert "5 files" in lib.store.context_for(1)[0]["content"]

    hits = lib.search([sid], "how much does the pro plan cost?")
    assert hits[0]["path"] == "notes/pricing.md"
    hits = lib.search([sid], "churn last quarter")
    assert hits[0]["path"] == "report.pdf" and hits[0]["page"] == 2
    assert lib.search([sid], "whiteboard")[0]["path"] == "whiteboard.png"  # found by its name
    text = passages_text(hits)
    assert text.startswith("[1] report.pdf, p.2\nChurn fell")
    assert attachable(hits, set()) == [str(root / "report.pdf")]
    assert attachable(hits, {str(root / "report.pdf")}) == []

    (root / "notes" / "team.txt").unlink()  # changes are picked up, and only those are read
    (root / "notes" / "plan.md").write_text("Launch is planned for 12 May in Lisbon.")
    lib.scan(sid)
    await lib.wait(sid)
    assert lib.jobs[sid]["files"] == 1
    assert lib.search([sid], "launch lisbon")[0]["path"] == "notes/plan.md"
    assert not any(h["path"] == "notes/team.txt" for h in lib.search([sid], "Priya design"))


def test_add_rejects_missing_folder_and_takes_urls(lib, tmp_path):
    with pytest.raises(ValueError, match="No folder"):
        lib.add(None, "x", str(tmp_path / "nope"), False, {})
    sid = lib.add(None, "web", "https://a.example/one\nhttps://a.example/two\nhttps://a.example/one", False, {})
    assert lib.store.get_source(sid)["kind"] == "web"
    assert lib.store.get_source(sid)["location"].splitlines() == ["https://a.example/one", "https://a.example/two"]


def test_words():
    assert words("The plans for Lisbon's offices") == ["plan", "lisbon", "office"]


# --- a fake Ollama and web site ----------------------------------------------------------------

fake = FastAPI()
SEEN: dict[str, list] = {"embed": [], "vision": []}
VOCAB = ["cat", "dog", "money", "price", "cost", "pet"]


def vec(text: str) -> list[float]:
    t = text.lower()
    # "price" and "cost" mean the same; "cat", "dog" and "pet" too
    return [float(any(w in t for w in ("cat", "dog", "pet", "kitten"))),
            float(any(w in t for w in ("price", "cost", "dollar", "money"))), 0.1]


@fake.get("/api/tags")
def tags():
    return {"models": [{"name": "nomic-embed-text:latest"}, {"name": "gemma3:4b"}, {"name": "llama3.2:3b"}]}


@fake.post("/api/show")
async def show(request: Request):
    model = (await request.json())["model"]
    return {"capabilities": {"nomic-embed-text:latest": ["embedding"], "gemma3:4b": ["completion", "vision"]}
            .get(model, ["completion"])}


@fake.post("/api/embed")
async def api_embed(request: Request):
    body = await request.json()
    SEEN["embed"].append(body["model"])
    return {"embeddings": [vec(t) for t in body["input"]]}


@fake.post("/api/chat")
async def chat(request: Request):
    body = await request.json()
    SEEN["vision"].append(bool(body["messages"][0].get("images")))
    return {"message": {"content": "Text: Q3 GOALS\nShows: a whiteboard with a sales chart."}}


@fake.get("/page")
def page():
    from fastapi.responses import HTMLResponse
    return HTMLResponse("<html><title>Refund policy</title><body><p>Refunds within 30 days.</p></body></html>")


@fake.get("/missing")
def missing():
    from fastapi.responses import PlainTextResponse
    return PlainTextResponse("no", status_code=404)


@pytest.fixture(scope="module")
def fake_server():
    config = uvicorn.Config(fake, host="127.0.0.1", port=11498, log_level="warning")
    server = uvicorn.Server(config)
    t = threading.Thread(target=server.run, daemon=True)
    t.start()
    for _ in range(100):
        if server.started:
            break
        time.sleep(0.05)
    yield "http://127.0.0.1:11498"
    server.should_exit = True
    t.join(5)


@pytest.fixture
def ollama(fake_server, monkeypatch):
    monkeypatch.setitem(enrich.RUNTIMES["ollama"], "url", fake_server)
    monkeypatch.setitem(enrich.RUNTIMES["lmstudio"], "url", "http://127.0.0.1:9")  # not running
    return fake_server


async def test_readers_found_and_images_read_and_meaning_search(tmp_path, ollama):
    found = await enrich.detect(tmp_path, {"claude": {"transport": "api", "secret": "k", "model": "m"}})
    assert [v["id"] for v in found["vision"]] == ["ollama:gemma3:4b", "api:claude"]
    assert [v["id"] for v in found["embed"]] == ["ollama:nomic-embed-text:latest"]
    assert found["transcribe"] == []  # no whisper here

    store = Store(tmp_path / "t.db")
    config = {"vision": "auto", "embed": "ollama:nomic-embed-text:latest"}
    lib = Library(store, tmp_path / "home", readers_config=lambda: json.dumps(config))
    root = tmp_path / "f"
    root.mkdir()
    (root / "board.png").write_bytes(png())
    (root / "a.txt").write_text("The kitten sleeps all day.")
    (root / "b.txt").write_text("Shipping is free over fifty euros.")
    sid = lib.add(None, "f", str(root), False, {"vision": True})
    lib.scan(sid)
    await lib.wait(sid)
    assert SEEN["vision"] == [True]
    assert lib.status(sid)["vecs"] == lib.status(sid)["chunks"] > 0
    assert lib.search([sid], "q3 goals")[0]["path"] == "board.png"  # words read from the image
    # No word in common: only meaning search finds it.
    assert lib.search([sid], "my pet")[0:1] == []
    hits = await lib.retrieve([sid], "my pet")
    assert hits[0]["path"] == "a.txt"


async def test_web_pages(tmp_path, ollama):
    lib = Library(Store(tmp_path / "t.db"), tmp_path / "home")
    sid = lib.add(None, "w", f"{ollama}/page\n{ollama}/missing", False, {})
    lib.scan(sid)
    await lib.wait(sid)
    hits = lib.search([sid], "refund days")
    assert hits[0]["path"] == f"{ollama}/page" and "Refunds within 30 days" in hits[0]["text"]
    assert lib.status(sid)["problems"][0]["note"].startswith("could not fetch: the site answered 404")


async def test_transcribe_with_whisper_cpp(tmp_path, lib, monkeypatch):
    """whisper.cpp found in PRISM's tools folder; a fake one writes the transcript."""
    import sys
    monkeypatch.setattr(enrich, "ffmpeg_bin", lambda home: None)  # an .mp3 goes in as it is
    home = tmp_path / "home"
    tools = home / "tools" / "whisper"
    tools.mkdir(parents=True)
    (tools / "ggml-base.bin").write_bytes(b"model")
    script = tools / "fake_whisper.py"
    script.write_text("import sys\nargs = sys.argv\nopen(args[args.index('-of') + 1] + '.txt', 'w')"
                      ".write('Welcome to the budget meeting.')\n")
    exe = tools / ("whisper-cli.bat" if sys.platform == "win32" else "whisper-cli")
    exe.write_text(f'#!/bin/sh\nexec "{sys.executable}" "{script}" "$@"\n')
    exe.chmod(0o755)
    assert enrich.whisper_bin(home) == str(exe)
    lib.readers_config = lambda: json.dumps({"transcribe": "auto"})
    root = tmp_path / "rec"
    root.mkdir()
    (root / "meeting.mp3").write_bytes(b"ID3fake")
    sid = lib.add(None, "rec", str(root), False, {"transcribe": True})
    lib.scan(sid)
    await lib.wait(sid)
    hit = lib.search([sid], "budget meeting")[0]
    assert hit["path"] == "meeting.mp3" and "Welcome to the budget meeting" in hit["text"]
