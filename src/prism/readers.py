"""Reading files into plain text for the library: one function per kind of file, all with what
ships with Python plus pypdf. What can't be read as text (images, scans, audio, video) is
indexed by its name and marked for a reader model (see enrich.py).

read(name, data) -> Doc: `pages` are (page number or None, text). A page number is the PDF
page, slide, sheet or message it came from, so a passage can be cited.
"""

import email
import email.policy
import io
import json
import mailbox
import re
import tempfile
import zipfile
from dataclasses import dataclass, field
from html.parser import HTMLParser
from pathlib import Path, PurePosixPath
from xml.etree import ElementTree

import pypdf

TEXT_EXT = set("""txt md markdown mdx rst log csv tsv json jsonl yaml yml toml xml ini cfg conf
env properties tex bib srt vtt org adoc py pyi js mjs cjs ts tsx jsx java kt kts scala c h cc
cpp hpp cs go rs rb php swift m mm r jl lua pl sh bash zsh ps1 bat cmd sql graphql proto css
scss sass less vue svelte dart ex exs erl hs clj fs vb gradle dockerfile makefile cmake""".split())
HTML_EXT = {"html", "htm", "xhtml"}
IMAGE_EXT = {"png", "jpg", "jpeg", "gif", "webp", "bmp", "tif", "tiff"}
AUDIO_EXT = {"mp3", "wav", "m4a", "flac", "ogg", "opus", "aac", "wma"}
VIDEO_EXT = {"mp4", "mov", "mkv", "webm", "avi", "m4v", "wmv"}
OFFICE_EXT = {"docx", "pptx", "xlsx", "odt", "odp", "ods", "epub"}
EMAIL_EXT = {"eml", "mbox"}
SUPPORTED = (TEXT_EXT | HTML_EXT | IMAGE_EXT | AUDIO_EXT | VIDEO_EXT | OFFICE_EXT | EMAIL_EXT
             | {"pdf", "ipynb", "rtf", "zip", "msg"})
UPLOADABLE = {"pdf"} | IMAGE_EXT - {"bmp", "tif", "tiff"}  # what every chat site accepts
MAX_TEXT = 2_000_000       # characters kept per file
MAX_ARCHIVE_MEMBER = 20 * 1024 * 1024
NOTION_ID = re.compile(r"\s[0-9a-f]{32}(?=\.\w+$|$)")  # Notion export names end in an id


@dataclass
class Doc:
    kind: str
    pages: list[tuple[int | None, str]] = field(default_factory=list)
    note: str | None = None   # why it was not (fully) read
    needs: str | None = None  # vision | transcribe: a reader model could read the rest


def ext_of(name: str) -> str:
    base = PurePosixPath(name.replace("\\", "/")).name.lower()
    if base in ("dockerfile", "makefile"):
        return base
    return base.rsplit(".", 1)[-1] if "." in base else ""


def display_name(name: str) -> str:
    """The file name as a person reads it (Notion's ids dropped)."""
    return NOTION_ID.sub("", PurePosixPath(name.replace("\\", "/")).name)


def supported(name: str) -> bool:
    return ext_of(name) in SUPPORTED


def kind_of(name: str) -> str:
    e = ext_of(name)
    for kind, exts in (("image", IMAGE_EXT), ("audio", AUDIO_EXT), ("video", VIDEO_EXT),
                       ("html", HTML_EXT), ("office", OFFICE_EXT), ("email", EMAIL_EXT)):
        if e in exts:
            return kind
    return e if e in ("pdf", "zip", "ipynb", "rtf", "msg") else "text"


def decode(data: bytes) -> str:
    for enc in ("utf-8-sig", "utf-16") if data[:2] in (b"\xff\xfe", b"\xfe\xff") else ("utf-8-sig",):
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            pass
    return data.decode("cp1252", errors="replace")


def read(name: str, data: bytes, depth: int = 0) -> Doc:
    kind, e = kind_of(name), ext_of(name)
    try:
        if kind == "text":
            return Doc("text", [(None, decode(data)[:MAX_TEXT])])
        if kind == "html":
            return Doc("html", [(None, html_text(decode(data)))])
        if kind == "pdf":
            return read_pdf(data)
        if kind == "office":
            return read_office(e, data)
        if kind == "email":
            return read_eml(data) if e == "eml" else read_mbox(data)
        if e == "ipynb":
            return read_notebook(data)
        if e == "rtf":
            return Doc("text", [(None, rtf_text(decode(data)))])
        if e == "zip":
            return read_zip(data, depth) if depth == 0 else Doc("zip", note="archive inside an archive")
        if e == "msg":
            return Doc("email", note="Outlook .msg can't be read - save the mail as .eml, or export the folder as .mbox")
        if kind == "image":
            return Doc("image", needs="vision")
        if kind in ("audio", "video"):
            return Doc(kind, needs="transcribe")
    except Exception as ex:  # a broken file is skipped, not the folder
        return Doc(kind, note=f"could not read: {str(ex)[:160] or type(ex).__name__}")
    return Doc(kind, note="not a supported kind of file")


# --- formats ---------------------------------------------------------------------------------

class _HTMLText(HTMLParser):
    SKIP = {"script", "style", "noscript", "svg", "template", "head"}
    BLOCK = {"p", "div", "br", "li", "tr", "h1", "h2", "h3", "h4", "h5", "h6", "section",
             "article", "pre", "blockquote", "table", "ul", "ol", "dt", "dd", "header", "footer"}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.out: list[str] = []
        self.skip = 0
        self.title = ""
        self._in_title = False

    def handle_starttag(self, tag, attrs):
        if tag == "title":
            self._in_title = True
        elif tag in self.SKIP:
            self.skip += 1
        elif tag in self.BLOCK:
            self.out.append("\n")
        if tag in ("h1", "h2", "h3"):
            self.out.append("## ")

    def handle_endtag(self, tag):
        if tag == "title":
            self._in_title = False
        elif tag in self.SKIP:
            self.skip = max(0, self.skip - 1)
        elif tag in self.BLOCK:
            self.out.append("\n")

    def handle_data(self, data):
        if self._in_title:
            self.title += data
        elif not self.skip:
            self.out.append(data)


def html_text(source: str) -> str:
    p = _HTMLText()
    p.feed(source)
    p.close()
    text = re.sub(r"[ \t\r\f\v]+", " ", "".join(p.out))
    text = re.sub(r"\n\s*\n\s*", "\n\n", text).strip()
    title = " ".join(p.title.split())
    return (f"{title}\n\n{text}" if title and not text.startswith(title) else text)[:MAX_TEXT]


def read_pdf(data: bytes) -> Doc:
    reader = pypdf.PdfReader(io.BytesIO(data))
    if reader.is_encrypted:
        try:
            reader.decrypt("")
        except Exception:
            return Doc("pdf", note="password-protected PDF")
    pages = []
    for i, page in enumerate(reader.pages, start=1):
        try:
            pages.append((i, page.extract_text() or ""))
        except Exception:
            pages.append((i, ""))
    chars = sum(len(t.strip()) for _, t in pages)
    blank = sum(len(t.strip()) < 20 for _, t in pages)
    if pages and blank * 2 > len(pages):  # mostly empty pages: scans (pictures of text)
        return Doc("pdf", [p for p in pages if p[1].strip()], needs="vision",
                   note=None if chars else "scanned PDF: no text layer")
    return Doc("pdf", pages)


W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
A = "{http://schemas.openxmlformats.org/drawingml/2006/main}"
S = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"


def _xml(z: zipfile.ZipFile, name: str) -> ElementTree.Element:
    return ElementTree.fromstring(z.read(name))


def _num(name: str) -> int:
    m = re.search(r"(\d+)\.xml$", name)
    return int(m.group(1)) if m else 0


def read_office(e: str, data: bytes) -> Doc:
    z = zipfile.ZipFile(io.BytesIO(data))
    names = z.namelist()
    if e == "docx":
        paras = []
        for part in ["word/document.xml"] + sorted(n for n in names if re.match(r"word/(footnotes|endnotes)\.xml", n)):
            if part in names:
                for p in _xml(z, part).iter(W + "p"):
                    line = "".join(t.text or "" for t in p.iter(W + "t"))
                    paras.append(line)
        return Doc("office", [(None, "\n".join(paras).strip())])
    if e == "pptx":
        pages = []
        slides = sorted((n for n in names if re.match(r"ppt/slides/slide\d+\.xml$", n)), key=_num)
        for n in slides:
            no = _num(n)
            text = "\n".join("".join(t.text or "" for t in p.iter(A + "t")) for p in _xml(z, n).iter(A + "p"))
            notes = f"ppt/notesSlides/notesSlide{no}.xml"
            if notes in names:
                extra = " ".join(t.text or "" for t in _xml(z, notes).iter(A + "t")).strip()
                text += f"\n(Notes: {extra})" if extra else ""
            pages.append((no, text.strip()))
        return Doc("office", pages)
    if e == "xlsx":
        shared = []
        if "xl/sharedStrings.xml" in names:
            shared = ["".join(t.text or "" for t in si.iter(S + "t")) for si in _xml(z, "xl/sharedStrings.xml").iter(S + "si")]
        titles = []
        if "xl/workbook.xml" in names:
            titles = [s.get("name") for s in _xml(z, "xl/workbook.xml").iter(S + "sheet")]
        pages = []
        sheets = sorted((n for n in names if re.match(r"xl/worksheets/sheet\d+\.xml$", n)), key=_num)
        for n in sheets:
            rows = []
            for row in _xml(z, n).iter(S + "row"):
                cells = []
                for c in row.iter(S + "c"):
                    v = c.find(S + "v")
                    if c.get("t") == "s" and v is not None:
                        cells.append(shared[int(v.text)] if int(v.text) < len(shared) else "")
                    elif c.get("t") == "inlineStr":
                        cells.append("".join(t.text or "" for t in c.iter(S + "t")))
                    elif v is not None:
                        cells.append(v.text or "")
                if any(cells):
                    rows.append(" | ".join(cells))
                if len(rows) >= 20_000:
                    break
            no = _num(n)
            title = titles[no - 1] if 0 < no <= len(titles) else f"Sheet {no}"
            pages.append((no, f"Sheet: {title}\n" + "\n".join(rows)))
        return Doc("office", pages)
    if e in ("odt", "odp", "ods"):
        root = _xml(z, "content.xml")
        text_ns = "{urn:oasis:names:tc:opendocument:xmlns:text:1.0}"
        lines = ["".join(el.itertext()) for el in root.iter() if el.tag in (text_ns + "p", text_ns + "h")]
        return Doc("office", [(None, "\n".join(lines).strip())])
    if e == "epub":
        parts = sorted(n for n in names if n.lower().endswith((".xhtml", ".html", ".htm")))
        return Doc("office", [(i, html_text(decode(z.read(n)))) for i, n in enumerate(parts, start=1)])
    return Doc("office", note="not a supported kind of file")


def _mail_text(msg: email.message.Message) -> str:
    head = "\n".join(f"{k}: {msg[k]}" for k in ("From", "To", "Date", "Subject") if msg[k])
    body, attached = "", []
    for part in msg.walk() if msg.is_multipart() else [msg]:
        if part.get_content_maintype() == "multipart":
            continue
        if part.get_filename():
            attached.append(part.get_filename())
            continue
        try:
            content = part.get_payload(decode=True) or b""
            text = content.decode(part.get_content_charset() or "utf-8", errors="replace")
        except (LookupError, AttributeError):
            continue
        if part.get_content_type() == "text/plain" and not body:
            body = text
        elif part.get_content_type() == "text/html" and not body:
            body = html_text(text)
    extra = f"\nAttachments: {', '.join(attached)}" if attached else ""
    return f"{head}{extra}\n\n{body.strip()}"


def read_eml(data: bytes) -> Doc:
    msg = email.message_from_bytes(data, policy=email.policy.compat32)
    return Doc("email", [(None, _mail_text(msg))])


def read_mbox(data: bytes) -> Doc:
    with tempfile.TemporaryDirectory() as d:  # mailbox reads files, not bytes
        path = Path(d) / "mail.mbox"
        path.write_bytes(data)
        box = mailbox.mbox(str(path), create=False)
        pages = [(i, _mail_text(m)) for i, m in enumerate(box, start=1)]
        box.close()
    return Doc("email", pages)


def read_notebook(data: bytes) -> Doc:
    nb = json.loads(decode(data))
    cells = []
    for c in nb.get("cells", []):
        src = c.get("source", "")
        src = "".join(src) if isinstance(src, list) else src
        cells.append(f"```\n{src}\n```" if c.get("cell_type") == "code" else src)
    return Doc("text", [(None, "\n\n".join(cells))])


def rtf_text(source: str) -> str:
    text = re.sub(r"\\'([0-9a-f]{2})", lambda m: bytes.fromhex(m.group(1)).decode("cp1252", "replace"), source)
    text = re.sub(r"\\(par|line)\b ?", "\n", text)
    text = re.sub(r"\{\\\*[^{}]*\}", "", text)
    text = re.sub(r"\\[a-z]+-?\d* ?", "", text)
    return re.sub(r"[{}]", "", text).strip()


def read_zip(data: bytes, depth: int) -> Doc:
    """An export (Notion, Google Takeout, a zipped folder): each file in it is read; pages
    are prefixed with the file's name inside the archive."""
    z = zipfile.ZipFile(io.BytesIO(data))
    pages, skipped = [], 0
    for info in z.infolist():
        if info.is_dir() or not supported(info.filename) or info.file_size > MAX_ARCHIVE_MEMBER:
            skipped += not info.is_dir()
            continue
        inner = read(info.filename, z.read(info), depth + 1)
        label = info.filename.replace("\\", "/")
        label = "/".join(NOTION_ID.sub("", p) for p in label.split("/"))
        if inner.pages:
            for page, text in inner.pages:
                if text.strip():
                    pages.append((None, f"[{label}{f' p.{page}' if page else ''}]\n{text}"))
        else:
            pages.append((None, f"[{label}]"))  # found by name at least
    return Doc("zip", pages, note=f"{skipped} files inside not read" if skipped else None)


# --- passages ----------------------------------------------------------------------------------

def chunks(pages: list[tuple[int | None, str]], size: int = 1200, overlap: int = 150) -> list[tuple[int | None, str]]:
    """Passages of about `size` characters, broken at paragraphs or sentences, overlapping a
    little so a sentence cut at the edge is whole in one of them."""
    out = []
    for page, text in pages:
        text = re.sub(r"\n{3,}", "\n\n", text.replace("\r\n", "\n")).strip()
        while text:
            if len(text) <= size:
                out.append((page, text))
                break
            cut = max(text.rfind("\n\n", 0, size), text.rfind("\n", 0, size), text.rfind(". ", 0, size))
            cut = cut + 1 if cut > size // 3 else size
            out.append((page, text[:cut].strip()))
            space = text.find(" ", cut - overlap)  # the next one starts a little back, at a word
            text = text[space + 1 if 0 <= space < cut else cut:]
    return [(p, t) for p, t in out if t]
