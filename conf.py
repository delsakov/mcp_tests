#!/usr/bin/env python3
import os, sys, argparse, hashlib, json, html, re
from urllib.parse import urljoin, quote
from xml.etree import ElementTree as ET
from datetime import datetime
from pathlib import Path

# --- Config via ENV ---
CONF_BASE_URL = os.environ.get("CONF_BASE_URL", "").rstrip("/")
CONF_EMAIL = os.environ.get("CONF_EMAIL", "")
CONF_API_TOKEN = os.environ.get("CONF_API_TOKEN", "")

RAG_UPLOAD_URL = os.environ.get("RAG_UPLOAD_URL", "")
RAG_API_TOKEN  = os.environ.get("RAG_API_TOKEN", "")

# Namespaces present in Confluence storage
NS = {
    "ac": "http://atlassian.com/content",
    "ri": "http://atlassian.com/resource/identifier",
}

CSS = """
<style>
body { font-family: system-ui, -apple-system, Segoe UI, Roboto, Ubuntu, "Helvetica Neue", Arial, sans-serif; line-height: 1.55; padding: 1.5rem; max-width: 900px; margin: auto; }
h1,h2,h3,h4,h5,h6 { line-height: 1.25; }
code, pre { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, "Liberation Mono", monospace; }
pre { background: #f6f8fa; padding: 0.75rem; border-radius: 6px; overflow: auto; }
table { border-collapse: collapse; width: 100%; }
th, td { border: 1px solid #ddd; padding: 6px; vertical-align: top; }
.admonition { border-left: 4px solid #999; padding: 0.5rem 0.75rem; background: #fafafa; margin: 1rem 0; }
.admonition.info { border-color: #2b7; }
.admonition.note { border-color: #27b; }
.admonition.warning { border-color: #d80; background: #fff8e6; }
details { background: #f7f7f9; padding: 0.5rem 0.75rem; border-radius: 6px; margin: 1rem 0; }
details > summary { cursor: pointer; font-weight: 600; }
.meta { color: #666; font-size: 0.9em; }
hr { border: 0; border-top: 1px solid #eee; margin: 1rem 0; }
</style>
"""

def sha256(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()

# --- HTTP helpers: requests (preferred) or urllib fallback ---
def use_requests():
    return "--no-requests" not in sys.argv

if use_requests():
    try:
        import requests
    except Exception:
        print("requests not available; rerun with --no-requests to use urllib fallback", file=sys.stderr)
        sys.exit(1)

def http_get_json(url, headers=None, params=None):
    if use_requests():
        import requests
        r = requests.get(url, headers=headers or {}, params=params or {}, timeout=60, auth=(CONF_EMAIL, CONF_API_TOKEN))
        r.raise_for_status()
        return r.json()
    else:
        import urllib.request, urllib.parse, base64
        if params:
            url = url + "?" + urllib.parse.urlencode(params)
        req = urllib.request.Request(url)
        if headers:
            for k, v in headers.items():
                req.add_header(k, v)
        # basic auth
        token = ("%s:%s" % (CONF_EMAIL, CONF_API_TOKEN)).encode("utf-8")
        req.add_header("Authorization", "Basic " + base64.b64encode(token).decode("ascii"))
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = resp.read()
            return json.loads(data.decode("utf-8"))

def http_post_file(url, headers=None, files=None, data=None):
    if not url:
        return {"ok": False, "error": "No RAG_UPLOAD_URL provided"}
    if use_requests():
        import requests
        hdrs = headers or {}
        resp = requests.post(url, headers=hdrs, files=files, data=data, timeout=120)
        try:
            j = resp.json()
        except Exception:
            j = {"status_code": resp.status_code, "text": resp.text[:500]}
        return j
    else:
        # Minimal multipart with urllib is cumbersome; for no-requests environments,
        # many RAG APIs accept raw JSON or octet-stream. Adjust here if needed.
        import urllib.request, mimetypes, uuid
        boundary = "----WebKitFormBoundary" + uuid.uuid4().hex
        body = []
        if data:
            for k, v in data.items():
                body.append(f"--{boundary}\r\nContent-Disposition: form-data; name=\"{k}\"\r\n\r\n{v}\r\n")
        if files:
            for k, (fname, bytes_or_fp, mime) in files.items():
                if hasattr(bytes_or_fp, "read"):
                    content = bytes_or_fp.read()
                elif isinstance(bytes_or_fp, (bytes, bytearray)):
                    content = bytes_or_fp
                else:
                    content = str(bytes_or_fp).encode("utf-8")
                mime = mime or mimetypes.guess_type(fname)[0] or "application/octet-stream"
                body.append(
                    f"--{boundary}\r\nContent-Disposition: form-data; name=\"{k}\"; filename=\"{fname}\"\r\nContent-Type: {mime}\r\n\r\n".encode("utf-8")
                )
                if isinstance(content, (bytes, bytearray)):
                    body.append(content)
                else:
                    body.append(content.encode("utf-8"))
                body.append(b"\r\n")
        body.append(f"--{boundary}--\r\n".encode("utf-8"))
        body_bytes = b"".join([b if isinstance(b, (bytes, bytearray)) else b.encode("utf-8") for b in body])
        req = urllib.request.Request(url, data=body_bytes, method="POST")
        req.add_header("Content-Type", f"multipart/form-data; boundary={boundary}")
        if headers:
            for k, v in headers.items():
                req.add_header(k, v)
        with urllib.request.urlopen(req, timeout=120) as resp:
            txt = resp.read().decode("utf-8", errors="replace")
            try:
                return json.loads(txt)
            except Exception:
                return {"status_code": resp.status, "text": txt[:500]}

# --- Confluence REST helpers ---
def conf_view_url(page_id: str) -> str:
    return f"{CONF_BASE_URL}/pages/viewpage.action?pageId={page_id}"

def conf_download_url(page_id: str, filename: str) -> str:
    return f"{CONF_BASE_URL}/download/attachments/{page_id}/{quote(filename)}"

def fetch_page(page_id: str):
    if not CONF_BASE_URL or not CONF_EMAIL or not CONF_API_TOKEN:
        raise RuntimeError("Set CONF_BASE_URL, CONF_EMAIL, CONF_API_TOKEN")
    path = f"{CONF_BASE_URL}/rest/api/content/{page_id}"
    params = {"expand": "body.storage,version,ancestors,space,metadata.labels"}
    data = http_get_json(path, params=params, headers={"Accept": "application/json"})
    return data

# --- Storage (XHTML) -> clean HTML ---
def txt(node):
    return "".join(node.itertext()) if node is not None else ""

def escape(s):
    return html.escape(s or "", quote=True)

def render_children(node, page_id):
    return "".join(render(n, page_id) for n in list(node) if isinstance(n.tag, str))

def get_ac_name(node):
    return node.attrib.get(f"{{{NS['ac']}}}name", "").lower()

def find_one(node, tag, ns=None):
    if ns:
        return node.find(f".//{{{ns}}}{tag}")
    return node.find(tag)

def render_macro_code(node):
    lang_el = node.find(f".//{{{NS['ac']}}}parameter[@{{{NS['ac']}}}name='language']")
    code_el = node.find(f".//{{{NS['ac']}}}plain-text-body")
    lang = (txt(lang_el) or "").strip()
    code = txt(code_el)
    return f'<pre><code class="language-{escape(lang)}">{escape(code)}</code></pre>'

def render_macro_expand(node, page_id):
    title_el = node.find(f".//{{{NS['ac']}}}parameter[@{{{NS['ac']}}}name='title']")
    body_el  = node.find(f".//{{{NS['ac']}}}rich-text-body")
    title = (txt(title_el) or "Details").strip()
    inner = render_children(body_el, page_id) if body_el is not None else ""
    return f"<details><summary>{escape(title)}</summary>{inner}</details>"

def render_macro_panel(node, page_id, kind="info"):
    title_el = node.find(f".//{{{NS['ac']}}}parameter[@{{{NS['ac']}}}name='title']")
    body_el  = node.find(f".//{{{NS['ac']}}}rich-text-body")
    title = (txt(title_el) or kind.title()).strip()
    inner = render_children(body_el, page_id) if body_el is not None else ""
    cls = "admonition " + kind
    return f'<div class="{cls}"><div><strong>{escape(title)}:</strong></div>{inner}</div>'

def render_inline_attachment(node, page_id):
    att = node.find(f".//{{{NS['ri']}}}attachment")
    if att is None:
        return ""
    filename = att.attrib.get(f"{{{NS['ri']}}}filename", "attachment")
    url = conf_download_url(page_id, filename)
    # distinguish <ac:image> vs link container
    if node.tag.endswith("image"):
        alt = escape(filename)
        return f'<img src="{escape(url)}" alt="{alt}" />'
    # generic link
    return f'<a href="{escape(url)}">{escape(filename)}</a>'

def render_inline_page_link(node):
    page = node.find(f".//{{{NS['ri']}}}page")
    if page is None:
        return ""
    title = page.attrib.get(f"{{{NS['ri']}}}content-title", "Confluence Page")
    # Without extra API lookups, we can link to search by title:
    href = f"{CONF_BASE_URL}/wiki/search?text={quote(title)}"
    return f'<a href="{escape(href)}">{escape(title)}</a>'

def render(node, page_id):
    tag = node.tag
    # Confluence macros
    if tag == f"{{{NS['ac']}}}structured-macro":
        name = get_ac_name(node)
        if name == "code":
            return render_macro_code(node)
        if name == "expand":
            return render_macro_expand(node, page_id)
        if name in ("info", "note", "panel", "warning"):
            kind = "warning" if name == "warning" else ("info" if name in ("info", "panel") else "note")
            return render_macro_panel(node, page_id, kind=kind)
        # Unknown macro: try to emit its rich-text-body content
        body_el = node.find(f".//{{{NS['ac']}}}rich-text-body")
        return render_children(body_el, page_id) if body_el is not None else ""

    # Images / explicit link wrappers
    if tag in (f"{{{NS['ac']}}}image", f"{{{NS['ac']}}}link"):
        # attachment?
        if node.find(f".//{{{NS['ri']}}}attachment") is not None:
            return render_inline_attachment(node, page_id)
        # page link?
        if node.find(f".//{{{NS['ri']}}}page") is not None:
            return render_inline_page_link(node)
        # fallback to inner text
        return escape(txt(node))

    # Standard XHTML-ish content: handle common tags
    short = tag.split("}")[-1]  # e.g., h1, p, ul, li, table...
    if short in ("h1","h2","h3","h4","h5","h6"):
        return f"<{short}>{escape(txt(node))}</{short}>"
    if short == "p":
        return f"<p>{escape(txt(node))}</p>"
    if short in ("ul","ol"):
        inner = "".join(render(li, page_id) for li in list(node))
        return f"<{short}>{inner}</{short}>"
    if short == "li":
        return f"<li>{escape(txt(node))}</li>"
    if short == "table":
        rows = "".join(render(tr, page_id) for tr in list(node))
        return f"<table>{rows}</table>"
    if short == "tr":
        cells = "".join(render(td, page_id) for td in list(node))
        return f"<tr>{cells}</tr>"
    if short in ("td","th"):
        return f"<{short}>{escape(txt(node))}</{short}>"
    if short == "a":
        href = node.attrib.get("href", "")
        text = txt(node) or href
        return f'<a href="{escape(href)}">{escape(text)}</a>'
    if short == "br":
        return "<br/>"
    if short == "hr":
        return "<hr/>"
    if short in ("em","strong","b","i","u","code","pre"):
        # keep basic inline semantics by wrapping text
        return f"<{short}>{escape(txt(node))}</{short}>"

    # Unknown/other tags: render text content
    return escape(txt(node))

def storage_xhtml_to_clean_html(storage_xml: str, page_id: str) -> str:
    # Confluence storage may omit a single <body>; parse and walk all children
    try:
        root = ET.fromstring(storage_xml)
    except ET.ParseError:
        # storage sometimes arrives as a fragment; wrap and retry
        storage_xml_wrapped = f"<root>{storage_xml}</root>"
        root = ET.fromstring(storage_xml_wrapped)

    # Find <body> if present, else use root
    body = root.find("body")
    container = body if body is not None else root

    parts = []
    for child in list(container):
        if isinstance(child.tag, str):
            parts.append(render(child, page_id))
    html_body = "".join(parts)

    # Normalize: collapse excessive blank tags (very light touch)
    html_body = re.sub(r"\n{3,}", "\n\n", html_body)

    return html_body

def build_html_document(meta: dict, body_html: str) -> str:
    head = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">{CSS}
<title>{html.escape(meta.get('title','Untitled'))}</title></head><body>
<div class="meta">
<!--
Title: {meta.get('title','')}
Space: {meta.get('space','')}
Labels: {", ".join(meta.get('labels', []))}
URL: {meta.get('url','')}
Last-Modified: {meta.get('modified','')}
Page-ID: {meta.get('id','')}
Version: {meta.get('version','')}
-->
<p><strong>{html.escape(meta.get('title',''))}</strong><br/>
<a href="{html.escape(meta.get('url',''))}">{html.escape(meta.get('url',''))}</a></p>
</div>
<hr/>
"""
    tail = "</body></html>"
    return head + body_html + tail

def save_html(out_dir: Path, space: str, title: str, page_id: str, html_text: str) -> Path:
    space_slug = re.sub(r"[^A-Za-z0-9._-]+", "-", space or "UNKNOWN").strip("-")
    title_slug = re.sub(r"[^A-Za-z0-9._-]+", "-", (title or "untitled")).strip("-")
    out_dir.mkdir(parents=True, exist_ok=True)
    fp = out_dir / f"{space_slug}__{title_slug}__{page_id}.html"
    fp.write_text(html_text, encoding="utf-8")
    return fp

def upload_to_rag(file_path: Path, page_meta: dict):
    if not RAG_UPLOAD_URL:
        return {"skipped": True, "reason": "RAG_UPLOAD_URL not set"}
    headers = {}
    if RAG_API_TOKEN:
        headers["Authorization"] = f"Bearer {RAG_API_TOKEN}"

    with open(file_path, "rb") as f:
        files = {
            "file": (file_path.name, f, "text/html"),
        }
        # Many RAG APIs allow extra metadata fields alongside the file:
        data = {
            "external_id": page_meta.get("id",""),
            "title": page_meta.get("title",""),
            "source_url": page_meta.get("url",""),
            "space": page_meta.get("space",""),
            "labels": ",".join(page_meta.get("labels", [])),
            "version": str(page_meta.get("version","")),
            "modified": page_meta.get("modified",""),
        }
        return http_post_file(RAG_UPLOAD_URL, headers=headers, files=files, data=data)

def process_id(page_id: str, out_dir: Path, upload: bool):
    page = fetch_page(page_id)
    storage = page.get("body",{}).get("storage",{}).get("value","") or ""
    title   = page.get("title","")
    space   = (page.get("space") or {}).get("key","")
    labels  = [l.get("name","") for l in (page.get("metadata",{}).get("labels",{}).get("results",[]) or [])]
    version = (page.get("version") or {}).get("number","")
    modified = (page.get("version") or {}).get("when","")
    url = conf_view_url(page_id)

    body_html = storage_xhtml_to_clean_html(storage, page_id)
    meta = {"id": page_id, "title": title, "space": space, "labels": labels,
            "version": version, "modified": modified, "url": url}
    doc = build_html_document(meta, body_html)
    fp = save_html(out_dir, space, title, page_id, doc)

    result = {"file": str(fp)}
    if upload:
        up = upload_to_rag(fp, meta)
        result["upload"] = up
    return result

def main():
    ap = argparse.ArgumentParser(description="Export Confluence pages (storage XHTML) to clean HTML and upload to RAG.")
    ap.add_argument("--out", required=True, help="Output directory for HTML files")
    ap.add_argument("--ids", nargs="*", help="List of Confluence pageIds")
    ap.add_argument("--ids-file", help="File path with one pageId per line")
    ap.add_argument("--upload", action="store_true", help="Upload each HTML to RAG via RAG_UPLOAD_URL")
    ap.add_argument("--no-requests", action="store_true", help="Force urllib fallback (no requests library)")
    args = ap.parse_args()

    if not CONF_BASE_URL or not CONF_EMAIL or not CONF_API_TOKEN:
        print("Please set CONF_BASE_URL, CONF_EMAIL, CONF_API_TOKEN env vars.", file=sys.stderr)
        sys.exit(2)

    ids = list(args.ids or [])
    if args.ids_file:
        with open(args.ids_file, "r", encoding="utf-8") as f:
            ids.extend([ln.strip() for ln in f if ln.strip()])
    ids = [i.strip() for i in ids if i and i.strip().isdigit()]

    if not ids:
        print("No pageIds provided.", file=sys.stderr)
        sys.exit(3)

    out_dir = Path(args.out)
    results = []
    for pid in ids:
        try:
            res = process_id(pid, out_dir, args.upload)
            print(json.dumps({"page_id": pid, **res}, ensure_ascii=False))
            results.append(res)
        except Exception as e:
            print(json.dumps({"page_id": pid, "error": str(e)}), file=sys.stderr)

if __name__ == "__main__":
    main()
