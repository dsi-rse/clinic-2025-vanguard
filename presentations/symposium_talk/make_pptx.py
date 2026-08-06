"""Build a PowerPoint copy of the talk: one slide image per slide, plus notes.

Each slide is a full-bleed image exported from slides.pdf, so the deck looks
exactly as it does in the browser -- the card grids, the maroon title field, and
the logo bar all survive, and nothing on the slide is editable. Use this when a
venue requires .pptx; `slides.html` remains the deck to present from.

Speaker notes are read out of the rendered deck through reveal's own API rather
than parsed from slides.qmd, because the title slide's notes exist only after
_title_notes.html injects them.

Requires python-pptx (pip install python-pptx) and poppler's pdftoppm, plus the
Chromium that regenerate_pdf.sh drives.

Run from presentations/symposium_talk with:
    micromamba run -n vanguard python make_pptx.py
"""

import itertools
import json
import os
import re
import shutil
import socket
import subprocess
import tempfile
import time
import urllib.request
from pathlib import Path

import websocket
from pptx import Presentation
from pptx.util import Inches

HERE = Path(__file__).parent
PDF = HERE / "slides.pdf"
HTML = HERE / "slides.html"
OUT_PPTX = HERE / "slides.pptx"

CHROME = Path(
    os.environ.get(
        "CHROME",
        "/home/t-9svena/.local/share/quarto/chromium/linux-869685/chrome-linux/chrome",
    )
)
# The deck is authored at 1280x720 CSS px; 2x gives a crisp image on a 4K projector.
RENDER_DPI = 192
SLIDE_WIDTH_IN = 13.333
SLIDE_HEIGHT_IN = 7.5


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def read_notes() -> list[str]:
    """Return each slide's speaker notes, in deck order, as plain text."""
    port = _free_port()
    profile = tempfile.mkdtemp(prefix="pptx_notes_chrome_")
    proc = subprocess.Popen(  # noqa: S603
        [
            str(CHROME),
            "--headless=new",
            "--disable-gpu",
            f"--user-data-dir={profile}",
            f"--remote-debugging-port={port}",
            "--remote-allow-origins=*",
            "about:blank",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        deadline = time.time() + 60
        page = None
        while time.time() < deadline and page is None:
            try:
                with urllib.request.urlopen(
                    f"http://127.0.0.1:{port}/json/list", timeout=2
                ) as r:
                    page = next(t for t in json.load(r) if t.get("type") == "page")
            except Exception:
                time.sleep(0.2)
        ws = websocket.create_connection(page["webSocketDebuggerUrl"], timeout=60)
        ids = itertools.count(1)

        def cmd(method: str, **params: object) -> dict:
            request_id = next(ids)
            ws.send(json.dumps({"id": request_id, "method": method, "params": params}))
            while True:
                msg = json.loads(ws.recv())
                if msg.get("id") == request_id:
                    if "error" in msg:
                        raise SystemExit(f"CDP error from {method}: {msg['error']}")
                    return msg["result"]

        cmd("Page.enable")
        cmd("Page.navigate", url=f"file://{HTML}")
        time.sleep(5.0)
        raw = cmd(
            "Runtime.evaluate",
            expression=(
                "JSON.stringify([...document.querySelectorAll('.reveal .slides section')]"
                ".map(s => Reveal.getSlideNotes(s) || ''))"
            ),
            returnByValue=True,
        )["result"]["value"]
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
        shutil.rmtree(profile, ignore_errors=True)

    notes = []
    for html in json.loads(raw):
        # Quarto appends its assistive-MathML stylesheet to the document's last
        # element, which is the final slide's notes aside. It is invisible in the
        # browser but would land in the notes pane as CSS text.
        html = re.sub(r"<(style|script)\b.*?</\1>", "", html, flags=re.S)
        paragraphs = re.findall(r"<p[^>]*>(.*?)</p>", html, re.S) or [html]
        text = "\n\n".join(re.sub(r"<[^>]+>", "", p).strip() for p in paragraphs)
        notes.append(re.sub(r"[ \t]+", " ", text).strip())
    return notes


def render_pages(workdir: Path) -> list[Path]:
    """Rasterize every page of slides.pdf and return the images in page order."""
    pdftoppm = shutil.which("pdftoppm")
    if pdftoppm is None:
        raise SystemExit("pdftoppm not found -- install poppler-utils")
    subprocess.run(  # noqa: S603
        [pdftoppm, "-r", str(RENDER_DPI), "-png", str(PDF), str(workdir / "slide")],
        check=True,
    )
    return sorted(workdir.glob("slide-*.png"))


def main() -> None:
    """Write slides.pptx: one full-bleed slide image per page, with its notes."""
    notes = read_notes()
    workdir = Path(tempfile.mkdtemp(prefix="pptx_pages_"))
    try:
        pages = render_pages(workdir)
        if len(pages) != len(notes):
            raise SystemExit(f"{len(pages)} PDF pages but {len(notes)} slides of notes")

        deck = Presentation()
        deck.slide_width = Inches(SLIDE_WIDTH_IN)
        deck.slide_height = Inches(SLIDE_HEIGHT_IN)
        blank = deck.slide_layouts[6]

        for page, note in zip(pages, notes):
            slide = deck.slides.add_slide(blank)
            slide.shapes.add_picture(
                str(page),
                0,
                0,
                width=Inches(SLIDE_WIDTH_IN),
                height=Inches(SLIDE_HEIGHT_IN),
            )
            slide.notes_slide.notes_text_frame.text = note

        deck.save(OUT_PPTX)
        print(f"wrote {OUT_PPTX}: {len(pages)} slides at {RENDER_DPI} dpi")
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


if __name__ == "__main__":
    main()
