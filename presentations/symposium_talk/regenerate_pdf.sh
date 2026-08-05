#!/usr/bin/env bash
# Render slides.qmd and export slides.pdf.
#
# Quarto's bundled Chromium (v91) `--print-to-pdf --virtual-time-budget=...`
# path prints on a fixed timer instead of waiting for reveal's print-pdf JS to
# lay out all `.pdf-page` sections -- it was landing a truncated 1-page PDF on
# every attempt (not just intermittently, despite the old comment here). This
# drives the same Chromium headless binary over the DevTools Protocol instead,
# polling `document.querySelectorAll('.reveal .pdf-page').length` until it
# appears and stabilizes before printing, which reliably captures every slide.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

QUARTO=/ess/home/home1/t-9svena/quarto-1.4.557/bin/quarto
CHROME=/home/t-9svena/.local/share/quarto/chromium/linux-869685/chrome-linux/chrome
# One title slide (from the YAML title) plus one per level-2 (##) heading.
EXPECTED_PAGES=$(($(grep -c "^## " slides.qmd) + 1))

"$QUARTO" render slides.qmd

micromamba run -n vanguard python3 - "$CHROME" "$(pwd)/slides.html" "$(pwd)/slides.pdf" <<'PYEOF'
import base64
import itertools
import json
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request

import websocket

chrome, html_path, pdf_path = sys.argv[1:4]


def free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def wait_for_tabs(port, timeout_s):
    deadline = time.time() + timeout_s
    url = f"http://127.0.0.1:{port}/json/list"
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2) as response:
                return json.load(response)
        except Exception:
            time.sleep(0.2)
    raise SystemExit("Chrome debugger endpoint never came up")


timeout_s = 60.0
port = free_port()
profile = tempfile.mkdtemp(prefix="quarto_cdp_chrome_")
proc = subprocess.Popen(
    [chrome, "--headless=new", "--disable-gpu", f"--user-data-dir={profile}",
     f"--remote-debugging-port={port}", "--remote-allow-origins=*", "about:blank"],
    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
)

try:
    tabs = wait_for_tabs(port, timeout_s)
    page = next(tab for tab in tabs if tab.get("type") == "page")
    ws = websocket.create_connection(page["webSocketDebuggerUrl"], timeout=timeout_s)
    ids = itertools.count(1)

    def cmd(method, **params):
        request_id = next(ids)
        ws.send(json.dumps({"id": request_id, "method": method, "params": params}))
        while True:
            msg = json.loads(ws.recv())
            if msg.get("id") == request_id:
                if "error" in msg:
                    raise SystemExit(f"CDP error from {method}: {msg['error']}")
                return msg["result"]

    def evaluate(expr):
        return cmd("Runtime.evaluate", expression=expr, returnByValue=True)["result"].get("value")

    cmd("Page.enable")
    cmd("Page.navigate", url=f"file://{html_path}?print-pdf")

    deadline = time.time() + timeout_s
    pages = 0
    while time.time() < deadline:
        pages = evaluate(
            "document.readyState === 'complete' "
            "? document.querySelectorAll('.reveal .pdf-page').length : 0"
        ) or 0
        if pages > 0:
            stable = evaluate("document.querySelectorAll('.reveal .pdf-page').length")
            time.sleep(1.0)
            if evaluate("document.querySelectorAll('.reveal .pdf-page').length") == stable:
                break
        time.sleep(0.5)
    if pages == 0:
        raise SystemExit("reveal never produced .pdf-page elements")

    result = cmd("Page.printToPDF", printBackground=True, preferCSSPageSize=True,
                  displayHeaderFooter=False, transferMode="ReturnAsBase64")
    with open(pdf_path, "wb") as f:
        f.write(base64.b64decode(result["data"]))
    print(f"wrote {pdf_path}: {pages} reveal pdf-pages")
finally:
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
    shutil.rmtree(profile, ignore_errors=True)
PYEOF

pages=$(pdfinfo slides.pdf 2>/dev/null | awk '/^Pages:/{print $2}')
if [ "${pages:-0}" != "${EXPECTED_PAGES}" ]; then
  echo "expected ${EXPECTED_PAGES} pages, got ${pages:-0}" >&2
  exit 1
fi
echo "slides.pdf has ${pages} pages, matching ${EXPECTED_PAGES} expected"
