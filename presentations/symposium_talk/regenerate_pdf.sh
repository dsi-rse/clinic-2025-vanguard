#!/usr/bin/env bash
# Render slides.qmd and export slides.pdf.
#
# Quarto's bundled Chromium (v91) intermittently emits a broken 1-page PDF
# instead of the full deck -- no error, just a silent truncation -- so this
# retries the print-to-pdf step until pdfinfo confirms all 8 pages landed.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

QUARTO=/ess/home/home1/t-9svena/quarto-1.4.557/bin/quarto
CHROME=/home/t-9svena/.local/share/quarto/chromium/linux-869685/chrome-linux/chrome
EXPECTED_PAGES=8
MAX_ATTEMPTS=5

"$QUARTO" render slides.qmd

for attempt in $(seq 1 "$MAX_ATTEMPTS"); do
  "$CHROME" --headless --disable-gpu --no-sandbox \
    --print-to-pdf=slides.pdf --print-to-pdf-no-header \
    --run-all-compositor-stages-before-draw --virtual-time-budget=15000 \
    "file://$(pwd)/slides.html?print-pdf" >/dev/null 2>&1

  pages=$(pdfinfo slides.pdf 2>/dev/null | awk '/^Pages:/{print $2}')
  echo "attempt ${attempt}: ${pages:-0} pages"
  if [ "${pages:-0}" = "${EXPECTED_PAGES}" ]; then
    echo "wrote slides.pdf (${pages} pages)"
    exit 0
  fi
done

echo "failed to produce a ${EXPECTED_PAGES}-page PDF after ${MAX_ATTEMPTS} attempts" >&2
exit 1
