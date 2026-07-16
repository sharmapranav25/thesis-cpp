"""Download PSSTRNet pretrained weights into <repo>/checkpoints/psstrnet/.

Upstream Google Drive link from
https://github.com/GuangtaoLyu/PSSTRNet README:
    https://drive.google.com/file/d/1YpxKbFQnzO2jcDd2nzEwsJWgmDq-S0Ty/view

Tries an unauthenticated direct-download URL first; falls back to printing
manual-download instructions if Google Drive returns the virus-scan
confirmation page (happens for files >100 MB).

Saves to <repo>/checkpoints/psstrnet/scut_syn.pth (the filename used by
methods/psstrnet.py and the upstream inference script).
"""

from __future__ import annotations

import re
import sys
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
CKPT_DIR = REPO_ROOT / "checkpoints" / "psstrnet"
CKPT_PATH = CKPT_DIR / "scut_syn.pth"
GDRIVE_FILE_ID = "1YpxKbFQnzO2jcDd2nzEwsJWgmDq-S0Ty"
GDRIVE_VIEW_URL = f"https://drive.google.com/file/d/{GDRIVE_FILE_ID}/view?usp=share_link"
GDRIVE_DOWNLOAD_URL = f"https://drive.google.com/uc?export=download&id={GDRIVE_FILE_ID}"
EXPECTED_MIN_MB = 5.0  # arbitrary low bar; real model is larger


def _is_html_response(path: Path) -> bool:
    """Google Drive returns an HTML virus-warning page for files >100 MB."""
    if not path.exists() or path.stat().st_size > 5_000_000:
        return False
    try:
        head = path.read_bytes()[:256].decode("utf-8", errors="ignore").lower()
    except Exception:
        return False
    return "<!doctype html" in head or "<html" in head or "google drive" in head


def _try_gdown_session(out_path: Path) -> bool:
    """Use a session that follows the confirm redirect — Google Drive's standard pattern."""
    import http.cookiejar
    cj = http.cookiejar.CookieJar()
    opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(cj))
    opener.addheaders = [("User-Agent", "Mozilla/5.0")]
    print(f"  [GET] {GDRIVE_DOWNLOAD_URL}", flush=True)
    resp = opener.open(GDRIVE_DOWNLOAD_URL)
    head = resp.read(8192)
    text = head.decode("utf-8", errors="ignore")
    if "<html" in text.lower() or "<!doctype" in text.lower():
        m = re.search(r'href="(/uc\?export=download[^"]*confirm=[^"]*)"', text)
        if m:
            confirm_url = "https://drive.google.com" + m.group(1).replace("&amp;", "&")
            print(f"  [confirm] {confirm_url}", flush=True)
            resp = opener.open(confirm_url)
            head = b""
        else:
            m2 = re.search(r"confirm=([0-9A-Za-z_-]+)", text)
            if m2:
                confirm_url = f"{GDRIVE_DOWNLOAD_URL}&confirm={m2.group(1)}"
                print(f"  [confirm] {confirm_url}", flush=True)
                resp = opener.open(confirm_url)
                head = b""
            else:
                return False
    out_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    last_pct = -1
    expected = int(resp.headers.get("Content-Length", 0))
    with out_path.open("wb") as f:
        if head:
            f.write(head)
            written += len(head)
        while True:
            buf = resp.read(1024 * 64)
            if not buf:
                break
            f.write(buf)
            written += len(buf)
            if expected:
                pct = int(written * 100 / expected)
                if pct != last_pct and pct % 5 == 0:
                    sys.stdout.write(f"\r    {pct}% ({written/1e6:.1f} MB)")
                    sys.stdout.flush()
                    last_pct = pct
    print(f"\n  [done] wrote {written/1e6:.1f} MB to {out_path}")
    return True


def main() -> int:
    if CKPT_PATH.exists() and CKPT_PATH.stat().st_size > EXPECTED_MIN_MB * 1e6 and not _is_html_response(CKPT_PATH):
        print(f"[skip] {CKPT_PATH} already present ({CKPT_PATH.stat().st_size/1e6:.1f} MB)")
        return 0
    print(f"[fetch] PSSTRNet weights -> {CKPT_PATH}")
    try:
        _try_gdown_session(CKPT_PATH)
    except Exception as e:
        print(f"  [error] {e}", file=sys.stderr)
    if _is_html_response(CKPT_PATH):
        try:
            CKPT_PATH.unlink()
        except OSError:
            pass
        print(
            "\nGoogle Drive returned an HTML virus-warning page (file >100 MB requires confirm).\n"
            "Manual workaround:\n"
            f"  1. Open {GDRIVE_VIEW_URL} in a browser, click Download, accept the confirmation.\n"
            f"  2. Save to: {CKPT_PATH}\n"
            "Or install gdown: '.venv\\Scripts\\pip install gdown' then run:\n"
            f"  python -m gdown {GDRIVE_FILE_ID} -O \"{CKPT_PATH}\"",
            file=sys.stderr,
        )
        return 1
    if not CKPT_PATH.exists() or CKPT_PATH.stat().st_size < EXPECTED_MIN_MB * 1e6:
        print(
            "\nDownload failed or file too small. Try the manual workaround above.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
