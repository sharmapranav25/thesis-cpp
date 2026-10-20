"""Explicit generic font inventory with cmap checks; no silent default font."""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from fontTools.ttLib import TTFont
from PIL import ImageFont

from .data import sha256_file

WINDOWS_GENERIC = (
    "arial.ttf", "arialbd.ttf", "ariali.ttf", "arialbi.ttf", "arialn.ttf", "arialnb.ttf",
    "verdana.ttf", "verdanab.ttf", "verdanai.ttf", "times.ttf", "timesbd.ttf", "timesi.ttf",
    "georgia.ttf", "georgiab.ttf", "georgiai.ttf", "cour.ttf", "courbd.ttf", "couri.ttf",
    "trebuc.ttf", "trebucbd.ttf", "trebucit.ttf", "tahoma.ttf", "tahomabd.ttf", "impact.ttf",
    "calibri.ttf", "calibrib.ttf", "cambria.ttf", "consola.ttf", "consolab.ttf", "comic.ttf",
)


def discover_fonts(charset, font_dir=None):
    if font_dir:
        root = Path(font_dir).resolve()
        if not root.is_dir():
            raise ValueError(f"Font directory not found: {root}")
        candidates = sorted(p for p in root.rglob("*") if p.suffix.lower() in {".ttf", ".otf"})
    else:
        root = Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts"
        available = {p.name.lower(): p for p in root.glob("*")} if root.exists() else {}
        candidates = [available[name] for name in WINDOWS_GENERIC if name in available]
        for folder in ("/usr/share/fonts/truetype/dejavu", "/usr/share/fonts/truetype/liberation",
                       "/usr/share/fonts/truetype/liberation2"):
            candidates.extend(sorted(Path(folder).glob("*.ttf")))
    fonts, rejected = [], []
    for path in candidates:
        try:
            with TTFont(str(path), lazy=True) as font:
                cmap = font.getBestCmap() or {}
                missing = [c for c in charset if ord(c) not in cmap or cmap[ord(c)] == ".notdef"]
                if missing:
                    raise ValueError("Missing characters: " + "".join(missing))
            ImageFont.truetype(str(path), 24)
            fonts.append({"path": str(path.resolve()), "sha256": sha256_file(path),
                          "provenance": "user-supplied generic font directory" if font_dir
                          else "generic OS font allowlist"})
        except Exception as exc:
            rejected.append({"path": str(path), "reason": str(exc)})
    if not fonts:
        raise ValueError("No usable generic fonts cover the charset. Supply --font-dir with generic TTF/OTF files.")
    return fonts, rejected


@lru_cache(maxsize=128)
def get_font(path, size):
    return ImageFont.truetype(path, size)


def verify_fonts(fonts):
    for font in fonts:
        if not Path(font["path"]).is_file() or sha256_file(font["path"]) != font["sha256"]:
            raise ValueError(f"Font missing or changed since fitting: {font['path']}")
