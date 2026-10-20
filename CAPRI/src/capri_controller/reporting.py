"""Readable run overview from saved evidence; no inference or rerendering."""
import html,json
from pathlib import Path
from . import runtime as U

def build(root):
    root=Path(root)
    body=['<!doctype html><meta charset="utf-8"><title>CAPRI run overview</title>',
        '<style>body{font:17px system-ui;max-width:1200px;margin:30px auto;padding:20px;background:#f4f6f8;color:#182033}img{width:100%}pre{white-space:pre-wrap;background:white;padding:16px}a{color:#145abd}</style>',
        '<h1>CAPRI run overview</h1><p>'+html.escape(root.name)+'</p>']
    for name,title in [('length_decision','Model-derived length'),('pilot_summary','Controller pilot'),('diversity','Parameter diversity'),('corpus_audit','Corpus audit')]:
        path=root/(name+'.json')
        if path.exists():body+=['<h2>'+title+'</h2>','<pre>'+html.escape(json.dumps(U.read(path),indent=2))+'</pre>']
    body+=['<p>Basic renderer usability is not strict per-character readability or solver accuracy. Parameter differences do not establish distinct perceptual families.</p>']
    if (root/'pilot.png').exists():body+=['<h2>Original, custom, DINO and mixed</h2><img src="pilot.png" alt="Original and three controller renders">']
    if (root/'report.html').exists():body+=['<p><a href="report.html">Full historical comparison report</a></p>']
    dest=root/'overview.html';dest.write_text('\n'.join(body),encoding='utf8');return dest
