from pathlib import Path
import shutil

root = Path('E:/pranav/Final sem/sourcefree_synth')
server = root / 'scripts/serve_style_monitor.py'
source = server.read_text(encoding='utf8')
old = "path.startswith('runs/style_')"
new = "path.startswith('runs/style_') or (path.startswith(('feature_specialists_v2/runs/curves_v1/', 'feature_specialists_v2/dots/runs/dots_v1/')) and target.suffix in ('.json', '.log', '.md'))"
assert old in source
backup = server.with_name('serve_style_monitor.before_specialists.py')
if not backup.exists():
    shutil.copy2(server, backup)
if new not in source:
    server.write_text(source.replace(old, new), encoding='utf8')
print('Monitor allowlist extended to read-only JSON/log/Markdown from the two specialist runs.')
