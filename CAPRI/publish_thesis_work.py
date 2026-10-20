"""Copy only the prepared overlay to the explicitly authorized synth directory."""
from pathlib import Path
import hashlib
import json
import shutil

HERE = Path(__file__).resolve().parent
STAGE = HERE / 'thesis_regeneration_work'
TARGET = Path(r'E:\pranav\Final sem\sourcefree_synth')

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    state_path = HERE / 'thesis_publish_state.json'
    state = json.loads(state_path.read_text(encoding='utf-8'))
    if TARGET.resolve() != Path(state['target']).resolve() or not TARGET.is_dir():
        raise RuntimeError('Unexpected destination')
    updates = []
    for p in STAGE.rglob('*'):
        rel = p.relative_to(STAGE)
        if not p.is_file() or set(rel.parts) & {'__pycache__', '.git'}:
            continue
        dest = TARGET / rel
        digest = sha(p)
        if dest.exists() and sha(dest) == digest:
            continue
        if dest.exists() and sha(dest) != state['known_existing'].get(rel.as_posix()):
            raise RuntimeError('Destination changed independently: ' + str(dest))
        updates.append((p, dest, rel.as_posix(), digest))
    for p, dest, rel, digest in updates:
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(p, dest)
        if sha(dest) != digest:
            raise RuntimeError('Copy verification failed: ' + str(dest))
        state['known_existing'][rel] = digest
    state_path.write_text(json.dumps(state, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'destination': str(TARGET), 'files_published': len(updates), 'deletions': 0, 'original_repo_modified': False}))

if __name__ == '__main__':
    main()
