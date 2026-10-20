import argparse
from pathlib import Path
from .prepare import prepare
from .core import read,write,digest,new_run,audit_annotations

def main():
    parser = argparse.ArgumentParser(description="Isolated feature-bank experiment")
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("prepare", help="Freeze 40 existing validation images per dataset")
    p.add_argument("--root", required=True)
    p.add_argument("--out", required=True)
    p = commands.add_parser("audit-freeze", help="Validate and freeze a separately completed user audit")
    p.add_argument("--selection",required=True)
    p.add_argument("--audit",required=True)
    p.add_argument("--out",required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.root, args.out)
    elif args.command == "audit-freeze":
        selection=Path(args.selection)
        manifest=read(selection/'manifest.json'); drafts=read(selection/'draft_annotations.json')
        if digest(selection/'manifest.json')!=drafts['manifest_sha256']:
            raise ValueError('Selection manifest changed after drafting')
        audited=read(args.audit)
        disagreements=audit_annotations(manifest,drafts,audited)
        out=new_run(args.out)
        write(out/'audited_annotations.json',audited)
        write(out/'disagreements.json',disagreements)
        write(out/'receipt.json',dict(manifest_sha256=digest(selection/'manifest.json'),
              draft_sha256=digest(selection/'draft_annotations.json'),
              user_audit_sha256=digest(args.audit),disagreements=len(disagreements),
              evaluation_only=True,training_use_forbidden=True))
        print('User audit frozen; real-image scoring still requires completed prior algorithm gates.')

if __name__ == "__main__":
    main()
