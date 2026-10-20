"""Read font payloads from official Debian packages without installing/executing them."""
import collections
import hashlib
import io
import re
import tarfile
from pathlib import PurePosixPath
from .common import read, write
from .fonts import fetch


def acquire(run):
    bundles = [('fonts-liberation', 'fonts-liberation_2.1.5-3_all.deb'),
               ('fonts-liberation', 'fonts-liberation_1.07.4-11_all.deb'),
               ('fonts-lato', 'fonts-lato_2.015-1_all.deb'),
               ('fonts-noto', 'fonts-noto-mono_20201225-6_all.deb')]
    files = []; receipts = []
    for directory, filename in bundles:
        url = 'https://deb.debian.org/debian/pool/main/f/' + directory + '/' + filename
        dest = run / 'fonts/debian' / filename
        try:
            dest.mkdir(parents=True, exist_ok=True)
            archive = dest / 'source.deb'
            data = archive.read_bytes() if archive.exists() else fetch(url)
            if data[:8] != b'!<arch>\n':
                raise ValueError('Invalid ar package')
            archive.write_bytes(data)
            digest = hashlib.sha256(data).hexdigest()
            pos = 8; payload = None
            while pos + 60 <= len(data):
                header = data[pos:pos+60]
                if header[58:60] != b'`\n':
                    raise ValueError('Invalid ar header')
                name = header[:16].decode().strip().rstrip('/')
                size = int(header[48:58]); pos += 60
                if name.startswith('data.tar.'):
                    payload = data[pos:pos+size]
                pos += size + size % 2
            if payload is None:
                raise ValueError('No data payload')
            extracted = []
            with tarfile.open(fileobj=io.BytesIO(payload), mode='r:*') as tar:
                for member in tar:
                    if not member.isfile() or not (member.name.lower().endswith(('.ttf', '.otf')) or
                                                  PurePosixPath(member.name).name == 'copyright'):
                        continue
                    path = dest / 'files' / member.name
                    if not path.resolve().is_relative_to((dest / 'files').resolve()):
                        raise ValueError('Unsafe package path')
                    content = tar.extractfile(member).read()
                    path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(content)
                    extracted.append(dict(path=str(path), sha256=hashlib.sha256(content).hexdigest(), member=member.name))
            licences = [f for f in extracted if f['member'].endswith('/copyright')]
            if not licences:
                raise ValueError('No packaged copyright/licence')
            files.extend(dict(**f, url=url, archive_sha256=digest, licences=licences) for f in extracted
                         if f['path'].lower().endswith(('.ttf', '.otf')))
            receipts.append(dict(url=url, sha256=digest, files=len(extracted), status='downloaded',
                                 provenance='Official Debian HTTPS package; retained content hash, not independent signed-index verification'))
        except Exception as exc:
            receipts.append(dict(url=url, status='failed', error=str(exc)))
    acquisition = read(run / 'fonts/acquisition.json'); resolved = 0
    for rec in acquisition['records']:
        if rec['status'] != 'unresolved':
            continue
        desired = re.sub(r'^(sys_|orig_|curated_)', '', PurePosixPath(rec['requested']).name)
        matches = [f for f in files if PurePosixPath(f['member']).name.lower() == desired.lower()]
        if matches:
            rec.update(matches[0]); rec.update(status='downloaded', match='debian_filename_match_unverified_historical_hash')
            resolved += 1
    acquisition['summary']['status_counts'] = dict(collections.Counter(r['status'] for r in acquisition['records']))
    write(run / 'fonts/debian_receipts.json', receipts)
    write(run / 'fonts/acquisition.json', acquisition)
    return dict(resolved=resolved, status_counts=acquisition['summary']['status_counts'], bundles=receipts)
