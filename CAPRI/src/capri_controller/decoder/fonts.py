"""Acquire pinned font assets, preserving exact versus family-only matches and provenance."""
import collections
import concurrent.futures
import hashlib
import json
import re
import string
import time
import urllib.parse
import urllib.request
from pathlib import Path, PurePosixPath
from .common import ROOT, read, write, sha, now


def norm(s):
    return re.sub('[^a-z0-9]', '', s.lower())


def fetch(url):
    for attempt in range(3):
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'CAPRI-Decoder-font-audit'})
            with urllib.request.urlopen(req, timeout=40) as response:
                return response.read()
        except Exception:
            if attempt == 2:
                raise
            time.sleep(attempt + 1)


def inventory():
    text = (ROOT / 'docs/HYBRID_FONT_CONTROLLER_REDESIGN_HANDOFF.md').read_text(encoding='utf-8')
    tail = text.split('Font Inventory (addons)', 1)[1]
    return [s.strip() for s in tail.splitlines() if s.strip().lower().endswith(('.ttf', '.otf', '.ttc'))]


def repository(run, repo, branch):
    dest = run / 'fonts/sources' / (repo.replace('/', '__') + '.json')
    if dest.exists():
        data = read(dest)
    else:
        url = 'https://api.github.com/repos/' + repo + '/git/trees/' + branch + '?recursive=1'
        data = json.loads(fetch(url))
        if data.get('truncated'):
            raise RuntimeError('Incomplete repository tree: ' + repo)
        write(dest, data)
    return data['sha'], {x['path']: x for x in data['tree'] if x['type'] == 'blob'}


def acquire(run, workers=8):
    repos = [('google/fonts', 'main'), ('josecl/cool-php-captcha', 'master'),
             ('dapphp/securimage', 'master'), ('source-foundry/Hack', 'master'), ('lepture/captcha', 'main'),
             ('notofonts/noto-fonts', 'main')]
    sources, failures = {}, {}
    for repo, branch in repos:
        try:
            sources[repo] = repository(run, repo, branch)
        except Exception as exc:
            failures[repo] = str(exc)
    candidates = []
    for repo, (revision, tree) in sources.items():
        for p in tree:
            if p.lower().endswith(('.ttf', '.otf')):
                candidates.append((repo, revision, p))
    records = []
    for requested in inventory():
        name = PurePosixPath(requested).name
        family = None
        desired = name
        if name.startswith('gf_'):
            _, family, desired = name.split('_', 2)
        elif name.startswith(('sys_', 'curated_', 'orig_', 'coolphpcaptcha_', 'securimage_', 'pycaptcha_')):
            desired = name.split('_', 1)[1]
        elif name.startswith('gh_hack_'):
            desired = name[len('gh_hack_'):]
        exact = [c for c in candidates if PurePosixPath(c[2]).name.lower() == desired.lower()
                 and (family is None or norm(PurePosixPath(c[2]).parent.name) == norm(family))]
        if not exact:
            exact = [c for c in candidates if norm(PurePosixPath(c[2]).stem) == norm(Path(desired).stem)
                     and (family is None or norm(PurePosixPath(c[2]).parent.name) == norm(family))]
        match = 'filename_match_unverified_historical_hash'
        options = exact
        if not options:
            # A current family variant is not an exact replacement for a historical file.
            wanted_family = norm(family or Path(desired).stem)
            options = [c for c in candidates if c[0] == 'google/fonts'
                       and norm(PurePosixPath(c[2]).parent.name) == wanted_family]
            match = 'family_only_variant'
        if not options:
            records.append(dict(requested=requested, status='unresolved'))
            continue
        options.sort(key=lambda c: (c[0] != 'google/fonts', 'italic' in c[2].lower(), len(c[2]), c[2]))
        repo, revision, p = options[0]
        parent = str(PurePosixPath(p).parent)
        tree = sources[repo][1]
        licenses = [x for x in tree if (str(PurePosixPath(x).parent) == parent or '/' not in x)
                    and re.match(r'^(ofl|license|licence|copying|copyright)', PurePosixPath(x).name, re.I)]
        metadata = parent + '/METADATA.pb'
        records.append(dict(requested=requested, status='resolved', match=match, repository=repo,
                            revision=revision, repository_path=p, git_blob_sha1=tree[p]['sha'],
                            licence_paths=licenses, metadata_path=metadata if metadata in tree else None))
    jobs = {}
    for r in records:
        if r['status'] != 'resolved':
            continue
        for p in [r['repository_path'], *r['licence_paths'], *([r['metadata_path']] if r['metadata_path'] else [])]:
            key = (r['repository'], r['revision'], p)
            jobs[key] = sources[r['repository']][1][p]['sha']

    def download(item):
        (repo, revision, p), expected = item
        dest = run / 'fonts/downloads' / repo.replace('/', '__') / p
        # Repository paths are remote data, not trusted filesystem instructions.
        if not dest.resolve().is_relative_to((run / 'fonts/downloads').resolve()):
            raise ValueError('Unsafe repository path')
        url = 'https://raw.githubusercontent.com/' + repo + '/' + revision + '/' + urllib.parse.quote(p)
        try:
            content = dest.read_bytes() if dest.exists() else fetch(url)
            blob = hashlib.sha1(b'blob ' + str(len(content)).encode() + b'\0' + content).hexdigest()
            if blob != expected:
                raise ValueError('Git blob hash mismatch: ' + p)
            if not dest.exists():
                dest.parent.mkdir(parents=True, exist_ok=True)
                tmp = dest.with_name(dest.name + '.partial')
                tmp.write_bytes(content)
                tmp.replace(dest)
            return item[0], dict(status='downloaded', path=str(dest), url=url, bytes=len(content),
                                 sha256=hashlib.sha256(content).hexdigest(), git_blob_sha1=blob)
        except Exception as exc:
            return item[0], dict(status='failed', url=url, error=str(exc))

    downloaded = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, min(workers, 12))) as pool:
        for i, (key, result) in enumerate(pool.map(download, jobs.items()), 1):
            downloaded[key] = result
            if i % 50 == 0:
                print(json.dumps(dict(downloaded_files=i, total_files=len(jobs))), flush=True)
                write(run / 'fonts/download_progress.json', dict(updated_at=now(), completed=i, total=len(jobs)))
    for r in records:
        if r['status'] != 'resolved':
            continue
        key = (r['repository'], r['revision'], r['repository_path'])
        r.update(downloaded[key])
        r['licences'] = [downloaded[(r['repository'], r['revision'], p)] for p in r.pop('licence_paths')]
        if r['metadata_path']:
            r['metadata'] = downloaded[(r['repository'], r['revision'], r['metadata_path'])]
    result = dict(created_at=now(), requested=len(records), status_counts=dict(collections.Counter(r['status'] for r in records)),
                  match_counts=dict(collections.Counter(r.get('match', 'unresolved') for r in records)),
                  repository_failures=failures, downloaded_files=len(downloaded))
    write(run / 'fonts/acquisition.json', dict(summary=result, records=records))
    return result


def index(run):
    from fontTools.ttLib import TTFont
    from fontTools.pens.recordingPen import DecomposingRecordingPen
    from PIL import ImageFont
    cfg = read(ROOT / 'configs/local.json')
    existing = read(Path(cfg['renderer']) / 'fonts/font_meta.json')
    acquisition = read(run / 'fonts/acquisition.json')
    inputs = [(Path(f['path']), dict(origin='existing_bank', legacy_metadata=f)) for f in existing]
    inputs += [(Path(r['path']), dict(origin='requested_inventory', acquisition=r))
               for r in acquisition['records'] if r['status'] == 'downloaded']
    by_hash, rejected = {}, []
    for path, provenance in inputs:
        try:
            digest = sha(path)
            if digest in by_hash:
                by_hash[digest]['provenance'].append(provenance)
                continue
            tt = TTFont(str(path), lazy=True)
            def name(i):
                value = tt['name'].getDebugName(i)
                return value.strip() if value else None
            family, style = name(16) or name(1), name(17) or name(2)
            cmap = tt.getBestCmap() or {}
            chars = ''.join(c for c in string.ascii_letters + string.digits
                            if ord(c) in cmap and tt.getGlyphID(cmap[ord(c)]) != 0)
            axes = [dict(tag=a.axisTag, min=float(a.minValue), default=float(a.defaultValue), max=float(a.maxValue))
                    for a in tt['fvar'].axes] if 'fvar' in tt else []
            caps_only = bool(chars) and all(cmap.get(ord(c)) == cmap.get(ord(c.upper())) for c in string.ascii_lowercase if c in chars)
            if caps_only:
                chars = ''.join(c for c in chars if not c.islower())
            pil = ImageFont.truetype(str(path), 40)
            chars = ''.join(c for c in chars if pil.getmask(c).getbbox())
            if len(chars) < 10 or not family:
                raise ValueError('Insufficient visible ASCII coverage or missing family')
            family_group = norm(family)
            bucket = int(hashlib.sha256(('capri-font-split-v1:' + family_group).encode()).hexdigest()[:8], 16) % 100
            split = 'train' if bucket < 80 else 'dev' if bucket < 88 else 'calibration' if bucket < 92 else 'test'
            legacy = provenance.get('legacy_metadata', {})
            # Normalized outlines detect renamed/duplicate ASCII faces across families.
            # This is exact-outline grouping, not a claim of exhaustive visual near-duplicate detection.
            glyphset=tt.getGlyphSet();units=float(tt['head'].unitsPerEm)
            outline=[]
            for c in chars:
                pen=DecomposingRecordingPen(glyphset)
                glyphset[cmap[ord(c)]].draw(pen)
                operations=[]
                for op,points in pen.value:
                    operations.append((op,[None if point is None else [round(float(v)/units,6) for v in point] for point in points]))
                outline.append((c,round(glyphset[cmap[ord(c)]].width/units,6),operations))
            signature=hashlib.sha256(json.dumps(outline,separators=(',',':')).encode()).hexdigest()
            downloaded=provenance.get('acquisition',{})
            meta_path=downloaded.get('metadata',{}).get('path')
            category=legacy.get('category','unknown')
            if category=='unknown' and meta_path and Path(meta_path).exists():
                m=re.search(r'category:\s*"([^"]+)"',Path(meta_path).read_text(encoding='utf-8'))
                if m:category={'SANS_SERIF':'Sans Serif','SERIF':'Serif','DISPLAY':'Display','HANDWRITING':'Handwriting','MONOSPACE':'Monospace'}.get(m[1],m[1])
            os2 = tt['OS/2'] if 'OS/2' in tt else None
            rec = dict(key=digest[:20], sha256=digest, path=str(path.resolve()), family=family, style=style,
                       family_group=family_group, split=split, chars=chars, caps_only=caps_only,
                       axes=axes, variable=bool(axes), italic=bool((os2.fsSelection & 1) if os2 else ('italic' in (style or '').lower())),
                       category=category, stroke=legacy.get('stroke', 'unknown'),ascii_outline_sha256=signature,
                       x_height=legacy.get('x_height'), width_rel=legacy.get('width_rel'),
                       weight_class=int(os2.usWeightClass) if os2 else None,
                       licence_name=name(13), licence_url=name(14), face_index=0, provenance=[provenance])
            by_hash[digest] = rec
            tt.close()
        except Exception as exc:
            rejected.append(dict(path=str(path), error=str(exc), origin=provenance['origin']))
    fonts = sorted(by_hash.values(), key=lambda f: f['sha256'])
    parent={f['family_group']:f['family_group'] for f in fonts}
    def root(name):
        while parent[name]!=name:
            parent[name]=parent[parent[name]];name=parent[name]
        return name
    signatures={}
    for f in fonts:
        sig=f['ascii_outline_sha256'];fam=f['family_group']
        if sig in signatures:
            a,b=root(fam),root(signatures[sig]);parent[max(a,b)]=min(a,b)
        signatures[sig]=fam
    for f in fonts:
        f['split_group']=root(f['family_group'])
        bucket=int(hashlib.sha256(('capri-font-split-v1:'+f['split_group']).encode()).hexdigest()[:8],16)%100
        f['split']='train' if bucket<80 else 'dev' if bucket<88 else 'calibration' if bucket<92 else 'test'
    result = dict(created_at=now(), unique_files=len(fonts), families=len({f['family_group'] for f in fonts}),
                  by_split=dict(collections.Counter(f['split'] for f in fonts)), rejected=len(rejected),
                  split_groups=len({f['split_group'] for f in fonts}),
                  split_rule='sha256 canonical family union exact normalized ASCII-outline aliases; 80/8/4/8 family allocation; nonidentical near-duplicates remain a limitation')
    write(run / 'fonts/font_manifest.json', dict(summary=result, fonts=fonts, rejected=rejected))
    return result


def archives(run):
    """Official release packages for legacy Linux font names in the supplied list."""
    import io
    import tarfile
    import zipfile
    bundles=[('ubuntu083','https://assets.ubuntu.com/v1/0cef8205-ubuntu-font-family-0.83.zip',None),
             ('freefont20120503','https://ftp.gnu.org/gnu/freefont/freefont-ttf-20120503.zip',None),
             ('dejavu237','https://downloads.sourceforge.net/project/dejavu/dejavu/2.37/dejavu-fonts-ttf-2.37.zip',
              '7576310b219e04159d35ff61dd4a4ec4cdba4f35c00e002a136f00e96a908b0a')]
    try:
        releases=json.loads(fetch('https://api.github.com/repos/liberationfonts/liberation-fonts/releases'))
        assets=[a for r in releases for a in r['assets'] if 'ttf' in a['name'] and a['name'].endswith('.tar.gz')]
        if assets:
            bundles.append(('liberation',assets[0]['browser_download_url'],None))
    except Exception as exc:
        print('Liberation release lookup:',str(exc),flush=True)
    evidence=[];files=[]
    for label,url,expected in bundles:
        archive=run/'fonts/archives'/label/('source.tar.gz' if url.endswith('.tar.gz') else 'source.zip')
        try:
            content=archive.read_bytes() if archive.exists() else fetch(url)
            digest=hashlib.sha256(content).hexdigest()
            if expected and digest!=expected:
                raise ValueError('Published release checksum mismatch')
            archive.parent.mkdir(parents=True,exist_ok=True)
            archive.write_bytes(content)
            if zipfile.is_zipfile(io.BytesIO(content)):
                z=zipfile.ZipFile(io.BytesIO(content))
                entries=[(n,z.read(n)) for n in z.namelist() if not n.endswith('/') and
                         (n.lower().endswith(('.ttf','.otf')) or re.search(r'(license|licence|copying|copyright|ofl|ufl)',PurePosixPath(n).name,re.I))]
            else:
                with tarfile.open(fileobj=io.BytesIO(content),mode='r:*') as z:
                    entries=[(m.name,z.extractfile(m).read()) for m in z if m.isfile() and
                             (m.name.lower().endswith(('.ttf','.otf')) or re.search(r'(license|licence|copying|copyright|ofl|ufl)',PurePosixPath(m.name).name,re.I))]
            extracted=[]
            for name,data in entries:
                path=archive.parent/'files'/name
                if not path.resolve().is_relative_to((archive.parent/'files').resolve()):
                    raise ValueError('Unsafe archive member')
                path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data)
                extracted.append(dict(path=str(path),sha256=hashlib.sha256(data).hexdigest(),member=name))
            licences=[x for x in extracted if not x['path'].lower().endswith(('.ttf','.otf'))]
            for f in extracted:
                if f['path'].lower().endswith(('.ttf','.otf')):
                    files.append(dict(**f,url=url,archive_sha256=digest,licences=licences))
            evidence.append(dict(label=label,url=url,sha256=digest,published_sha256=expected,files=len(extracted),status='downloaded'))
        except Exception as exc:
            evidence.append(dict(label=label,url=url,status='failed',error=str(exc)))
    acquisition=read(run/'fonts/acquisition.json')
    resolved=0
    for r in acquisition['records']:
        if r['status']!='unresolved':
            continue
        desired=PurePosixPath(r['requested']).name
        if desired.startswith(('sys_','orig_')):
            desired=desired.split('_',1)[1]
        matches=[f for f in files if PurePosixPath(f['member']).name.lower()==desired.lower()]
        if matches:
            r.update(matches[0]);r.update(status='downloaded',match='release_filename_match_unverified_historical_hash')
            resolved+=1
    acquisition['summary']['status_counts']=dict(collections.Counter(r['status'] for r in acquisition['records']))
    acquisition['summary']['archive_resolved']=resolved
    write(run/'fonts/archive_receipts.json',evidence)
    write(run/'fonts/acquisition.json',acquisition)
    return dict(resolved=resolved,bundles=evidence,status_counts=acquisition['summary']['status_counts'])


def historical(run):
    """Resolve removed Google static faces using their upstream path history."""
    acquisition=read(run/'fonts/acquisition.json')
    cache={};attempted=[]
    for rec in acquisition['records']:
        if rec['status']!='unresolved':
            continue
        desired=PurePosixPath(rec['requested']).name
        desired=re.sub(r'^(sys_|orig_|curated_)','',desired)
        special={'FredokaOne.ttf':'ofl/fredokaone/FredokaOne-Regular.ttf',
                 'SourceSansPro.ttf':'ofl/sourcesanspro/SourceSansPro-Regular.ttf',
                 'OswaldBold.ttf':'ofl/oswald/Oswald-Bold.ttf'}
        path=special.get(desired)
        if desired.startswith('OpenSans-'):
            path=('apache/opensanscondensed/' if 'Cond' in desired else 'apache/opensans/')+desired
        elif desired.startswith('RobotoSlab-'):
            path='apache/robotoslab/'+desired
        elif desired.startswith('Lato-'):
            path='ofl/lato/'+desired
        if path is None:
            continue
        if path not in cache:
            try:
                history=json.loads(fetch('https://api.github.com/repos/google/fonts/commits?path='+urllib.parse.quote(path)+'&per_page=1'))
                if not history:
                    raise ValueError('No upstream file history')
                latest=history[0]
                revisions=[latest['sha']]+[p['sha'] for p in latest['parents']]
                content=None
                for rev in revisions:
                    url='https://raw.githubusercontent.com/google/fonts/'+rev+'/'+urllib.parse.quote(path)
                    try:
                        content=fetch(url);break
                    except Exception:
                        continue
                if content is None:
                    raise ValueError('No readable historical revision')
                folder=run/'fonts/historical/google__fonts'/rev/PurePosixPath(path).parent
                folder.mkdir(parents=True,exist_ok=True)
                dest=folder/PurePosixPath(path).name;dest.write_bytes(content)
                licences=[]
                for name in ['OFL.txt','LICENSE.txt']:
                    license_url='https://raw.githubusercontent.com/google/fonts/'+rev+'/'+str(PurePosixPath(path).parent)+'/'+name
                    try:
                        data=fetch(license_url);lp=folder/name;lp.write_bytes(data)
                        licences.append(dict(path=str(lp),url=license_url,sha256=hashlib.sha256(data).hexdigest()))
                        break
                    except Exception:
                        pass
                cache[path]=dict(status='downloaded',match='historical_upstream_face_unverified_inventory_hash',path=str(dest),
                                 url=url,sha256=hashlib.sha256(content).hexdigest(),repository='google/fonts',revision=rev,
                                 repository_path=path,licences=licences)
                attempted.append(dict(path=path,status='downloaded',revision=rev))
            except Exception as exc:
                cache[path]=None;attempted.append(dict(path=path,status='unresolved',error=str(exc)))
        if cache[path]:
            rec.update(cache[path])
    acquisition['summary']['status_counts']=dict(collections.Counter(r['status'] for r in acquisition['records']))
    write(run/'fonts/historical_receipts.json',attempted)
    write(run/'fonts/acquisition.json',acquisition)
    return dict(status_counts=acquisition['summary']['status_counts'],attempted=attempted)
