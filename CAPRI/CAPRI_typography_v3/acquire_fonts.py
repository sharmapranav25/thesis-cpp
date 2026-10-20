"""Acquire independent OFL fonts at an immutable Google Fonts revision."""
from pathlib import Path
import json,hashlib,urllib.request,concurrent.futures,string,shutil
from fontTools.ttLib import TTFont
ROOT=Path(__file__).resolve().parent
BASE=ROOT.parent/'CAPRI_v2/base'
COMMIT='23e54b51ddffbc7713c583748e3bd86f62b1fa4a'
NAMES='arimo tinos cousine opensans lato raleway rubik worksans nunito librefranklin merriweather playfairdisplay teko bangers abel acme alegreya asap barlow breeSerif cabin comfortaa exo2 fjallaone'.lower().split()
def sha(b):return hashlib.sha256(b).hexdigest()
def fetch(url):
 return urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'CAPRI-research-font-provenance'}),timeout=60).read()
def acquire(name):
 try:
  url=f'https://api.github.com/repos/google/fonts/contents/ofl/{name}?ref={COMMIT}'
  listing=json.loads(fetch(url));ttfs=sorted([r for r in listing if r['name'].endswith('.ttf')],key=lambda r:('Italic' in r['name'],'Regular' not in r['name'],r['name']))
  lic=next(r for r in listing if r['name']=='OFL.txt');r=ttfs[0]
  folder=ROOT/'fonts'/name;folder.mkdir(parents=True,exist_ok=True)
  data=fetch(r['download_url']);notice=fetch(lic['download_url'])
  assert b'SIL OPEN FONT LICENSE' in notice.upper()
  file=folder/'font.ttf';file.write_bytes(data);(folder/'OFL.txt').write_bytes(notice)
  with TTFont(file) as font:
   cmap=font.getBestCmap();assert all(ord(c) in cmap for c in string.ascii_letters+string.digits)
   family=font['name'].getDebugName(1)
  return dict(family=family,path=str(file.relative_to(ROOT)),sha256=sha(data),license_path=str((folder/'OFL.txt').relative_to(ROOT)),license_sha256=sha(notice),source=r['download_url'],commit=COMMIT,ascii_alphanumeric_checked=True)
 except Exception as e:return dict(rejected=name,error=repr(e))
def main():
 (ROOT/'fonts').mkdir(exist_ok=True);rows=[]
 for i,r in enumerate(json.loads((BASE/'assets/fonts.json').read_text())):
  folder=ROOT/'fonts'/f'existing_{i:02d}';folder.mkdir(exist_ok=True)
  for src,dst in [(r['path'],'font.ttf'),(r['license_path'],'OFL.txt')]:shutil.copy2(BASE/src,folder/dst)
  assert sha((folder/'font.ttf').read_bytes())==r['sha256']
  assert sha((folder/'OFL.txt').read_bytes())==r['license_sha256']
  rows.append(dict(family=r['family'],path=str((folder/'font.ttf').relative_to(ROOT)),sha256=r['sha256'],license_path=str((folder/'OFL.txt').relative_to(ROOT)),license_sha256=r['license_sha256'],source=r.get('source'),ascii_alphanumeric_checked=True))
 with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:results=list(pool.map(acquire,NAMES))
 rows.extend(r for r in results if 'rejected' not in r)
 unique={r['sha256']:r for r in rows};rows=list(unique.values())
 (ROOT/'fonts.json').write_text(json.dumps(rows,indent=2),encoding='utf8')
 (ROOT/'font_acquisition.json').write_text(json.dumps(dict(commit=COMMIT,accepted=len(rows),rejected=[r for r in results if 'rejected' in r]),indent=2))
 print((ROOT/'font_acquisition.json').read_text())
if __name__=='__main__':main()
