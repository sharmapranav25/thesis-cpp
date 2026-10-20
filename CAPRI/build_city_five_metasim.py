from pathlib import Path
W=Path(__file__).resolve().parent
D=W/'city_five_metasim_01';D.mkdir(exist_ok=False)
SOURCE=Path('E:/pranav/Final sem/synth_data/city_capri_phase3_5digit_100k')
old='city_capri_v3_variable_2to8_100k'
s=(W/'city_metasim_worker.py').read_text().replace(old,SOURCE.name)
needle="cells=[(b,n) for b in ('numeric','alphabetic','alphanumeric') for n in sorted(set(a.lengths))]"
assert needle in s;s=s.replace(needle,"cells=[('numeric',5)]")
helper="""def make_phase3_text(basegen,rng,bucket,n,partition):
 for attempt in range(10000):
  text,sub=basegen.make_text(rng,bucket,n,partition)
  k=int(hashlib.sha256(('capri-phase3-city-v1:'+text).encode()).hexdigest()[:8],16)%100
  if k<80:return text,sub
 raise RuntimeError('No permitted training text')

"""
s=s.replace('def main():',helper+'def main():',1).replace('text,sub=basegen.make_text(rng,bucket,n,a.partition)','text,sub=make_phase3_text(basegen,rng,bucket,n,a.partition)')
compile(s,'city_metasim_worker.py','exec');(D/'city_metasim_worker.py').write_text(s)
(D/'city_metasim_pair.py').write_bytes((W/'city_metasim_pair.py').read_bytes())
c=(W/'generate_city_metasim_100k.py').read_text().replace(old,SOURCE.name).replace('9996 if i<9 else 10036','10000').replace('202609286000','202609295500').replace('workers=6','workers=4').replace('max_workers=6','max_workers=4')
c=c.replace("'--lengths','2','3','4','5','6','7','8'","'--lengths','5'").replace('lengths=list(range(2,9))','lengths=[5]').replace("buckets=['numeric','alphabetic','alphanumeric']","buckets=['numeric']")
c=c.replace("length_basis='Broad first-stage 2-8 support; no target length assumed'","length_basis='Preserved user-selected five-digit source scheme; see copied scheme.json'")
c=c.replace('city_variable_100k_metasim_status.json','city_phase3_5digit_metasim_status.json')
c=c.replace(" assert len(r['label'])==r['length'] and 2<=r['length']<=8", " assert len(r['label'])==r['length']==5 and r['label'].isascii() and r['label'].isdigit()")
c=c.replace("     t=r['label'];bucket=r['bucket'];assert t.isascii()", "     t=r['label'];bucket=r['bucket'];assert t.isascii() and bucket=='numeric'\n     assert int(hashlib.sha256(('capri-phase3-city-v1:'+t).encode()).hexdigest()[:8],16)%100<80")
c=c.replace(" write(OUT/'protocol.json',protocol)"," protocol['source_protocol_sha256']=hashlib.sha256((SOURCE/'protocol.json').read_bytes()).hexdigest()\n protocol['text_partition']='phase12-text-v1 train AND capri-phase3-city-v1 train'\n (OUT/'scheme.json').write_bytes((SOURCE/'scheme.json').read_bytes())\n (OUT/'source_protocol.json').write_bytes((SOURCE/'protocol.json').read_bytes())\n write(OUT/'protocol.json',protocol)")
c=c.replace('lengths2-8','exactly5ASCII digits').replace('# City paired Meta-Sim-inspired corpus','# City five-digit paired Meta-Sim-inspired corpus')
compile(c,'generate_city_metasim_100k.py','exec');(D/'generate_city_metasim_100k.py').write_text(c)
print(D)
