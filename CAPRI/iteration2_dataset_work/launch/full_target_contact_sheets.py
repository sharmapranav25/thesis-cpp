import json,math
from pathlib import Path
from PIL import Image,ImageDraw,ImageFont
root=Path('C:/Users/pranav/Documents/ChatGPT/thesis/iter2_full_target_recreation_01')
dest=root/'contact_sheets';dest.mkdir(exist_ok=True)
rows=[json.loads(line) for line in (root/'predictions.jsonl').read_text().splitlines()]
font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',14)
titlefont=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',24)
manifest=[]
for family in ('CryptBB','City','DarkBay'):
    selected=[r for r in rows if r['family']==family]
    for start in range(0,len(selected),100):
        batch=selected[start:start+100];cellw,cellh=200,135
        sheet=Image.new('RGB',(2000,75+math.ceil(len(batch)/10)*cellh),'#dce1e8');draw=ImageDraw.Draw(sheet)
        draw.text((12,8),f'{family}: images {start+1}-{start+len(batch)} of {len(selected)}',font=titlefont,fill='black')
        draw.text((12,42),'Every cell: REAL ABOVE / RECREATION BELOW. Fresh text; default colours/positions. Red label = readability check failed.',font=font,fill='black')
        for offset,row in enumerate(batch):
            x=(offset%10)*cellw;y=75+(offset//10)*cellh
            draw.rectangle((x+2,y+2,x+cellw-2,y+cellh-2),fill='white')
            colour='black' if row.get('render_usable') else '#b40000'
            draw.text((x+6,y+4),f'#{row["index"]+1} | predicted length {row.get("length","?")}',font=font,fill=colour)
            if 'error' in row:continue
            for k,suffix in enumerate(('original','render')):
                path=root/f'{family}_{row["index"]:02d}_{suffix}.png'
                with Image.open(path) as im:
                    im=im.convert('RGB');im.thumbnail((188,49),Image.Resampling.LANCZOS)
                    sheet.paste(im,(x+6+(188-im.width)//2,y+25+k*53+(49-im.height)//2))
        path=dest/f'{family}_{start//100+1:02d}.jpg';sheet.save(path,quality=93,subsampling=0)
        manifest.append(dict(family=family,start=start+1,end=start+len(batch),path=str(path)))
(dest/'manifest.json').write_text(json.dumps(manifest,indent=2))
print(json.dumps(dict(sheets=len(manifest),pairs=len(rows),directory=str(dest))))
