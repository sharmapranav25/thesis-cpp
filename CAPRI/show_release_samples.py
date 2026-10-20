from pathlib import Path
import json
from PIL import Image,ImageDraw,ImageFont
root=Path(__file__).parent/'phase12_validation_01'
names=['cryptbb','city','courier','darkbay','outlined_text_captcha','roncuevas','pymlex','sloth']
font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',17)
small=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',14)
for page in range(2):
    board=Image.new('RGB',(1050,760),'#f1f3f5');d=ImageDraw.Draw(board)
    for c,label in enumerate(['Numeric','Alphabetic','Alphanumeric']):d.text((12+350*c,8),label,font=font,fill='black')
    for r,name in enumerate(names[page*4:page*4+4]):
        rows=[json.loads(s) for s in (root/name/'manifest.jsonl').read_text().splitlines()];y=40+180*r
        d.text((12,y),name,font=font,fill='black')
        for c,bucket in enumerate(['numeric','alphabetic','alphanumeric']):
            row=next(v for v in rows if v['bucket']==bucket and v['length']==5)
            im=Image.open(root/name/row['filename']).convert('RGB');im.thumbnail((330,108));board.paste(im,(12+350*c,y+28));d.text((12+350*c,y+145),'Generated label: '+row['label'],font=small,fill='black')
    board.save(root/f'release_examples_{page+1}.png')
