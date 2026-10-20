"""Unselected first six held-out examples, preserving each image's aspect ratio."""
import common as U
from PIL import Image,ImageDraw,ImageOps

def main():
    import sys
    for family in sys.argv[1:] or ['CryptBB','DarkBay']:
        out=U.ROOT/f'{family}_run_01';split=U.read(out/'target_splits.json')
        baseline=U.read(out/'baseline/labels.json');refined=U.read(out/'refined/labels.json')
        sheet=Image.new('RGB',(1080,680),'#f4f5f7');draw=ImageDraw.Draw(sheet)
        draw.text((12,10),family+' | first six held-out examples; synthetic strings are different',fill='black')
        for j,title in enumerate(['TARGET','HYBRID START','CONTROLLER RESULT']):draw.text((j*360+12,35),title,fill='black')
        for i in range(6):
            for j,path in enumerate([split['test'][i]['source'],out/baseline[i]['image'],out/refined[i]['image']]):
                im=Image.open(path).convert('RGB');im=ImageOps.contain(im,(344,88),Image.Resampling.NEAREST)
                x=j*360+(360-im.width)//2;y=60+i*102+(88-im.height)//2
                sheet.paste(im,(x,y))
        sheet.save(out/'preview.png')
        result=U.read(out/'results.json');a=result['test']['baseline'];b=result['test']['refined']
        print(family,{k:round(100*(1-b[k]/a[k]),1) for k in ['text','background','appearance','audit']},'usable',a['usable'],b['usable'],flush=True)
if __name__=='__main__':main()
