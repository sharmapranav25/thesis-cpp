"""Pixel-only spatial style measurements. Heuristic glyph proposals, not OCR."""
import cv2
import numpy as np
from feature_bank.regions import regions,overlap_edges
from .regions_text import propose
from .background_blocks import measure_blocks

def palette(pixels,k=5):
    if not len(pixels):return [[128,128,128]]
    q=(pixels.astype('int32')//32).clip(0,7);code=q[:,0]*64+q[:,1]*8+q[:,2]
    counts=np.bincount(code,minlength=512);ids=np.argsort(-counts)[:min(k,np.count_nonzero(counts))]
    return [np.median(pixels[code==i],axis=0).astype(int).tolist() for i in ids]

def extract_style(image):
    rgb=np.asarray(image.convert('RGB'));h,w=rgb.shape[:2]
    gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY);hsv=cv2.cvtColor(rgb,cv2.COLOR_RGB2HSV)
    median=float(np.median(gray));dark_background=median<128
    contrast=gray.astype(float)-median if dark_background else median-gray.astype(float)
    ink=(contrast>max(25,float(np.quantile(contrast,.78)))).astype('uint8')
    n,components,stats,centers=cv2.connectedComponentsWithStats(ink,8)
    textmask=np.zeros((h,w),np.uint8);glyphs=[]
    for i in range(1,n):
        x,y,bw,bh,area=map(int,stats[i])
        if .15*h<=bh<=.94*h and 1<=bw<=.27*w and area>=max(5,h*w*.0006):
            textmask[components==i]=1
            yy,xx=np.where(components==i);cov=np.cov(np.stack([xx,yy])) if len(xx)>2 else np.eye(2)
            vals,axes=np.linalg.eigh(cov);axis=axes[:,-1]
            angle=float((np.degrees(np.arctan2(axis[1],axis[0]))-90+90)%180-90)
            # Shape axis may reflect the letter's design rather than rotation.
            glyphs.append(dict(box=[x/w,y/h,(x+bw)/w,(y+bh)/h],center=[float(centers[i,0]/w),float(centers[i,1]/h)],
                height=bh/h,width=bw/w,axis_tilt=angle,tilt_is_rotation=False))
    mser=propose(gray)
    if len(mser)>=3:
        glyphs=[];textmask[:]=0
        for c in mser:
            x,y,x1,y1=c['box'];pts=c['points'];textmask[pts[:,1],pts[:,0]]=1
            cov=np.cov(pts.T);_,axes=np.linalg.eigh(cov);axis=axes[:,-1]
            angle=float((np.degrees(np.arctan2(axis[1],axis[0]))+90)%180-90)
            axis_tilt=float((angle-90+90)%180-90)
            glyphs.append(dict(box=[x/w,y/h,x1/w,y1/h],center=[(x+x1)/(2*w),(y+y1)/(2*h)],
                height=(y1-y)/h,width=(x1-x)/w,axis_tilt=axis_tilt,tilt_is_rotation=False))
    glyphs.sort(key=lambda g:g['center'][0]);dilated=cv2.dilate(textmask,np.ones((3,3),np.uint8))>0
    bg=rgb[~dilated];fg=rgb[textmask>0];bgpalette=palette(bg);fgpalette=palette(fg)
    # Keep high-contrast residual glyph colors out of the background palette.
    anchor=np.array(bgpalette[0],float)
    bgpalette=[c for c in bgpalette if np.linalg.norm(np.array(c)-anchor)<110] or [bgpalette[0]]
    # Robust column medians describe low-frequency background change without
    # storing or reusing reference patches. End bands reduce single-column noise.
    column=np.median(rgb,axis=0)
    band=max(1,w//8)
    gradient_left=np.median(column[:band],axis=0)
    gradient_right=np.median(column[-band:],axis=0)
    gradient_strength=float(np.linalg.norm(gradient_right-gradient_left))
    edges=cv2.Canny(gray,40,120);edges[dilated]=0
    detected=cv2.HoughLinesP(edges,1,np.pi/180,threshold=max(8,w//12),minLineLength=max(8,w//10),maxLineGap=3)
    lines=[];hist=np.zeros(12)
    if detected is not None:
        for x0,y0,x1,y1 in detected[:,0][:150]:
            length=float(np.hypot(x1-x0,y1-y0));angle=float(np.degrees(np.arctan2(y1-y0,x1-x0))%180)
            hist[min(11,int(angle//15))]+=length
            lines.append(dict(points=[int(x0),int(y0),int(x1),int(y1)],angle=angle,length=length))
    hist=hist/max(hist.sum(),1)
    residual=np.max(np.abs(rgb.astype('int16')-cv2.medianBlur(rgb,3).astype('int16')),axis=2)
    texture_fraction=float((residual[~dilated]>20).mean()) if (~dilated).any() else 0.
    nn,_,ss,_=cv2.connectedComponentsWithStats((edges>0).astype('uint8'),8)
    small=int(sum(1 for s in ss[1:] if 1<=s[4]<=8 and s[2]<=4 and s[3]<=4))
    rs=regions(w,h,scales=((3,10),));blocks=[]
    for r in rs:
        x0,y0,x1,y1=r['core_box'];patch=rgb[y0:y1,x0:x1];ee=edges[y0:y1,x0:x1]
        blocks.append(dict(id=r['id'],core_box=r['core_box'],box=r['box'],rgb_mean=patch.mean((0,1)).tolist(),
            edge_fraction=float((ee>0).mean()),text_fraction=float(textmask[y0:y1,x0:x1].mean()),
            global_line_ids=[i for i,l in enumerate(lines) if cv2.clipLine((x0,y0,x1-x0,y1-y0),tuple(l['points'][:2]),tuple(l['points'][2:]))[0]]))
    adjacent=overlap_edges(rs)
    byid={b['id']:b for b in blocks}
    for a in adjacent:
        p,q=byid[a['a']],byid[a['b']]
        a['color_difference']=float(np.linalg.norm(np.array(p['rgb_mean'])-q['rgb_mean'])/441.7)
        a['shared_global_line_ids']=sorted(set(p['global_line_ids'])&set(q['global_line_ids']))
    heights=[g['height'] for g in glyphs];tilts=[g['axis_tilt'] for g in glyphs]
    result=dict(size=[w,h],background_palette=bgpalette,foreground_palette=fgpalette,dark_background=dark_background,
        gradient_left=gradient_left.tolist(),gradient_right=gradient_right.tolist(),gradient_strength=gradient_strength,
        glyph_proposals=glyphs,glyph_count_estimate=len(glyphs),glyph_segmentation_status='MSER_heuristic' if len(mser)>=3 else 'contrast_heuristic',
        glyph_stroke_width=float(np.median([c['stroke'] for c in mser])) if mser else None,
        hollow_score=float(np.median([c['hollow'] for c in mser])) if mser else 0.,
        text_height=float(np.median(heights)) if heights else .5,
        text_tilt_proxy=float(np.median(np.abs(tilts))) if tilts else None,
        background_edge_fraction=float((edges>0).mean()),small_component_density=small/(w*h),
        texture_fraction=texture_fraction,
        line_angle_histogram=hist.tolist(),lines=lines,blocks=blocks,adjacency=adjacent,
        saturation_mean=float(hsv[:,:,1].mean()/255),gray_std=float(gray.std()/255),
        text_occupancy=float(textmask.mean()),coverage_pixels=int(sum((r['core_box'][2]-r['core_box'][0])*(r['core_box'][3]-r['core_box'][1]) for r in rs)))
    result['background_block_evidence']=measure_blocks(rgb,textmask)
    assert result['coverage_pixels']==w*h
    return result

def vector(style):
    colors=np.array([b['rgb_mean'] for b in style['blocks']])/255
    edges=np.array([b['edge_fraction'] for b in style['blocks']]);text=np.array([b['text_fraction'] for b in style['blocks']])
    return np.r_[colors.mean(0),colors.std(0),style['saturation_mean'],style['gray_std'],style['text_height'],
        style['text_occupancy'],style['background_edge_fraction'],min(style['small_component_density']*100,1),
        style['line_angle_histogram'],edges.reshape(3,10).mean(1),text.reshape(3,10).mean(1),
        colors.reshape(3,10,3).mean(1).flatten()].astype(float)
