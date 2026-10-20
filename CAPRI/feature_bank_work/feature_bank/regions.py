"""Native-pixel region geometry. No detector, dataset-specific rule or labels."""
import math
import numpy as np

GRID_VERSION = 'regions-v1'
SCALES = ((2, 4), (3, 6))
HALO_FRACTION = 0.125  # Each side of a base brick; adjacent overlap ~25%.

def regions(width, height, scales=SCALES, halo=HALO_FRACTION):
    if width < 1 or height < 1 or not 0 <= halo <= 0.5:
        raise ValueError('Invalid image or overlap')
    output=[]
    for rows, cols in scales:
        if not (1 <= rows <= height and 1 <= cols <= width):
            raise ValueError('Grid would create empty bricks')
        xs=[i*width//cols for i in range(cols+1)]
        ys=[i*height//rows for i in range(rows+1)]
        for r in range(rows):
            for c in range(cols):
                core=[xs[c],ys[r],xs[c+1],ys[r+1]]
                hx=math.ceil((core[2]-core[0])*halo); hy=math.ceil((core[3]-core[1])*halo)
                box=[max(0,core[0]-hx),max(0,core[1]-hy),min(width,core[2]+hx),min(height,core[3]+hy)]
                output.append(dict(id=f'{rows}x{cols}-r{r+1}c{c+1}',scale=[rows,cols],row=r,column=c,
                   core_box=core,box=box,normalized_box=[box[0]/width,box[1]/height,box[2]/width,box[3]/height],
                   coordinate_convention='x0,y0 inclusive; x1,y1 exclusive',padding_pixels=0))
    return output

def overlap_edges(records):
    edges=[]
    for i,a in enumerate(records):
        for b in records[i+1:]:
            if a['scale']!=b['scale']: continue
            x0=max(a['box'][0],b['box'][0]); y0=max(a['box'][1],b['box'][1])
            x1=min(a['box'][2],b['box'][2]); y1=min(a['box'][3],b['box'][3])
            if x1>x0 and y1>y0:
                edges.append(dict(a=a['id'],b=b['id'],shared_box=[x0,y0,x1,y1],shared_pixels=(x1-x0)*(y1-y0),
                                  meaning='shared source pixels only; not evidence of a connected feature'))
    return edges

def verify_crops(image,records):
    source=np.asarray(image)
    for scale in {tuple(r['scale']) for r in records}:
        coverage=np.zeros(source.shape[:2],dtype=np.int32)
        accumulated=np.zeros(source.shape,dtype=np.int64)
        for r in records:
            if tuple(r['scale'])!=scale: continue
            x0,y0,x1,y1=r['box']; crop=np.asarray(image.crop(r['box']))
            if not np.array_equal(crop,source[y0:y1,x0:x1]): raise ValueError('Crop changed pixels')
            accumulated[y0:y1,x0:x1]+=crop; coverage[y0:y1,x0:x1]+=1
        if not (coverage>0).all(): raise ValueError('Uncovered source pixels')
        if not np.array_equal(accumulated,source.astype('int64')*coverage[:,:,None]):
            raise ValueError('Overlap reassembly does not reconstruct original')
    return {'native_pixels_exact':True,'full_coverage_per_scale':True,'overlap_reassembly_exact':True}

def check_group_partitions(records):
    groups={}; hashes={}
    for row in records:
        for mapping,key in ((groups,row['base_group']),(hashes,row['parent_pixel_sha256'])):
            if key in mapping and mapping[key]!=row['partition']:
                raise ValueError('Base group or duplicate parent crosses partitions')
            mapping[key]=row['partition']
    return True
