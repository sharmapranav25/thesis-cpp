"""Frozen output dimensions used by the custom encoder checkpoint."""
CONT={
 'size':(.4,1.15),'width':(.55,1.4),'weight':(0.,1.),'spacing':(-.05,1.3),
 'rotation':(0.,30.),'jitter':(0.,.24),'size_jitter':(0.,.25),'slant':(-.25,.25),
 'bg_h':(0.,1.),'bg_s':(0.,.65),'bg_v':(.12,1.),'bg_hspread':(0.,1.),
 'fg_h':(0.,1.),'fg_s':(0.,.9),'fg_v':(.05,.95),'fg_hspread':(0.,1.),'fg_opacity':(.6,1.),
 'fx_h':(0.,1.),'fx_s':(0.,.75),'fx_v':(.12,.98),'fx_hspread':(0.,1.),'fx_opacity':(.2,1.),
 'block_count':(0.,30.),'block_width':(.025,.35),'block_height':(.1,1.35),'block_contrast':(.02,.35),'outline_fraction':(0.,1.),
 'band_count':(0.,8.),'band_width':(.01,.12),'band_slope':(0.,.55),
 'grid_count':(0.,30.),'line_count':(0.,35.),'curve_count':(0.,25.),'curve_amplitude':(.03,.35),
 'dots':(0.,700.),'noise':(0.,.15),'pixel':(1.,3.),'overlay':(0.,.5)}
CAT={'font':96,'length':7,'charset':5,'fg_mode':4,'block_mode':2,'fx_mode':3,'bold':2,'border':2,'font_mix':2}
CN=list(CONT)
KN=list(CAT)
CHARSETS=['digits','upper','lower','upper_digits','mixed']
