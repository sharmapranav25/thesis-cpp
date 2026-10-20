from pathlib import Path
p=Path('render_direct_bank.py').read_text()
p=p.replace("'direct_bank_renders_01'","'count_font_bank_01'")
p=p.replace('from style_pipeline.render import render','from style_pipeline.render import render\nfrom train_character_count import CountCNN,encode')
p=p.replace('families={};variants={};old=', '''count_run=ROOT/'runs/style_character_count_01'
count_checkpoint=count_run/'count_model.pt';count_hash=digest(count_checkpoint)
count_model=CountCNN().cuda().eval();count_model.load_state_dict(torch.load(count_checkpoint,weights_only=True)['state_dict'])
temperature=read(count_run/'test_results.json')['temperature']
count_references=read(count_run/'real_inference.json');count_records={}
families={};variants={};old=''' )
p=p.replace("    p,h,_=infer(model,images);",'''    count_rows=count_references[name]['predictions'];count_images=[load(r['path']) for r in count_rows]
    with torch.no_grad(): count_probs=(count_model(torch.tensor(np.stack([encode(im) for im in count_images]),device='cuda'))/temperature).softmax(1).cpu().numpy()
    family_prob=count_probs.mean(0);cdf=family_prob.cumsum();cdf[-1]=1
    lengths=[int(np.searchsorted(cdf,q)+2) for q in (.125,.375,.625,.875)]
    count_records[name]=dict(probabilities=family_prob.tolist(),lengths=lengths,mode=int(family_prob.argmax()+2),references=[dict(path=r['path'],pixel_sha256=pixel_hash(im),probabilities=prob.tolist()) for r,im,prob in zip(count_rows,count_images,count_probs)])
    p,h,_=infer(model,images);''')
p=p.replace("choice(list('ABCDEFGHJKLMNPQRSTUVWXYZ23456789'),5)","choice(list('ABCDEFGHJKLMNPQRSTUVWXYZ23456789'),lengths[i])")
p=p.replace('        records.append(recipe)','''        assert len(text)==lengths[i]
        draw.text((10+i*250,410),f'Generated length {len(text)} | estimated',fill='black')
        records.append(dict(text=text,requested_length=len(text),recipe=recipe))''')
p=p.replace('assert digest(checkpoint)==before','''assert digest(checkpoint)==before
assert digest(count_checkpoint)==count_hash
write(out/'count_inference.json',count_records)
for k in range(2):
    board=Image.new('RGB',(1000,1720),'white')
    for i,panel in enumerate(summary_panels[k*4:k*4+4]):board.paste(panel,(0,430*i))
    board.save(out/f'bank_{k+1}.png')''')
p=p.replace('generic_character_count=5','count_model_sha256=count_hash,count_temperature=temperature,count_policy="family mean posterior; four fixed stratified quantiles .125/.375/.625/.875, not a calibrated real error margin or verified family length distribution",count_support=list(range(2,9)),optional_phase3=False')
p=p.replace("sequence='Phase1 then frozen font inference; no former Phase2 recipe/refiner or typography gains'","sequence='Phase1 CV + frozen features + fresh frozen count inference, then Phase2 font inference; no optional refinement'")
Path('render_count_font_bank.py').write_text(p,encoding='utf8')
