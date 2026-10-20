from pathlib import Path
p=Path(__file__).parent;s=(p/'run_v2_tier1_04.py').read_text()
s=s.replace("/'v2_tier1_04'","/'v2_tier1_05'")
s=s.replace("for name,separate in [('shared_local_head',False),('grid_specialists',True)]:", "for name,separate in [('no_local_aux',True),('coarse_curve_aux',True)]:")
s=s.replace("n=cols*rows;aux.append(F.binary_cross_entropy_with_logits(model.last_local[:,offset:offset+n],t[ids,offset:offset+n],pos_weight=local_weights[g]));offset+=n", """n=cols*rows;terms=F.binary_cross_entropy_with_logits(model.last_local[:,offset:offset+n],t[ids,offset:offset+n],pos_weight=local_weights[g],reduction='none')
                if name=='coarse_curve_aux' and g>=6:
                    keep=torch.ones(5,device='cuda');keep[1]=0;terms=terms*keep
                aux.append(terms.mean());offset+=n""")
s=s.replace("loss=loss+auxiliary", "loss=loss+(0 if name=='no_local_aux' else 1)*auxiliary")
# Training stays exactly matched to04 except auxiliary intervention.
s=s.replace("fresh_seeds=[51000000,52000000,53000000]", "fresh_seeds=[61000000,62000000,63000000],interventions=['all auxiliary weight0','only 5x3/10x3 curve auxiliary terms zero; no reweighting of remaining terms']")
s=s.replace("for name,separate in [('original03',None),('shared_local_head',False),('grid_specialists',True)]:", "for name,separate in [('original03',None),('original04',True),('no_local_aux',True),('coarse_curve_aux',True)]:")
s=s.replace("else OUT/(name+'.pt')", "else HERE/'v2_tier1_04/grid_specialists.pt' if name=='original04' else OUT/(name+'.pt')")
s=s.replace("51000000", "61000000").replace("52000000", "62000000").replace("53000000", "63000000")
s=s.replace("if separate is True:\n            real={}", "if name=='coarse_curve_aux':\n            real={}")
(p/'run_v2_tier1_05.py').write_text(s,encoding='utf8')
print('built05')
