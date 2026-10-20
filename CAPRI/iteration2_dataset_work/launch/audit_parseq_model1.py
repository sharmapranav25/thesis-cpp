"""Read-only checkpoint audit; never changes the active trainer or its checkpoint."""
import runpy,json,time,collections
from pathlib import Path
v=runpy.run_path(str(Path(__file__).with_name('train_parseq0_model1.py')))
torch=v['torch'];np=v['np'];torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.2)
out=Path('C:/Users/pranav/Documents/ChatGPT/thesis/parseq0_model1_audit_01');out.mkdir(exist_ok=True)
source=v['PROJECT']/'runs/parseq0_model1/epochs/epoch_03.pt'
checkpoint=torch.load(source,map_location='cpu',weights_only=False);cfg=checkpoint['config'];model=v['m1'].build_parseq(cfg).cuda().eval();model.load_state_dict(checkpoint['state_dict']);model.log=lambda *a,**k:None
report=dict(checkpoint=str(source),epoch=checkpoint['epoch'],checkpoint_sha256=v['sha'](source),splits={})
@torch.inference_mode()
def trial(images,labels):
    memory=model.model.encode(images);target=model.tokenizer.encode(labels,model.device);perm=model.gen_tgt_perms(target)[0];tm,qm=model.generate_attn_masks(perm)
    logits=model.model.head(model.model.decode(target[:,:-1],memory,tm,(target[:,:-1]==model.pad_id)|(target[:,:-1]==model.eos_id),tgt_query_mask=qm))
    valid=(target[:,1:]!=model.pad_id)&(target[:,1:]!=model.eos_id)
    acc=float((logits.argmax(-1)[valid]==target[:,1:][valid]).float().mean())
    import torch.nn.functional as F
    loss=float(F.cross_entropy(logits.flatten(0,1),target[:,1:].flatten(),ignore_index=model.pad_id))
    decoded={}
    for refinements in (0,1):
        model.model.refine_iters=refinements
        probs=model(images).softmax(-1);texts,_=model.tokenizer.decode(probs);decoded[str(refinements)]=texts
    return loss,acc,decoded
for split in ('train','selection'):
    dataset=v['HybridImages'](split,cfg,False);indices=np.random.default_rng(731).choice(len(dataset),64,replace=False);losses=[];accuracies=[];pred={str(i):[] for i in (0,1)};truth=[];blank_losses=[];swapped_losses=[]
    for begin in range(0,64,16):
        pairs=[dataset[int(i)] for i in indices[begin:begin+16]];images=torch.stack([p[0] for p in pairs]).cuda();labels=[p[1] for p in pairs];truth+=labels
        loss,acc,decoded=trial(images,labels);losses.append(loss);accuracies.append(acc)
        for key in pred:pred[key]+=decoded[key]
        model.model.refine_iters=1
        blank_losses.append(float(model.training_step((torch.ones_like(images),labels),0)));swapped_losses.append(float(model.training_step((images.flip(0),labels),0)))
    details={}
    for key,texts in pred.items():
        details[key]=dict(exact_accuracy=sum(a==b for a,b in zip(truth,texts))/64,cer=sum(v['M'].levenshtein(a,b) for a,b in zip(truth,texts))/sum(map(len,truth)),
            unique_predictions=len(set(texts)),top_predictions=collections.Counter(texts).most_common(5),examples=[dict(label=a,prediction=b) for a,b in zip(truth,texts)][:12])
    report['splits'][split]=dict(teacher_forced_loss=float(np.mean(losses)),teacher_forced_character_accuracy=float(np.mean(accuracies)),blank_image_loss=float(np.mean(blank_losses)),swapped_image_loss=float(np.mean(swapped_losses)),decoding_by_refinement_iterations=details)
    # Direct label/token round-trip and original/augmented previews.
    montage=v['Image'].new('RGB',(800,8*150),'white')
    from PIL import ImageDraw
    draw=ImageDraw.Draw(montage);augmented=v['HybridImages'](split,cfg,True)
    for row,index in enumerate(indices[:8]):
        clean,label=dataset[int(index)];aug,_=augmented[int(index)]
        for col,tensor in enumerate((clean,aug)):
            pixels=((tensor.numpy().transpose(1,2,0)+1)*127.5).clip(0,255).astype(np.uint8);im=v['Image'].fromarray(pixels).resize((384,96));montage.paste(im,(col*400,row*150));draw.text((col*400,row*150+100),f'{label} | '+('clean' if col==0 else 'augmented'),fill='black')
    montage.save(out/f'{split}_inputs.jpg')
    print(json.dumps(dict(split=split,results=report['splits'][split])),flush=True)
assert v['sha'](source)==report['checkpoint_sha256']
v['write'](out/'report.json',report)
