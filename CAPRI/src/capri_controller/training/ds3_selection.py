"""User-designated evaluation selection; independent of the training curriculum."""
import copy
from .ds3 import ROOT, U


def evaluation_records():
    completed=U.read(ROOT/'completion.json')
    assert completed['all_complete']
    override=ROOT/'checkpoint_selection_override.json'
    if not override.exists():return completed['seeds']
    rule=U.read(override)
    assert rule['selection']=='final_epoch' and rule['epoch']==20
    records=[]
    for original in completed['seeds']:
        seed=original['seed'];assert original['final_epoch']==20
        record=copy.deepcopy(original)
        path=ROOT/f'seed_{seed}/epoch_20.pth'
        digest=U.read(ROOT/f'seed_{seed}/epoch_20.receipt.json')['checkpoint_sha256']
        assert U.sha(path)==digest and digest==original['final_sha256']
        validation=next(r for r in U.read(ROOT/f'seed_{seed}/validation_history.json') if r['epoch']==20)
        record.update(epoch=20,checkpoint=str(path),sha256=digest,metrics=validation,
            selection='Final epoch20 designated by user after seed48 real epoch results were examined',
            original_synthetic_selection=dict(epoch=original['epoch'],checkpoint=original['checkpoint'],sha256=original['sha256']))
        records.append(record)
    return records


def freeze_selection():
    records=evaluation_records()
    receipt=dict(seeds=records,selection=records[0]['selection'])
    path=ROOT/'evaluation_selection.json'
    if path.exists():assert U.read(path)==receipt
    else:U.dump(path,receipt)
    return receipt
