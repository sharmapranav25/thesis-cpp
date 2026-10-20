"""Iteration 2 baseline training (configs/iter2_baseline.yaml).

  python -m feature_bank.iter2.train --config configs/iter2_baseline.yaml --out runs/<name> [--epochs N]
      [--train-limit N] [--resume]

Every epoch: raw + EMA checkpoints, per-head training losses, EMA scored on steering / selection / generalisation
(metrics.scores), one line appended to history.jsonl, status.json rewritten. Checkpoint choice happens once, on
selection, after training (choose.py / the report), never on generalisation.
"""
import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from feature_bank.iter2 import data as D
from feature_bank.iter2 import losses as Lo
from feature_bank.iter2 import metrics as M
from feature_bank.iter2.model import Iter2Model, param_counts


def to_dev(nb, device):
    b = {k: torch.from_numpy(np.ascontiguousarray(v)).to(device, non_blocking=True) for k, v in nb.items()}
    b["image"] = b["image"].float().div_(255.0)
    return b


def lr_at(step, total, cfg):
    if step < cfg["warmup_steps"]:
        return cfg["lr"] * (step + 1) / cfg["warmup_steps"]
    p = (step - cfg["warmup_steps"]) / max(1, total - cfg["warmup_steps"])
    return cfg["min_lr"] + 0.5 * (cfg["lr"] - cfg["min_lr"]) * (1 + math.cos(math.pi * min(1.0, p)))


class EMA:
    def __init__(self, model, decay):
        self.decay, self.t = decay, 0
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model):
        d = min(self.decay, (1 + self.t) / (10 + self.t))
        self.t += 1
        for k, v in model.state_dict().items():
            if v.dtype.is_floating_point:
                self.shadow[k].mul_(d).add_(v.detach(), alpha=1 - d)
            else:
                self.shadow[k].copy_(v)


def leak_check(train_sets, fixed_sets):
    tr = set()
    for s in train_sets:
        tr |= set(s.meta["sha256"])
    bad = {s.name: len(tr & set(s.meta["sha256"])) for s in fixed_sets}
    assert not any(bad.values()), f"training images found in fixed sets: {bad}"
    return bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--train-limit", type=int, default=None, help="use only the first N training images (smoke)")
    ap.add_argument("--score-limit", type=int, default=None, help="score on the first N images of each fixed set")
    ap.add_argument("--resume", action="store_true")
    a = ap.parse_args()
    cfg = yaml.safe_load(Path(a.config).read_text(encoding="utf-8"))
    tc = cfg["train"]
    epochs = a.epochs or tc["epochs"]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.yaml").write_text(yaml.safe_dump(cfg), encoding="utf-8")
    device = torch.device("cuda")
    torch.cuda.set_per_process_memory_fraction(tc.get("gpu_memory_fraction", 0.8))
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    train_sets = D.open_split(cfg["data"], "train")
    fixed = {k: D.open_split(cfg["data"], k) for k in cfg["score"]["every_epoch"]}
    leaks = leak_check(train_sets, [s for v in fixed.values() for s in v])
    items, widths = D.items_of(train_sets)
    if a.train_limit:
        items, widths = items[:a.train_limit], widths[:a.train_limit]
    if a.score_limit:
        for v in fixed.values():
            for s in v:
                s.ix = {k: x[:a.score_limit] for k, x in s.ix.items()}
    meta0 = train_sets[0].meta
    eff_heads, txt_heads = meta0["effect_heads"], meta0["text_heads"]
    lab_eff = np.concatenate([s.ix["eff_level"] for s in train_sets])
    lab_txt = np.concatenate([s.ix["txt_level"] for s in train_sets])
    eff_w = Lo.level_weights(lab_eff).to(device) if tc.get("level_balance") else None
    txt_w = Lo.level_weights(lab_txt).to(device) if tc.get("level_balance") else None

    torch.manual_seed(0)
    model = Iter2Model(n_effects=len(eff_heads), n_text=len(txt_heads), **cfg["model"]).to(device)
    if tc.get("loss_weighting", "fixed") != "fixed":
        raise ValueError('This baseline requires fixed loss weights')
    uw = Lo.FixedWeights().to(device)
    decay, no_decay = [], []
    for n, p in list(model.named_parameters()) + [("uw." + n, p) for n, p in uw.named_parameters()]:
        (no_decay if p.ndim < 2 or n.startswith("uw.") or "bias" in n or "emb" in n or "pos" in n else decay).append(p)
    opt = torch.optim.AdamW([dict(params=decay, weight_decay=tc["weight_decay"]), dict(params=no_decay, weight_decay=0)],
                            lr=tc["lr"], betas=(0.9, 0.95))
    ema = EMA(model, tc["ema"])
    rng = np.random.default_rng(0)
    steps_per_epoch = len(D.width_batches(items, widths, tc["batch"], np.random.default_rng(1)))
    total = steps_per_epoch * epochs
    start_epoch, step = 1, 0
    if a.resume and (out / "last.pt").exists():
        ck = torch.load(out / "last.pt", map_location=device, weights_only=False)
        model.load_state_dict(ck["model"]); uw.load_state_dict(ck["uw"]); opt.load_state_dict(ck["opt"])
        ema.shadow, ema.t = ck["ema"], ck["ema_t"]
        start_epoch, step = ck["epoch"] + 1, ck["step"]
        rng = np.random.default_rng(ck["epoch"])
        print(f"resumed after epoch {ck['epoch']}", flush=True)
    info = dict(train_images=len(items), steps_per_epoch=steps_per_epoch, epochs=epochs, params=param_counts(model),
                leaks=leaks, eff_heads=eff_heads, txt_heads=txt_heads,
                fixed={k: sum(len(s) for s in v) for k, v in fixed.items()})
    (out / "run_info.json").write_text(json.dumps(info, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in info.items() if k not in ("eff_heads", "txt_heads")}), flush=True)

    for epoch in range(start_epoch, epochs + 1):
        model.train()
        t0 = time.time()
        batches = D.width_batches(items, widths, tc["batch"], rng)
        acc = {g: 0.0 for g in Lo.GROUPS}
        head_acc = {k: torch.zeros(n, device=device) for k, n in
                    [("eff_presence", len(eff_heads)), ("eff_ordinal", len(eff_heads)), ("txt_presence", len(txt_heads)),
                     ("txt_ordinal", len(txt_heads))]}
        head_n = {k: torch.zeros_like(v) for k, v in head_acc.items()}
        nb = 0
        for k, cb in enumerate(D.prefetch(batches)):
            b = to_dev(cb, device)
            for gr in opt.param_groups:
                gr["lr"] = lr_at(step, total, tc)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                o = model(b["image"], b["cols"])
            g, per = Lo.compute(o, b, eff_w, txt_w)
            loss = uw(g)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), tc["grad_clip"])
            opt.step()
            ema.update(model)
            step += 1
            nb += 1
            for gname in Lo.GROUPS:
                acc[gname] += float(g[gname].detach())
            for hk in head_acc:
                l, n = per[hk]
                head_acc[hk] += l.detach() * (n > 0)
                head_n[hk] += (n > 0).float()
            if k % 100 == 0:
                print(f"epoch {epoch} step {k}/{len(batches)} loss {float(loss):.4f} "
                      + " ".join(f"{gname}={float(g[gname]):.3f}" for gname in Lo.GROUPS)
                      + f" {time.time() - t0:.0f}s", flush=True)
        train_time = time.time() - t0
        ed = out / f"epoch_{epoch:02d}"
        ed.mkdir(exist_ok=True)
        torch.save(model.state_dict(), ed / "raw.pt")
        torch.save(ema.shadow, ed / "ema.pt")
        torch.save(dict(model=model.state_dict(), uw=uw.state_dict(), opt=opt.state_dict(), ema=ema.shadow, ema_t=ema.t,
                        epoch=epoch, step=step), out / "last.pt")
        # score the EMA weights
        live = {k: v.clone() for k, v in model.state_dict().items()}
        model.load_state_dict(ema.shadow)
        sc = {}
        t1 = time.time()
        for name, sets in fixed.items():
            P, L = M.predict(model, sets, device)
            sc[name] = M.scores(P, L, eff_heads, txt_heads)
            torch.cuda.empty_cache()
        model.load_state_dict(live)
        heads_loss = {hk: {h: round(float(v), 4) for h, v in zip(eff_heads if hk.startswith("eff") else txt_heads,
                                                                 (head_acc[hk] / head_n[hk].clamp(min=1)).tolist())}
                      for hk in head_acc}
        rec = dict(epoch=epoch, step=step, train_seconds=round(train_time), score_seconds=round(time.time() - t1),
                   loss={gname: round(acc[gname] / max(1, nb), 4) for gname in Lo.GROUPS},
                   overall_loss=round(sum(acc.values()) / max(1, nb), 4),
                   loss_weighting="fixed",
                   uncertainty={gname: round(float(v), 3) for gname, v in zip(Lo.GROUPS, uw.log_var.tolist())},
                   lr=lr_at(step, total, tc), summary={k: v["summary"] for k, v in sc.items()})
        (ed / "scores.json").write_text(json.dumps(dict(record=rec, scores=sc, head_loss=heads_loss), indent=1),
                                        encoding="utf-8")
        with open(out / "history.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
        (out / "status.json").write_text(json.dumps(dict(epoch=epoch, of=epochs, epoch_seconds=round(time.time() - t0),
                                                         finished=time.strftime("%Y-%m-%d %H:%M:%S")), indent=1))
        print("EPOCH", json.dumps(rec), flush=True)


if __name__ == "__main__":
    main()
