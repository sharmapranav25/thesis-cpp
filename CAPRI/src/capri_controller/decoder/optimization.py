"""Finite optimizer updates with bounded AMP retries for subsequent phases."""
import torch
from .checkpoints import rng_state,restore_rng

MAX_ATTEMPTS=9


def finite_update(model,optimizer,scaler,forward_loss,on_retry=None):
    """Retry the same closure/RNG/buffers; advance optimizer exactly once.

    Scheduler, EMA and sampler advancement belong to the caller after return.
    The closure must not mutate external state or advance a reference sampler.
    No AMP scaler means a nonfinite backward fails immediately.
    """
    rng=rng_state();buffers={k:v.detach().clone() for k,v in model.named_buffers()}
    parameters=[p for p in model.parameters() if p.requires_grad]
    for attempt in range(MAX_ATTEMPTS):
        if attempt:
            restore_rng(rng)
            with torch.no_grad():
                for k,v in model.named_buffers():v.copy_(buffers[k])
        optimizer.zero_grad(set_to_none=True);loss=forward_loss()
        if loss.numel()!=1 or not torch.isfinite(loss):raise FloatingPointError('Nonfinite forward loss')
        if scaler is not None:scaler.scale(loss).backward();scaler.unscale_(optimizer)
        else:loss.backward()
        try:norm=torch.nn.utils.clip_grad_norm_(parameters,1.,error_if_nonfinite=True)
        except RuntimeError as exc:
            if scaler is None or 'non-finite' not in str(exc):raise
            invalid=[name for name,p in model.named_parameters() if p.grad is not None and not torch.isfinite(p.grad).all()]
            if not invalid:raise FloatingPointError('Nonfinite norm with finite gradient elements') from exc
            scale=scaler.get_scale()
            if attempt+1==MAX_ATTEMPTS or scale*.5<1e-8:
                raise FloatingPointError('Persistent nonfinite gradients after bounded retries') from exc
            scaler.update(new_scale=scale*.5);state=scaler.state_dict();state['_growth_tracker']=0;scaler.load_state_dict(state)
            if on_retry:on_retry(dict(attempt=attempt+1,previous_scale=scale,retry_scale=scale*.5,nonfinite_parameters=invalid))
            continue
        if scaler is not None:
            previous=scaler.get_scale();scaler.step(optimizer);scaler.update()
            if scaler.get_scale()<previous:raise FloatingPointError('Unexpected AMP skip after finite gradients')
        else:optimizer.step()
        return loss.detach(),norm.detach(),attempt
    raise AssertionError('Unreachable retry state')
