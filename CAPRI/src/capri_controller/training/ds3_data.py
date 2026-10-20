"""Frozen Model 3 adaptive rule. All inputs are training-pool measurements."""
import copy
import numpy as np
from torch.utils.data import Sampler

NAMES = ['easy', 'medium', 'hard']


def percentile(values, ids):
    """Deterministic ordinal percentiles; sample ID breaks exact score ties."""
    values = np.asarray(values, dtype=np.float64)
    assert np.isfinite(values).all() and len(values) == len(ids)
    order = np.lexsort((ids, values))
    result = np.empty(len(values), dtype=np.float64)
    result[order] = (np.arange(len(values)) + .5) / len(values)
    return result


class AdaptiveSampler(Sampler):
    def __init__(self, rows, seed, callback=None):
        self.ids = np.array([r['index'] for r in rows], dtype=np.int64)
        assert len(set(self.ids)) == len(self.ids)
        self.raw = np.array([r['nll'] for r in rows], dtype=np.float64)
        self.normalized = percentile(self.raw, self.ids)
        self.smooth = self.normalized.copy()
        self.buckets = np.array([NAMES.index(r['bucket']) for r in rows])
        self.mass = np.array([1., 0., 0.])
        self.epoch = 1
        self.seed = seed
        self.callback = callback
        self.last_draw = None
        self.check()

    def __len__(self):
        return len(self.ids)

    def check(self):
        assert np.isfinite(self.raw).all() and np.isfinite(self.smooth).all()
        assert all(np.any(self.buckets == i) for i in range(3))
        assert np.isfinite(self.mass).all() and np.all(self.mass >= 0)
        assert abs(self.mass.sum() - 1) < 1e-12
        if self.epoch > 1:
            assert self.mass.min() >= .2 - 1e-12 and self.mass.max() <= .5 + 1e-12

    def probabilities(self):
        return np.array([self.mass[b] / np.sum(self.buckets == b) for b in range(3)])[self.buckets]

    def __iter__(self):
        self.check()
        counts = np.rint(len(self) * self.mass).astype(int)
        counts[np.argmax(self.mass)] += len(self) - counts.sum()
        # Same independent sampling stream as DS2, explicitly reproducible at boundaries.
        rng = np.random.default_rng(self.seed + self.epoch * 100003)
        before = copy.deepcopy(rng.bit_generator.state)
        draws = np.concatenate([rng.choice(np.flatnonzero(self.buckets == b), n, replace=True)
                                for b, n in enumerate(counts) if n])
        rng.shuffle(draws)
        used = draws[:len(draws) // 128 * 128]
        self.last_draw = dict(epoch=self.epoch, mass=self.mass.tolist(), intended_draws=counts.tolist(),
                              actual_draws=np.bincount(self.buckets[draws], minlength=3).tolist(),
                              optimized_draws=np.bincount(self.buckets[used], minlength=3).tolist(),
                              dropped=len(draws)-len(used), rng_before=before,
                              rng_after=copy.deepcopy(rng.bit_generator.state))
        if self.callback:
            self.callback(self.last_draw, self.ids[draws])
        self.epoch += 1
        return iter(draws.tolist())

    def update(self, raw):
        raw = np.asarray(raw, dtype=np.float64)
        assert raw.shape == self.raw.shape and np.isfinite(raw).all() and np.all(raw >= 0)
        old = self.buckets.copy()
        self.raw = raw
        self.normalized = percentile(raw, self.ids)
        self.smooth = .5 * self.smooth + .5 * self.normalized
        order = np.lexsort((self.ids, self.smooth))
        for b, group in enumerate(np.array_split(order, 3)):
            self.buckets[group] = b
        # Bounded error proxy for a complete synthetic string. No validation inputs.
        risk = -np.expm1(-np.minimum(raw, 50))
        group_risk = np.array([risk[self.buckets == b].mean() for b in range(3)])
        proposal = .2 + .4 * ((group_risk + 1e-6) / (group_risk.sum() + 3e-6))
        # Cap .5 and redistribute excess equally among remaining groups.
        for _ in range(3):
            excess = np.maximum(proposal - .5, 0).sum()
            proposal = np.minimum(proposal, .5)
            free = proposal < .5 - 1e-14
            if excess > 0:
                proposal[free] += excess / free.sum()
        # First transition is exempt from slew limit to restore all-pool coverage.
        previous = np.full(3, 1/3) if self.epoch == 2 else self.mass.copy()
        delta = proposal - previous
        alpha = min(1., .1 / max(np.abs(delta).max(), 1e-15))
        self.mass = previous + alpha * delta
        self.check()
        return dict(group_risk=group_risk.tolist(), proposal=proposal.tolist(),
                    next_mass=self.mass.tolist(), moved=int(np.sum(old != self.buckets)),
                    max_mass_change=float(np.max(np.abs(self.mass-previous))))

    def state_dict(self):
        return copy.deepcopy({k: v for k, v in vars(self).items() if k != 'callback'})

    def load_state_dict(self, state):
        assert np.array_equal(self.ids, state['ids']) and self.seed == state['seed']
        for k, v in copy.deepcopy(state).items():
            setattr(self, k, v)
        self.check()
