"""Whole-image deformations applied last (after the text): the image is resampled bilinearly, the label maps with nearest
neighbour through the same displacement field, so labels stay aligned. Each records the measured amount it applied
(mean displacement in px, or blur sigma); nothing drawn afterwards can hide it.
"""
import math

import numpy as np
from PIL import Image, ImageFilter

from captcha_effects.effects import effect, _value_noise


def _remap(cv, dx, dy):
    h, w = cv.h, cv.w
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    sx, sy = np.clip(xx + dx, 0, w - 1), np.clip(yy + dy, 0, h - 1)
    x0, y0 = np.floor(sx).astype(int), np.floor(sy).astype(int)
    x1, y1 = np.minimum(x0 + 1, w - 1), np.minimum(y0 + 1, h - 1)
    fx, fy = (sx - x0)[..., None], (sy - y0)[..., None]
    im = cv.img
    cv.img = (im[y0, x0] * (1 - fx) * (1 - fy) + im[y0, x1] * fx * (1 - fy) + im[y1, x0] * (1 - fx) * fy + im[y1, x1] * fx * fy)
    xn, yn = np.rint(sx).astype(int), np.rint(sy).astype(int)
    cv.opaque = cv.opaque[yn, xn]
    cv.translucent = cv.translucent[yn, xn]
    return float(np.mean(np.hypot(dx, dy)))


@effect("wave", "wave_warp", where="post")
def wave(cv, level, rng, **kw):
    if level <= 0:
        return {}
    s = level / 10
    amp = (0.01 + 0.07 * s) * cv.h
    period = rng.uniform(0.25, 0.9) * cv.w
    ph = rng.uniform(0, 2 * math.pi)
    axis = str(rng.choice(["y", "y", "x", "both"]))
    yy, xx = np.mgrid[0:cv.h, 0:cv.w].astype(np.float32)
    dy = amp * np.sin(2 * math.pi * xx / period + ph) if axis in ("y", "both") else np.zeros_like(xx)
    dx = amp * np.sin(2 * math.pi * yy / (period * cv.h / cv.w) + ph) if axis in ("x", "both") else np.zeros_like(xx)
    disp = _remap(cv, dx, dy)
    cv.record(effect="wave", head="wave_warp", layer="post", params=dict(amp=amp, period=period, axis=axis), measured=disp)
    return dict(amp=amp, period=period, axis=axis, measured=disp)


@effect("morph", "morph_distortion", where="post")
def morph(cv, level, rng, **kw):
    if level <= 0:
        return {}
    s = level / 10
    amp = (0.01 + 0.06 * s) * cv.h
    cell = int(rng.choice([6, 10, 16]))
    dx = (_value_noise(rng, cv.h, cv.w, cell) - 0.5) * 2 * amp
    dy = (_value_noise(rng, cv.h, cv.w, cell) - 0.5) * 2 * amp
    disp = _remap(cv, dx, dy)
    cv.record(effect="morph", head="morph_distortion", layer="post", params=dict(amp=amp, cell=cell), measured=disp)
    return dict(amp=amp, cell=cell, measured=disp)


@effect("perspective", "perspective", where="post")
def perspective(cv, level, rng, **kw):
    if level <= 0:
        return {}
    s = level / 10
    k = (0.03 + 0.37 * s) * float(rng.choice([-1, 1]))
    shear = (0.03 + 0.42 * s) * float(rng.choice([-1, 1]))
    yy, xx = np.mgrid[0:cv.h, 0:cv.w].astype(np.float32)
    u, v = xx / cv.w - 0.5, yy / cv.h - 0.5
    dx = shear * v * cv.w * 0.5
    dy = k * u * v * cv.h
    disp = _remap(cv, dx, dy)
    cv.record(effect="perspective", head="perspective", layer="post", params=dict(k=k, shear=shear), measured=disp)
    return dict(k=k, shear=shear, measured=disp)


@effect("blur", "blur", where="post")
def blur(cv, level, rng, **kw):
    if level <= 0:
        return {}
    sigma = 0.3 + 3.4 * level / 10
    im = Image.fromarray(cv.img.clip(0, 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(sigma))
    cv.img = np.asarray(im, np.float32)
    cv.record(effect="blur", head="blur", layer="post", params=dict(sigma=sigma), measured=sigma)
    return dict(sigma=sigma, measured=sigma)


POST_EFFECTS = ["wave", "morph", "perspective", "blur"]
