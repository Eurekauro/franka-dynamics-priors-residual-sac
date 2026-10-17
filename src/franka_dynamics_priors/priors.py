"""Resident, safe-NPZ adapters for frozen Sim-trained priors.

Unlike :mod:`p2_simtrained_adapters`, this module reads a verified NPZ once at
construction.  Its prediction equations are intentionally identical; it is an
I/O and allocation optimisation only.  It never writes model artifacts.
"""
from pathlib import Path
import numpy as np


def _finite(y, name):
    if y.shape[1:] != (7,) or not np.isfinite(y).all():
        raise ValueError(f"non-finite or invalid prediction from {name}: {y.shape}")
    return y


def _poly_features(x, powers):
    """Match sklearn PolynomialFeatures(powers_, include_bias=False) exactly."""
    return np.prod(np.power(x[:, None, :], powers[None, :, :]), axis=2)


class ResidentPredictor:
    def __init__(self, path):
        self.path = Path(path)
        with np.load(self.path, allow_pickle=False) as z:
            self.a = {key: np.asarray(z[key]).copy() for key in z.files}
        self.keys = set(self.a)
        # Precompute the immutable Nyström projection.  This is algebraically
        # identical to k @ (U / sqrt(ev)) at every call.
        if {'C','U','ev','coef','xm','xs','ym','ys','ls'} <= self.keys:
            self.a['projection'] = self.a['U'] / np.sqrt(self.a['ev'])

    def predict(self, x):
        x = np.asarray(x, dtype=np.float64)
        if x.ndim != 2 or x.shape[1] != 21:
            raise ValueError(f"expected (N,21), got {x.shape}")
        a, k = self.a, self.keys
        if {'C','W','width','xm','xs','ym','ys'} <= k:
            zn=(x-a['xm'])/a['xs']; d2=((zn[:,None,:]-a['C'][None,:,:])**2).sum(2)
            return _finite(np.c_[np.exp(-d2/(2*float(a['width'])**2)),np.ones(len(x))] @ a['W'] * a['ys'] + a['ym'], self.path)
        if {'W','xm','xs','ym','ys'} <= k:
            zn=(x-a['xm'])/a['xs']; return _finite(np.c_[zn,np.ones(len(x))] @ a['W'] * a['ys'] + a['ym'], self.path)
        if {'coefs_0','coefs_1','coefs_2','intercepts_0','intercepts_1','intercepts_2','xm','xs','ym','ys'} <= k:
            v=(x-a['xm'])/a['xs']
            for i in range(3):
                v=v@a[f'coefs_{i}']+a[f'intercepts_{i}']
                if i < 2: v=np.maximum(0.,v)
            return _finite(v*a['ys']+a['ym'], self.path)
        if {'coef','intercept','xm','xs','poly_powers','fm','fs'} <= k:
            zn=(x-a['xm'])/a['xs']
            f=(_poly_features(zn,a['poly_powers'])-a['fm'])/a['fs']
            return _finite(f@a['coef'].T+a['intercept'], self.path)
        if {'gp_x_train','gp_alpha','xm','xs','ym','ys','length_scale'} <= k:
            zn=(x-a['xm'])/a['xs']; q=((zn[:,None,:]-a['gp_x_train'][None,:,:])/a['length_scale'])**2
            return _finite(np.exp(-.5*q.sum(2))@a['gp_alpha']*a['ys']+a['ym'], self.path)
        if {'C','U','ev','coef','xm','xs','ym','ys','ls'} <= k:
            zn=(x-a['xm'])/a['xs']; q=((zn[:,None,:]-a['C'][None,:,:])/float(a['ls']))**2
            return _finite((np.exp(-.5*q.sum(2))@a['projection']@a['coef'])*a['ys']+a['ym'], self.path)
        raise ValueError(f"unsupported safe model schema: {self.path}")


def load_resident(path):
    return ResidentPredictor(path)
