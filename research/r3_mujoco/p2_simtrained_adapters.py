"""Pure-NPZ inference adapters for frozen Sim-trained inverse-dynamics priors.

This module deliberately has no MuJoCo or sklearn runtime dependency.  It is a
small, auditable bridge between the frozen 21-D [q,dq,ddq] artifacts and the
seven direct-arm torque outputs used by the P2 safety-qualified plant.
"""
from pathlib import Path
import numpy as np


def _load(path):
    return np.load(Path(path), allow_pickle=False)


def _linear(z, coef, intercept):
    return z @ coef.T + intercept


def _poly_features(z, powers):
    # sklearn PolynomialFeatures powers_ convention, with include_bias=False.
    return np.prod(np.power(z[:, None, :], powers[None, :, :]), axis=2)


def predict_npz(path, x):
    """Return finite (N,7) torque predictions from a single safe NPZ artifact."""
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != 21:
        raise ValueError(f"expected (N,21), got {x.shape}")
    with _load(path) as z:
        keys = set(z.files)
        if {'C', 'W', 'width', 'xm', 'xs', 'ym', 'ys'} <= keys:  # frozen RBF
            zn=(x-z['xm'])/z['xs']; d2=((zn[:,None,:]-z['C'][None,:,:])**2).sum(2)
            phi=np.exp(-d2/(2*float(z['width'])**2)); y=np.c_[phi,np.ones(len(x))]@z['W']*z['ys']+z['ym']
            return _finite(y, path)
        if {'W', 'xm', 'xs', 'ym', 'ys'} <= keys:  # frozen near-OLS
            zn = (x-z['xm'])/z['xs']; y = np.c_[zn,np.ones(len(x))] @ z['W']*z['ys']+z['ym']
            return _finite(y, path)
        if {'coefs_0','coefs_1','coefs_2','intercepts_0','intercepts_1','intercepts_2','xm','xs','ym','ys'} <= keys:
            a=(x-z['xm'])/z['xs']
            for i in range(3): a=np.maximum(0., a@z[f'coefs_{i}']+z[f'intercepts_{i}']) if i<2 else a@z[f'coefs_{i}']+z[f'intercepts_{i}']
            return _finite(a*z['ys']+z['ym'], path)
        if {'coef','intercept','xm','xs','poly_powers','fm','fs'} <= keys:
            zn=(x-z['xm'])/z['xs']; f=(_poly_features(zn,z['poly_powers'])-z['fm'])/z['fs']
            return _finite(_linear(f,z['coef'],z['intercept']), path)
        if {'gp_x_train','gp_alpha','xm','xs','ym','ys','length_scale'} <= keys:
            zn=(x-z['xm'])/z['xs']; xt=z['gp_x_train']; ls=z['length_scale']
            k=np.exp(-0.5*(((zn[:,None,:]-xt[None,:,:])/ls)**2).sum(2))
            return _finite((k@z['gp_alpha'])*z['ys']+z['ym'], path)
        if {'C','U','ev','coef','xm','xs','ym','ys','ls'} <= keys:
            zn=(x-z['xm'])/z['xs']; k=np.exp(-.5*(((zn[:,None]-z['C'][None])/float(z['ls']))**2).sum(2));f=k@(z['U']/np.sqrt(z['ev']));return _finite((f@z['coef'])*z['ys']+z['ym'],path)
    raise ValueError(f"unsupported safe model schema: {path}")


def _finite(y, path):
    if y.shape[1:] != (7,) or not np.isfinite(y).all():
        raise ValueError(f"non-finite or invalid prediction from {path}: {y.shape}")
    return y
