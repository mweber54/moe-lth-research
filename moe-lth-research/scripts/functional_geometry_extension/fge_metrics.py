"""Pure numerical routines (float64) for activation, subspace and gradient geometry."""
import math
import torch

EPS_REL = 1e-10


def center(x):
    return x - x.mean(0, keepdim=True)


def cka_centered(x, y):
    """Centered linear CKA over observations (rows). None when a centered norm is zero."""
    x = center(x.double())
    y = center(y.double())
    hsic_xy = (x.T @ y).square().sum()
    hsic_xx = (x.T @ x).square().sum()
    hsic_yy = (y.T @ y).square().sum()
    den = (hsic_xx * hsic_yy).sqrt()
    if float(x.square().sum()) == 0.0 or float(y.square().sum()) == 0.0 or float(den) == 0.0:
        return None
    return float(hsic_xy / den)


def spectrum(x):
    """Centered feature-space eigendecomposition. Returns (lam desc, V desc, n_obs, total energy)."""
    xc = center(x.double())
    cov = xc.T @ xc
    lam, vec = torch.linalg.eigh(cov)
    lam = lam.flip(0).clamp_min(0)
    vec = vec.flip(1)
    return dict(lam=lam, vec=vec, xc=xc, n=x.shape[0], total=float(lam.sum()))


def rank_status(sp, k):
    """Validity of a rank-k subspace: numerical rank, N-1 limit, cutoff degeneracy."""
    lam = sp['lam']
    lam1 = float(lam[0]) if len(lam) else 0.0
    if lam1 <= 0:
        return dict(valid=False, reason='zero_energy', numerical_rank=0, degenerate=False)
    numerical_rank = int((lam > EPS_REL * lam1).sum())
    limit = min(numerical_rank, sp['n'] - 1)
    if k > limit:
        return dict(valid=False, reason='rank_exceeds_numerical_or_N_minus_1', numerical_rank=numerical_rank, degenerate=False)
    degenerate = bool(k < len(lam) and float(lam[k]) > 0 and float(lam[k - 1]) / float(lam[k]) < 1 + 1e-6)
    return dict(valid=True, reason='ok', numerical_rank=numerical_rank, degenerate=degenerate)


def retained_energy(sp, k):
    return float(sp['lam'][:k].sum() / sp['total']) if sp['total'] > 0 else None


def feature_overlap(sa, sb, k):
    """Mean squared cosine of principal angles between top-k feature-space subspaces in R^d."""
    m = sa['vec'][:, :k].T @ sb['vec'][:, :k]
    sv = torch.linalg.svdvals(m)
    return float(sv.square().sum() / k), float(sv.min())


def sample_overlap(sa, sb, k):
    """Same, for the top-k left singular (sample-space) subspaces in R^N."""
    def basis(s):
        lam = s['lam'][:k].clamp_min(1e-300)
        return (s['xc'] @ s['vec'][:, :k]) / lam.sqrt()[None, :]
    m = basis(sa).T @ basis(sb)
    sv = torch.linalg.svdvals(m)
    return float(sv.square().sum() / k), float(sv.min())


def pair_vector_stats(a, b):
    """Complete-output comparison on identical inputs: pooled relative error, mean per-row cosine."""
    a = a.double()
    b = b.double()
    nb = float(b.norm())
    rel = float((a - b).norm() / nb) if nb > 0 else None
    na = a.norm(dim=1)
    nbr = b.norm(dim=1)
    ok = (na > 0) & (nbr > 0)
    cos = float(((a * b).sum(1)[ok] / (na[ok] * nbr[ok])).mean()) if int(ok.sum()) else None
    return dict(relative_error=rel, mean_cosine=cos, n_valid=int(ok.sum()), zero_exclusions=int((~ok).sum()))


def energy_retention(g, m):
    """||M*g||^2 / ||g||^2; None when g is zero (undefined)."""
    g = g.double()
    den = float(g.square().sum())
    if den == 0.0:
        return None
    return float((g * m.double()).square().sum() / den)


def cosine(a, b):
    a = a.double().flatten()
    b = b.double().flatten()
    den = float(a.norm() * b.norm())
    return float((a * b).sum() / den) if den > 0 else None


def gram_subspace_overlap(ka, kb, kab, r):
    """Exact mean squared principal-angle cosine of the top-r row spaces of two gradient sets.

    ka = Xa Xa^T, kb = Xb Xb^T, kab = Xa Xb^T (n x n). Uncentered. Returns (overlap, status).
    """
    la, va = torch.linalg.eigh(ka.double())
    lb, vb = torch.linalg.eigh(kb.double())
    la, va = la.flip(0), va.flip(1)
    lb, vb = lb.flip(0), vb.flip(1)
    if float(la[0]) <= 0 or float(lb[0]) <= 0:
        return None, 'zero_gradient'
    ra = int((la > EPS_REL * la[0]).sum())
    rb = int((lb > EPS_REL * lb[0]).sum())
    if r > min(ra, rb):
        return None, f'rank_exceeds_numerical_rank({ra},{rb})'
    qa = va[:, :r] / la[:r].sqrt()[None, :]
    qb = vb[:, :r] / lb[:r].sqrt()[None, :]
    m = qa.T @ kab.double() @ qb
    sv = torch.linalg.svdvals(m)
    return float(sv.square().sum() / r), 'ok'
