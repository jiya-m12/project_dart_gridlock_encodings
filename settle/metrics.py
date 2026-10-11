"""Solution-quality metrics on the payment-selection distribution."""
import numpy as np

from .exact import feasibility_mask, values, is_value


def quality(inst, px):
    """px: probability of each of the 2^n payment selections."""
    feas = feasibility_mask(inst)
    v = values(inst)
    opt = v[feas].max()
    is_opt = feas & is_value(v, opt)
    p_opt = float(px[is_opt].sum())
    uniform_p_opt = is_opt.sum() / len(px)
    return {
        "p_opt": p_opt,
        "p_feasible": float(px[feas].sum()),
        "enrichment": p_opt / uniform_p_opt,                    # vs uniform random sampling
        "exp_ratio": float((px * np.where(feas, v, 0.0)).sum() / opt),   # infeasible counts as 0
        "uniform_p_opt": float(uniform_p_opt),
        "uniform_p_feasible": float(feas.mean()),
    }
