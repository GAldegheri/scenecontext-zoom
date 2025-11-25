
import numpy as np
import pandas as pd
from scipy.stats import norm

import statsmodels.api as sm
import statsmodels.formula.api as smf

from tqdm import tqdm

# ===============================================================
# Utility: SDT transforms
# ===============================================================

def _clip01(p, eps=1e-4):
    """Clip probabilities to an open interval (0,1) to avoid infs in probit."""
    return np.clip(np.asarray(p), eps, 1 - eps)

def dprime_from_rates(H, FA):
    """Compute d' from hit and false-alarm rates (scalar or array)."""
    Hc = _clip01(H)
    FAc = _clip01(FA)
    return norm.ppf(Hc) - norm.ppf(FAc)

def criterion_from_rates(H, FA):
    """Compute criterion (c) from hit and FA (equal-variance SDT)."""
    Hc = _clip01(H)
    FAc = _clip01(FA)
    return -0.5 * (norm.ppf(Hc) + norm.ppf(FAc))

# ===============================================================
# Model specification helpers
# ===============================================================

def prepare_sdt_dataframe(df, cond_col='expected', group_col='p_exp',
                          trial_type_col='trial_type', delta_col='abs_diff',
                          resp_col='resp_diff', subj_col='subject'):
    """
    Ensure required columns exist and create is_diff indicator for the SDT-constrained model.
    Expects:
      - cond: 'A'/'B' (string or category)
      - group: e.g., '25','50','75' (string or category)
      - trial_type: 'same'/'diff'
      - delta: float intensity difference
      - resp_diff: 0/1 (1 if said 'different')
      - subj: subject ID
    Returns a copy with the 'is_diff' numeric column added and categories aligned.
    """
    out = df.copy()
    # Basic column checks
    required = [cond_col, group_col, trial_type_col, delta_col, resp_col, subj_col]
    for c in required:
        if c not in out.columns:
            raise ValueError(f"Missing required column '{c}'")

    # is_diff indicator
    out['is_diff'] = (out[trial_type_col].astype(str).str.lower() == 'diff').astype(int)

    # enforce categorical treatment
    out[cond_col] = pd.Categorical(out[cond_col], categories=[1,0])
    # group can be categorical; keep its observed categories
    out[group_col] = pd.Categorical(out[group_col])

    # rename to standard names internally to simplify formulas
    out = out.rename(columns={cond_col:'cond', group_col:'group',
                              trial_type_col:'trial_type', delta_col:'delta',
                              resp_col:'resp_diff', subj_col:'subj'})
    return out

def sdt_constrained_formula(include_history=False, extra_covariates=None):
    """
    Build the formula string for the SDT-constrained probit model.
    Base components:
      Criterion block (active on SAME): Intercept + C(cond) + C(group)
      Sensitivity block (active on DIFFERENT): is_diff + is_diff:delta + is_diff:delta:C(cond)
    Optional history terms (as additive shifts; e.g., prev_resp, prev_corr, trial_index).
    """
    base = "resp_diff ~ C(cond) + C(group) + is_diff + is_diff:delta + is_diff:delta:C(cond)"
    if include_history and extra_covariates:
        # Add covariates additively; they will mostly shift criterion and/or mild slope
        covs = " + ".join(extra_covariates)
        return base + " + " + covs
    return base

# ===============================================================
# Frequentist fits: GLM (cluster-robust) and GEE (population-averaged)
# ===============================================================

def fit_sdt_probit_glm(df_sdt, formula=None, cluster_col='subj'):
    """
    Fit a Binomial-Probit GLM with cluster-robust SEs by 'cluster_col' (subject).
    Returns the fitted model result.
    """
    if formula is None:
        formula = sdt_constrained_formula()

    model = smf.glm(
        formula=formula,
        data=df_sdt,
        family=sm.families.Binomial(link=sm.families.links.probit())
    )
    res = model.fit(cov_type='cluster', cov_kwds={'groups': df_sdt[cluster_col]})
    return res

def fit_sdt_probit_gee(df_sdt, formula=None, group_col='subj'):
    """
    Fit a population-averaged GEE with Binomial-Probit and independence working correlation.
    Returns the fitted GEE result.
    """
    if formula is None:
        formula = sdt_constrained_formula()
    # Note: statsmodels formula GEE is via smf.gee
    res = smf.gee(
        formula=formula,
        groups=group_col,
        data=df_sdt,
        family=sm.families.Binomial(link=sm.families.links.probit())
    ).fit()
    return res

# ===============================================================
# Standardization over Δ (and groups) to get per-condition H and FA
# ===============================================================

def group_weights(df_sdt):
    """Compute weights for each group level based on frequency (for marginalization)."""
    gcounts = df_sdt['group'].value_counts(normalize=True).sort_index()
    return gcounts.to_dict()

def reference_delta_grid(df_sdt, lo=0.2, hi=0.8, n=7):
    """
    Build a Δ grid using pooled quantiles between lo and hi.
    Default: 7 points from 20th to 80th percentile.
    """
    qs = np.linspace(lo, hi, n)
    return np.quantile(df_sdt['delta'], qs)

def predict_FA(model, cond_label, group_levels, group_wts):
    """
    Predict FA for a condition by averaging SAME-trial predictions across groups.
    For SAME trials, is_diff=0 and delta is irrelevant (set to 0).
    """
    nd = []
    for g in group_levels:
        nd.append(dict(cond=cond_label, group=g, is_diff=0, delta=0.0))
    new = pd.DataFrame(nd)
    # statsmodels returns probabilities for GLM/GEE .predict(new)
    p = model.predict(new)
    # Weighted average over groups
    w = np.array([group_wts[g] for g in new['group']])
    w = w / w.sum()
    return float(np.sum(w * p))

def predict_H_over_grid(model, cond_label, dgrid, group_levels, group_wts):
    """
    Predict H(Δ) for a condition by averaging DIFFERENT-trial predictions
    across the Δ grid (equal weights) and across groups (group_wts).
    """
    rows = []
    for g in group_levels:
        for d in dgrid:
            rows.append(dict(cond=cond_label, group=g, is_diff=1, delta=d))
    new = pd.DataFrame(rows)
    p = model.predict(new)
    # average over delta (equal weights) within each group
    new['pred'] = p
    H_by_group = new.groupby('group')['pred'].mean()
    # then weighted average over groups
    H = 0.0
    for g in group_levels:
        H += group_wts[g] * H_by_group.loc[g]
    return float(H)

def sdt_from_model(model, df_sdt, dgrid=None, cond_levels=('A','B')):
    """
    Compute per-condition (A,B) d′ and criterion by standardizing predictions
    over a common Δ grid (and marginalizing groups by their pooled frequencies).
    Returns a dict with H, FA, d′, criterion for each condition.
    """
    if dgrid is None:
        dgrid = reference_delta_grid(df_sdt)

    # group marginalization weights from pooled data
    gw = group_weights(df_sdt)
    group_levels = list(df_sdt['group'].cat.categories)

    out = {}
    for cond in cond_levels:
        FA = predict_FA(model, cond, group_levels, gw)
        H  = predict_H_over_grid(model, cond, dgrid, group_levels, gw)
        out[cond] = dict(
            H=H, FA=FA,
            dprime=dprime_from_rates(H, FA),
            criterion=criterion_from_rates(H, FA)
        )
    return out

def sdt_cluster_bootstrap(df, B=1000, seed=123):
    rng = np.random.default_rng(seed)
    df_sdt = prepare_sdt_dataframe(df)
    formula = sdt_constrained_formula()
    dgrid = reference_delta_grid(df_sdt)

    # Fit once on full data (point estimate)
    fit0 = fit_sdt_probit_glm(df_sdt, formula=formula, cluster_col='subj')
    est0 = sdt_from_model(fit0, df_sdt, dgrid=dgrid)
    point = {
        'dprime_A': est0['A']['dprime'], 'dprime_B': est0['B']['dprime'],
        'criterion_A': est0['A']['criterion'], 'criterion_B': est0['B']['criterion']
    }

    subs = df_sdt['subj'].unique()
    boot = []
    for _ in tqdm(range(B)):
        resubj = rng.choice(subs, size=len(subs), replace=True)
        dfb = pd.concat([df_sdt[df_sdt['subj']==s] for s in resubj], ignore_index=True)
        fitb = fit_sdt_probit_glm(dfb, formula=formula, cluster_col='subj')
        estb = sdt_from_model(fitb, dfb, dgrid=dgrid)
        boot.append({
            'dA': estb['A']['dprime'], 'dB': estb['B']['dprime'],
            'cA': estb['A']['criterion'], 'cB': estb['B']['criterion'],
            'd_diff': estb['B']['dprime'] - estb['A']['dprime'],
            'c_diff': estb['B']['criterion'] - estb['A']['criterion'],
        })
    boot = pd.DataFrame(boot)

    def ci(x, lo=2.5, hi=97.5):
        return np.percentile(x, [lo, hi])

    out = {
        'point': point,
        'dprime_diff_CI': ci(boot['d_diff']),
        'criterion_diff_CI': ci(boot['c_diff']),
        'p_two_sided_dprime_diff': 2*min((boot['d_diff']>0).mean(), (boot['d_diff']<0).mean()),
        'p_two_sided_criterion_diff': 2*min((boot['c_diff']>0).mean(), (boot['c_diff']<0).mean()),
    }
    return out

# ===============================================================
# Optional: Simple synthetic data generator (for local testing)
# ===============================================================

def simulate_same_diff(
    n_subj=30, trials_per_subj=192, pA=0.5,
    alpha_A=1.0, alpha_B=1.0, c_A=0.0, c_B=0.2,
    groups=('25','50','75'), group_probs=(1/3,1/3,1/3),
    delta_dist=('truncnorm', 0.1, 2.5, 7),  # (kind, lo, hi, ngrid) for pooled Δ
    seed=123
):
    """
    Generate a synthetic dataset resembling same/different with a range of Δ.
    - Each subject gets trials_per_subj trials, half same/different within each condition.
    - Condition assignment per trial follows pA (A-proportion).
    - Evidence model: E ~ N(alpha * Δ, 1) on different; N(0,1) on same.
    - Response: say 'different' if E > c_{cond}.
    - Δ values are drawn from a pooled grid to mimic adaptive sampling.
    """
    rng = np.random.default_rng(seed)
    subj_ids = np.arange(n_subj)
    # pooled delta grid
    kind, lo, hi, n = delta_dist
    if kind == 'truncnorm':
        dgrid = np.linspace(lo, hi, n)
        # sample with more weight near middle to mimic staircase
        w = np.exp(-0.5*((dgrid - (lo+hi)/2)/((hi-lo)/4))**2)
        w = w / w.sum()
    else:
        dgrid = np.linspace(0.1, 2.5, 7)
        w = np.ones_like(dgrid)/len(dgrid)

    rows = []
    for s in subj_ids:
        group = rng.choice(groups, p=np.array(group_probs))
        nA = int(round(pA * trials_per_subj / 2.0) * 2)
        nB = trials_per_subj - nA
        # ensure even within each condition for same/diff split
        if nB % 2 == 1:
            nB -= 1; nA += 1
        trials = (
            [('A','same')]*(nA//2) + [('A','diff')]*(nA//2) +
            [('B','same')]*(nB//2) + [('B','diff')]*(nB//2)
        )
        rng.shuffle(trials)
        for t_idx, (cond, ttype) in enumerate(trials):
            is_diff = int(ttype == 'diff')
            delta = rng.choice(dgrid, p=w) if is_diff else 0.0
            alpha = alpha_A if cond=='A' else alpha_B
            c = c_A if cond=='A' else c_B
            mu = alpha*delta if is_diff else 0.0
            e = rng.normal(mu, 1.0)
            resp_diff = int(e > c)
            rows.append(dict(
                subj=f"S{s:03d}", cond=cond, group=group,
                trial_type=ttype, delta=float(delta),
                resp_diff=resp_diff, trial_index=t_idx+1
            ))
    df = pd.DataFrame(rows)
    # categorical
    df['cond'] = pd.Categorical(df['cond'], categories=['A','B'])
    df['group'] = pd.Categorical(df['group'], categories=list(groups))
    return df

# ===============================================================
# __main__ demo (guarded)
# ===============================================================

if __name__ == "__main__":
    # Small demo with simulated data
    df_raw = simulate_same_diff(n_subj=20, trials_per_subj=160, pA=0.5,
                                alpha_A=1.0, alpha_B=1.1, c_A=0.0, c_B=0.2)
    df_sdt = prepare_sdt_dataframe(df_raw)

    # Formula (you can append history covariates you have, e.g., '+ prev_resp + prev_corr + trial_index')
    formula = sdt_constrained_formula()

    # --- Fit GLM (cluster-robust) ---
    glm_res = fit_sdt_probit_glm(df_sdt, formula=formula, cluster_col='subj')
    print("\n[GLM cluster-robust summary]\n", glm_res.summary())

    # --- Fit GEE (population-averaged) ---
    try:
        gee_res = fit_sdt_probit_gee(df_sdt, formula=formula, group_col='subj')
        print("\n[GEE summary]\n", gee_res.summary())
        model_for_pred = gee_res
    except Exception as e:
        print("GEE fit failed, will use GLM for prediction. Error:", e)
        model_for_pred = glm_res

    # Build a pooled delta grid (20th-80th percentiles)
    dgrid = reference_delta_grid(df_sdt, lo=0.2, hi=0.8, n=7)

    # SDT per condition, standardized over Δ and groups
    out = sdt_from_model(model_for_pred, df_sdt, dgrid=dgrid, cond_levels=('A','B'))
    print("\n[Standardized SDT estimates]")
    for cond, vals in out.items():
        print(cond, vals)
