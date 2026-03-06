"""
Generate kpi_demo.csv with:
  - Independent per-metric performance per agency (so weights actually change rankings)
  - Per-LOB range capped at 0.40 for all TOPSIS ratio metrics
  - Realistic activity counts (Submitted, Quoted, Bound, PoliciesActive, etc.)
  - All constraints: LR < 1, Quoted <= Submitted, Bound <= Quoted,
    Renewals <= EligibleForRenewal, PoliciesActive > 0

Industry Standards 2024 (P&C Data Model 2.xlsx):
  Workers Comp   : LR 45-50%, ER 23-26%
  Commercial Auto: LR 70-80%, ER 20-30%
  BOP            : LR 55-65%, ER 27-35%
"""
import pandas as pd
import numpy as np

np.random.seed(42)

df = pd.read_csv("kpi_upload.csv")

# ── 1. Per-LOB metric ranges (max - min <= 0.40 for all ratio metrics) ────────
# Format: (min, max) for each metric
# LR: lower = better  |  ER: lower = better
# RR: higher = better |  SQ: higher = better
LOB_RANGES = {
    "Workers Comp": {
        "LossRatio":         (0.38, 0.74),   # range 0.36, centered on industry ~50%
        "ExpenseRatio":      (0.19, 0.34),   # range 0.15, industry 23-26%
        "RetentionRate":     (0.56, 0.95),   # range 0.39
        "SubmissionQuality": (0.20, 0.60),   # range 0.40
    },
    "Commercial Auto": {
        "LossRatio":         (0.60, 0.96),   # range 0.36, industry 70-80%; cap 0.96
        "ExpenseRatio":      (0.18, 0.38),   # range 0.20, industry 20-30%
        "RetentionRate":     (0.52, 0.92),   # range 0.40
        "SubmissionQuality": (0.18, 0.58),   # range 0.40
    },
    "Business Owners Policy": {
        "LossRatio":         (0.46, 0.86),   # range 0.40, industry ~55-65%
        "ExpenseRatio":      (0.24, 0.50),   # range 0.26, industry 27-35%
        "RetentionRate":     (0.52, 0.92),   # range 0.40
        "SubmissionQuality": (0.18, 0.58),   # range 0.40
    },
}

# ── 2. Size tiers for activity volumes (independent of ratio performance) ──────
# Larger agencies have more policies and higher EP — but ratio performance is independent
SIZE_TIERS = [0.15, 0.25, 0.30, 0.20, 0.10]   # 1=large ... 5=small
agent_codes = df["AgentCode"].unique()
size_tiers  = np.random.choice([1, 2, 3, 4, 5], size=len(agent_codes), p=SIZE_TIERS)
size_map    = dict(zip(agent_codes, size_tiers))

# (mean, std) for active policies and premium per policy per month, by size-tier and LOB
SIZE_PARAMS = {
    "Workers Comp": {
        # pp_monthly = earned premium per active policy per month
        1: dict(pol=(130,30), pp=(725,85), sub=(28,7),  qr=(.80,.04), br=(.62,.04), cp=.11),
        2: dict(pol=( 82,20), pp=(705,75), sub=(19,6),  qr=(.74,.04), br=(.54,.04), cp=.11),
        3: dict(pol=( 48,13), pp=(685,65), sub=(13,5),  qr=(.66,.04), br=(.46,.04), cp=.10),
        4: dict(pol=( 26, 8), pp=(655,55), sub=( 8,3),  qr=(.56,.04), br=(.38,.04), cp=.09),
        5: dict(pol=( 13, 5), pp=(625,45), sub=( 4,2),  qr=(.44,.04), br=(.28,.04), cp=.08),
    },
    "Commercial Auto": {
        1: dict(pol=(110,28), pp=(625,75), sub=(22,6),  qr=(.78,.04), br=(.60,.04), cp=.12),
        2: dict(pol=( 70,18), pp=(605,65), sub=(15,5),  qr=(.70,.04), br=(.52,.04), cp=.11),
        3: dict(pol=( 40,11), pp=(582,55), sub=(10,4),  qr=(.62,.04), br=(.44,.04), cp=.10),
        4: dict(pol=( 22, 7), pp=(555,45), sub=( 6,3),  qr=(.52,.04), br=(.36,.04), cp=.09),
        5: dict(pol=( 11, 4), pp=(525,35), sub=( 3,2),  qr=(.40,.04), br=(.26,.04), cp=.08),
    },
    "Business Owners Policy": {
        1: dict(pol=(260,55), pp=(198,38), sub=(35,9),  qr=(.79,.04), br=(.61,.04), cp=.12),
        2: dict(pol=(165,38), pp=(188,32), sub=(24,7),  qr=(.72,.04), br=(.53,.04), cp=.11),
        3: dict(pol=( 95,24), pp=(177,27), sub=(16,5),  qr=(.64,.04), br=(.45,.04), cp=.10),
        4: dict(pol=( 52,13), pp=(164,22), sub=( 9,4),  qr=(.54,.04), br=(.37,.04), cp=.09),
        5: dict(pol=( 26, 8), pp=(150,18), sub=( 5,2),  qr=(.42,.04), br=(.27,.04), cp=.08),
    },
}

# ── 3. Agency-level ratio baselines: INDEPENDENTLY drawn per metric ────────────
# Each agency-LOB pair gets its own score [0,1] for each ratio metric,
# drawn from a uniform distribution — completely independent across metrics.
# This ensures that changing weights actually changes rankings.
agency_ratio_base = {}
for (ac, lob), _ in df.groupby(["AgentCode", "LOB"]):
    seed = int(ac) * 13 + abs(hash(lob)) % 88883
    np.random.seed(seed)
    r = LOB_RANGES[lob]

    # Draw independent [0,1] scores for each metric
    lr_u  = np.random.uniform(0, 1)   # 0=best(low LR), 1=worst(high LR)
    er_u  = np.random.uniform(0, 1)   # 0=best(low ER), 1=worst(high ER)
    rr_u  = np.random.uniform(0, 1)   # 0=worst(low RR), 1=best(high RR)
    sq_u  = np.random.uniform(0, 1)   # 0=worst(low SQ), 1=best(high SQ)

    # Map to LOB-specific ranges
    lr_base = r["LossRatio"][0]         + lr_u * (r["LossRatio"][1]         - r["LossRatio"][0])
    er_base = r["ExpenseRatio"][0]      + er_u * (r["ExpenseRatio"][1]      - r["ExpenseRatio"][0])
    rr_base = r["RetentionRate"][0]     + rr_u * (r["RetentionRate"][1]     - r["RetentionRate"][0])
    sq_base = r["SubmissionQuality"][0] + sq_u * (r["SubmissionQuality"][1] - r["SubmissionQuality"][0])

    agency_ratio_base[(ac, lob)] = dict(lr=lr_base, er=er_base, rr=rr_base, sq=sq_base)

# ── 4. Agency-level size baselines ───────────────────────────────────────────
size_base_cache = {}
for (ac, lob), _ in df.groupby(["AgentCode", "LOB"]):
    seed = int(ac) * 7 + abs(hash(lob)) % 99991
    np.random.seed(seed)
    sz = size_map[ac]
    sp = SIZE_PARAMS[lob][sz]
    pol_base = max(5, int(np.random.normal(sp["pol"][0], sp["pol"][1])))
    pp_base  = max(80.0, float(np.random.normal(sp["pp"][0], sp["pp"][1])))
    size_base_cache[(ac, lob)] = (pol_base, pp_base)

# ── 5. Row-level generation ───────────────────────────────────────────────────
rows = []
for idx, row in df.iterrows():
    ac  = row["AgentCode"]
    lob = row["LOB"]
    sz  = size_map[ac]
    sp  = SIZE_PARAMS[lob][sz]
    rb  = agency_ratio_base[(ac, lob)]
    pol_base, pp_base = size_base_cache[(ac, lob)]

    np.random.seed(int(ac) * 31 + idx)

    # ── Active policies (±6% monthly variation) ──
    pol_active = max(3, int(pol_base * np.random.uniform(0.94, 1.06)))

    # ── Earned premium ──
    ep = max(500.0, pol_active * pp_base * np.random.uniform(0.94, 1.06))

    # ── Ratios: agency baseline ± small monthly noise (±2-3%) ──
    noise = 0.025
    lr = float(np.clip(rb["lr"] + np.random.uniform(-noise, noise), 0.26, 0.97))
    er = float(np.clip(rb["er"] + np.random.uniform(-noise * 0.6, noise * 0.6), 0.14, 0.58))
    rr = float(np.clip(rb["rr"] + np.random.uniform(-noise, noise), 0.30, 0.97))
    sq = float(np.clip(rb["sq"] + np.random.uniform(-noise, noise), 0.05, 0.95))

    # ── Submission pipeline: derive from sq and size ──
    sub_count = max(1, int(np.random.normal(sp["sub"][0], sp["sub"][1])))
    # Reverse-engineer quote/bind rates from sq target
    qr = float(np.clip(np.random.normal(sp["qr"][0], sp["qr"][1]), 0.25, 0.97))
    # bind rate from quoted = sq / qr (so bound/submitted = sq)
    br_target = sq / qr if qr > 0 else sq
    br = float(np.clip(br_target + np.random.uniform(-0.03, 0.03), 0.05, 0.97))

    quoted = max(0, min(sub_count, round(sub_count * qr)))
    bound  = max(0, min(quoted,    round(quoted    * br)))

    # Recompute SQ and derived ratios from actual counts
    actual_sq = round(bound / sub_count, 4) if sub_count > 0 else 0.0
    actual_hr = round(bound / quoted,    4) if quoted   > 0 else 0.0
    actual_cr = actual_sq

    # ── DeclinedWithdrawn ──
    dw = sub_count - quoted
    dw_reason = ""
    if dw > 0:
        dw_reason = np.random.choice(["Declined", "Withdrawn"], p=[0.55, 0.45])

    # ── Renewal pipeline ──
    elig     = max(1, round(pol_active / 12))
    renewals = max(0, min(elig, round(elig * rr * np.random.uniform(0.96, 1.04))))

    # ── Other counts ──
    new_count     = bound
    rewrite_count = max(0, int(np.random.poisson(max(0.1, pol_active * 0.015))))

    # ── Financials ──
    incurred = round(ep * lr, 2)
    ca       = round(ep * sp["cp"] * np.random.uniform(0.88, 1.12), 2)
    cwp      = round(ep * np.random.uniform(1.02, 1.08), 2)
    ep_utility = round(er * np.random.uniform(0.96, 1.04), 6)

    r = row.copy()
    r["LossRatio"]                = round(lr, 4)
    r["ExpenseRatio"]             = round(er, 4)
    r["RetentionRate"]            = round(rr, 4)
    r["SubmissionQuality"]        = actual_sq
    r["HitRatio"]                 = actual_hr
    r["CloseRatio"]               = actual_cr
    r["EarnedPremium"]            = round(ep, 2)
    r["IncurredLoss"]             = incurred
    r["CommissionAmount"]         = ca
    r["CommissionWrittenPremium"] = cwp
    r["PoliciesActive"]           = pol_active
    r["NewCount"]                 = new_count
    r["RewriteCount"]             = rewrite_count
    r["Submitted"]                = sub_count
    r["Quoted"]                   = quoted
    r["Bound"]                    = bound
    r["EligibleForRenewal"]       = elig
    r["Renewals"]                 = renewals
    r["DeclinedWithdrawn"]        = dw
    r["DeclinedWithdrawnReason"]  = dw_reason
    r["EP_utility_peer"]          = ep_utility
    rows.append(r)

out = pd.DataFrame(rows)
out.to_csv("Files/kpi_demo.csv", index=False)
print(f"Generated {len(out)} rows | {out['AgentCode'].nunique()} agent-codes")

# ── 6. Constraint checks ──────────────────────────────────────────────────────
assert (out["LossRatio"]   <= 0.97).all(),                         "LossRatio > 0.97!"
assert (out["Quoted"]      <= out["Submitted"]).all(),             "Quoted > Submitted!"
assert (out["Bound"]       <= out["Quoted"]).all(),                "Bound > Quoted!"
assert (out["Renewals"]    <= out["EligibleForRenewal"]).all(),    "Renewals > EligibleForRenewal!"
assert (out["PoliciesActive"] > 0).all(),                          "PoliciesActive = 0!"
print("All constraints passed.")

# ── 7. Aggregated range check (what TOPSIS sees after groupby mean) ───────────
agg = out.groupby("AgentCode")[
    ["LossRatio", "ExpenseRatio", "RetentionRate", "SubmissionQuality"]
].mean()

print("\n=== Aggregated metric ranges (what TOPSIS sees) ===")
for col in ["LossRatio", "ExpenseRatio", "RetentionRate", "SubmissionQuality"]:
    lo, hi = agg[col].min(), agg[col].max()
    print(f"  {col:25s}: [{lo:.3f}, {hi:.3f}]  range={hi-lo:.3f}")

# ── 8. Correlation check (should be low between metrics) ─────────────────────
print("\n=== Metric correlations (want these LOW so weights matter) ===")
print(agg.corr().round(2).to_string())
