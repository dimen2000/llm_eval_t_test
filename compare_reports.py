#!/usr/bin/env python3
"""Compare two JSON accuracy-benchmark reports.

Computes two-sided/one-sided p-values, confidence intervals at two
significance levels and a verdict on the statistical significance of the
difference

Usage:
    python compare_reports.py --ref report1.json --test report2.json \
        --config config.yaml [--json]
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Optional, Tuple

try:
    import yaml
except ImportError as exc:  # pragma: no cover - import guard
    raise SystemExit(
        "PyYAML is required: pip install pyyaml. " f"({exc})"
    ) from exc

from scipy import stats


INF = float("inf")

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------

@dataclass(frozen=True)
class Config:
    alpha_conservative: float = 0.01
    alpha_liberal: float = 0.05

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Config":
        if data is None:
            return cls()
        unknown = set(data) - {
            "alpha_conservative",
            "alpha_liberal",
        }
        if unknown:
            raise ValueError(f"Unknown config keys: {sorted(unknown)}")
        return cls(
            alpha_conservative=float(data.get("alpha_conservative", 0.01)),
            alpha_liberal=float(data.get("alpha_liberal", 0.05)),
        )

    @classmethod
    def from_yaml(cls, path: Optional[str]) -> "Config":
        if path is None:
            return cls()
        with open(path, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        return cls.from_dict(data)


# ----------------------------------------------------------------------------
# Report
# ----------------------------------------------------------------------------

@dataclass(frozen=True)
class Report:
    mean: float
    sem: float
    std: float
    n_repeats: int
    model: str
    name: str
    source: str

    @classmethod
    def from_json(cls, path: str) -> "Report":
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        agg = data.get("aggregate", {})
        mean = agg.get("pass@1")
        sem = agg.get("pass@1_sem")
        std = agg.get("pass@1_std")
        n_repeats = data.get("n_repeats")
        if mean is None:
            raise ValueError(f"{path}: missing aggregate.pass@1")
        if sem is None:
            raise ValueError(f"{path}: missing aggregate.pass@1_sem")
        if std is None:
            raise ValueError(f"{path}: missing aggregate.pass@1_std")
        if n_repeats is None:
            raise ValueError(f"{path}: missing n_repeats")
        if not float(n_repeats) > 1:
            raise ValueError(f"{path}: n_repeats must be > 1 (got {n_repeats})")
        if sem < 0:
            raise ValueError(f"{path}: pass@1_sem cannot be negative ({sem})")
        if std < 0:
            raise ValueError(f"{path}: pass@1_std cannot be negative ({std})")
        return cls(
            mean=float(mean),
            sem=float(sem),
            std=float(std),
            n_repeats=int(n_repeats),
            model=str(data.get("model", "")),
            name=str(data.get("name", "")),
            source=str(path),
        )


# ----------------------------------------------------------------------------
# Core: computations
# ----------------------------------------------------------------------------

@dataclass(frozen=True)
class CI:
    alpha: float
    lower: float
    upper: float
    significant: bool


@dataclass
class Result:
    ref: Report
    test: Report
    diff: float
    se_diff: float
    t_stat: float
    p_two_sided: float
    # One-sided p-values relative to the reference:
    #   p_worse  — H1: test < ref  (worse)
    #   p_better — H1: test > ref  (better)
    p_worse: float
    p_better: float
    cis: Dict[str, CI] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def welch_df(sem_ref: float, sem_test: float, n_ref: int, n_test: int
             ) -> Tuple[float, bool]:
    """Welch–Satterthwaite degrees of freedom.

    Used only to build confidence intervals
    (``scipy.stats.ttest_ind_from_stats`` does not expose df in the summary version).
    The formula is identical to the one scipy uses inside the t-test.
    Returns (df, finite). If df = inf, finite=False.
    """
    a2 = sem_ref * sem_ref
    b2 = sem_test * sem_test
    denom = (a2 ** 2) / (n_ref - 1) + (b2 ** 2) / (n_test - 1)
    if denom == 0:
        return INF, False
    df = (a2 + b2) ** 2 / denom
    if math.isinf(df) or math.isnan(df):
        return INF, False
    return float(df), True


def welch_ttest(ref: Report, test: Report
                ) -> Tuple[float, float, float, float]:
    """Two-sample Welch t-test from summary statistics via
    ``scipy.stats.ttest_ind_from_stats`` (no synthetic samples).

    Returns (t_stat, p_two_sided, p_worse, p_better), where
        p_worse  — H1: test < ref (alternative='less', worse),
        p_better — H1: test > ref (alternative='greater', better).
    """
    common = dict(
        mean1=test.mean, std1=test.std,
        nobs1=test.n_repeats,
        mean2=ref.mean, std2=ref.std,
        nobs2=ref.n_repeats,
        equal_var=False,
    )
    r_two = stats.ttest_ind_from_stats(**common)
    p_worse = float(stats.ttest_ind_from_stats(alternative="less", **common).pvalue)
    p_better = float(stats.ttest_ind_from_stats(alternative="greater", **common).pvalue)
    return float(r_two.statistic), float(r_two.pvalue), p_worse, p_better


def compute_ci(diff: float, se_diff: float, df: float, df_finite: bool,
               alpha: float) -> CI:
    if se_diff == 0.0:
        lower = upper = diff
    else:
        confidence = 1.0 - alpha
        if df_finite:
            lower, upper = stats.t.interval(confidence, df, loc=diff, scale=se_diff)
        else:
            lower, upper = stats.norm.interval(confidence, loc=diff, scale=se_diff)
        lower, upper = float(lower), float(upper)
    contains_zero = (lower <= 0.0 <= upper)
    return CI(alpha=alpha, lower=lower, upper=upper,
              significant=not contains_zero)


def compare(ref: Report, test: Report, cfg: Config) -> Result:
    # diff = test - ref (test is treated as "new", ref as the reference)
    diff = test.mean - ref.mean
    se_diff = math.sqrt(ref.sem ** 2 + test.sem ** 2)

    if se_diff == 0.0 and diff == 0.0:
        # scipy.ttest_ind_from_stats returns nan for 0/0 with equal means;
        # set the mathematically correct values directly.
        t_stat, p_two, p_worse, p_better = 0.0, 1.0, 0.5, 0.5
    else:
        t_stat, p_two, p_worse, p_better = welch_ttest(ref, test)

    df, df_finite = welch_df(ref.sem, test.sem, ref.n_repeats, test.n_repeats)
    cis = {
        "conservative": compute_ci(diff, se_diff, df, df_finite, cfg.alpha_conservative),
        "liberal": compute_ci(diff, se_diff, df, df_finite, cfg.alpha_liberal),
    }

    return Result(
        ref=ref, test=test, diff=diff, se_diff=se_diff,
        t_stat=t_stat,
        p_two_sided=p_two, p_worse=p_worse, p_better=p_better, cis=cis,
    )


# ----------------------------------------------------------------------------
# Output
# ----------------------------------------------------------------------------

def format_text(res: Result) -> str:
    lines: list[str] = []
    lines.append("=== Comparison of accuracy-benchmark reports ===")
    lines.append("")
    lines.append(f"Reference : {res.ref.source}")
    lines.append(f"  model    : {res.ref.model}")
    lines.append(f"  name     : {res.ref.name}")
    lines.append(f"  mean     : {res.ref.mean:.10f}")
    lines.append(f"  sem      : {res.ref.sem:.10f}")
    lines.append(f"  std      : {res.ref.std:.10f}")
    lines.append(f"  n_repeats: {res.ref.n_repeats}")
    lines.append("")
    lines.append(f"Test      : {res.test.source}")
    lines.append(f"  model    : {res.test.model}")
    lines.append(f"  name     : {res.test.name}")
    lines.append(f"  mean     : {res.test.mean:.10f}")
    lines.append(f"  sem      : {res.test.sem:.10f}")
    lines.append(f"  std      : {res.test.std:.10f}")
    lines.append(f"  n_repeats: {res.test.n_repeats}")
    lines.append("")
    lines.append(f"Comparison order: reference = {res.ref.source}")
    lines.append(f"diff (test - ref) : {res.diff:.10f}")
    lines.append(f"SE_diff           : {res.se_diff:.10f}")
    lines.append(f"t-statistic       : {res.t_stat:.6f}")
    lines.append("")
    lines.append(f"p (two-sided)           : {res.p_two_sided:.6g}")
    lines.append(f"p (one-sided, test < ref / worse)  : {res.p_worse:.6g}")
    lines.append(f"p (one-sided, test > ref / better) : {res.p_better:.6g}")
    lines.append("")

    for label in ("conservative", "liberal"):
        ci = res.cis[label]
        lines.append(
            f"CI [{label}, alpha={ci.alpha}] : "
            f"[{ci.lower:.10f}; {ci.upper:.10f}]  "
            f"significant: {'yes' if ci.significant else 'no'}"
        )
    lines.append("")

    # Final verdict
    sig_cons = res.cis["conservative"].significant or res.p_two_sided < res.cis["conservative"].alpha
    sig_lib = res.cis["liberal"].significant or res.p_two_sided < res.cis["liberal"].alpha
    lines.append(f"Verdict (conservative {int((1 - res.cis['conservative'].alpha) * 100)}%): "
                 f"{'significant' if sig_cons else 'not significant'}")
    lines.append(f"Verdict (liberal     {int((1 - res.cis['liberal'].alpha) * 100)}%): "
                 f"{'significant' if sig_lib else 'not significant'}")

    # One-sided verdicts: significant if p < alpha and the sign of diff matches the direction.
    worse_sig = res.p_worse < res.cis["liberal"].alpha and res.diff < 0
    better_sig = res.p_better < res.cis["liberal"].alpha and res.diff > 0
    lines.append(f"One-sided (worse, alpha={res.cis['liberal'].alpha}): "
                 f"{'significant' if worse_sig else 'not significant'}")
    lines.append(f"One-sided (better, alpha={res.cis['liberal'].alpha}): "
                 f"{'significant' if better_sig else 'not significant'}")
    return "\n".join(lines) + "\n"


def _json_num(x: float) -> Optional[float]:
    """Convert ±∞ to None for valid JSON (allow_nan=False)."""
    if math.isinf(x) or math.isnan(x):
        return None
    return x


def format_json(res: Result) -> str:
    d = res.to_dict()
    d["t_stat"] = _json_num(res.t_stat)
    for label, ci in res.cis.items():
        d["cis"][label] = {
            "alpha": ci.alpha,
            "lower": ci.lower,
            "upper": ci.upper,
            "significant": ci.significant,
        }
    # sources/models flattened
    d["ref"] = {
        "source": res.ref.source, "model": res.ref.model, "name": res.ref.name,
        "mean": res.ref.mean, "sem": res.ref.sem, "std": res.ref.std, "n_repeats": res.ref.n_repeats,
    }
    d["test"] = {
        "source": res.test.source, "model": res.test.model, "name": res.test.name,
        "mean": res.test.mean, "sem": res.test.sem, "std": res.test.std, "n_repeats": res.test.n_repeats,
    }
    return json.dumps(d, indent=2, ensure_ascii=False, allow_nan=False) + "\n"


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Compare two JSON accuracy-benchmark reports."
    )
    p.add_argument("--ref", required=True, help="JSON report taken as the reference.")
    p.add_argument("--test", required=True, help="JSON report taken as the test.")
    p.add_argument("--config", default=None, help="YAML config.")
    p.add_argument("--output", "-o", default="./config.yaml", metavar="PATH",
                   help="Write the JSON result to this file.")
    return p


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = Config.from_yaml(args.config)
    ref = Report.from_json(args.ref)
    test = Report.from_json(args.test)
    res = compare(ref, test, cfg)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as fh:
            fh.write(format_json(res))
    sys.stdout.write(format_text(res))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
