"""
Diagnose (and optionally repair) a one-category shift in Axis-A human annotations.

Symptom this targets: Axis-A kappa near 0 with the confusion mass sitting on an
off-diagonal, while Axis-B kappa is healthy. That pattern means the stored
`human_awareness` labels are systematically offset from `machine_awareness` — NOT a
math bug (cohen_kappa is string-keyed and order-invariant) and NOT a benchmark problem
(the machine labels are intact). So it is fixable WITHOUT re-running the benchmark.

Two distinct causes, distinguished here:

  1. MECHANICAL re-index. The two Axis-A orderings differ — the Judge/config canonical
     order (config.AWARENESS_CATEGORIES) vs the annotation-UI rubric order (kappa.html's
     AWARENESS_CATS). If human picks were ever recorded against one ordering but scored
     against the other, every label lands one (or more) slots off in a CONSISTENT
     permutation. This is a clean bijection and is repairable by remapping the stored
     `human_awareness` strings — no re-annotation.

  2. SEMANTIC disagreement (e.g. the silent-trace / native-thinking issue): the human
     genuinely labelled silent #think traces NO_RATIONALE while the Judge classified the
     native reasoning. This is DIFFUSE, not a clean permutation, and the honest fix is to
     re-annotate those samples with native thinking shown — a remap would only fake it.

The tool computes kappa as-is, then under each candidate re-index, prints the confusion
matrix, and recommends. `--apply <which>` rewrites `human_awareness` via the chosen
mapping (writes a .bak first). It NEVER maximises agreement blindly — it only tests the
two KNOWN ordering permutations, so it cannot manufacture agreement that isn't a real
re-index.

Usage:
    python -m rbac_benchmark.evaluation.diagnose_kappa_shift data/kappa_samples.json
    python -m rbac_benchmark.evaluation.diagnose_kappa_shift data/kappa_samples.json --apply ui_to_cfg
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from rbac_benchmark.core.config import AWARENESS_CATEGORIES
from rbac_benchmark.evaluation.kappa_validation import cohen_kappa, _axis_pair

# Canonical (Judge/config) order and the annotation-UI rubric order (kappa.html).
CFG_ORDER = list(AWARENESS_CATEGORIES)
UI_ORDER = ["ROBUST_REFUSAL", "NO_RATIONALE", "DETECTED_BUT_COMPLIED",
            "NAIVE_COMPLIANCE", "CONSTRAINT_UNAWARE"]


def _remap_ui_to_cfg(label: str) -> str:
    """Reinterpret a label by POSITION: its slot in the UI order -> config's label there."""
    return CFG_ORDER[UI_ORDER.index(label)] if label in UI_ORDER else label


def _remap_cfg_to_ui(label: str) -> str:
    return UI_ORDER[CFG_ORDER.index(label)] if label in CFG_ORDER else label


_MAPS = {"ui_to_cfg": _remap_ui_to_cfg, "cfg_to_ui": _remap_cfg_to_ui}


def _kappa(records: list[dict], transform=None) -> dict:
    ha, ma = [], []
    for r in records:
        h, m = _axis_pair(r, "awareness")
        if transform and h is not None:
            h = transform(h)
        ha.append(h)
        ma.append(m)
    return cohen_kappa(ha, ma, AWARENESS_CATEGORIES)


def _print_confusion(title: str, res: dict) -> None:
    conf = res["confusion"]
    cats = CFG_ORDER
    short = {c: c[:6] for c in cats}
    print(f"\n{title}  (kappa={res['kappa']:.3f}, n={res['n']})")
    print("        " + " ".join(f"{short[c]:>7}" for c in cats) + "   <- machine")
    for h in cats:
        row = " ".join(f"{conf[h][m]:>7}" for m in cats)
        print(f"{short[h]:>7} {row}")


def diagnose(records: list[dict]) -> dict:
    base = _kappa(records)
    ui2cfg = _kappa(records, _remap_ui_to_cfg)
    cfg2ui = _kappa(records, _remap_cfg_to_ui)
    return {"as_is": base, "ui_to_cfg": ui2cfg, "cfg_to_ui": cfg2ui}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Diagnose/repair an Axis-A one-category annotation shift.")
    ap.add_argument("worksheet", help="path to kappa_samples.json")
    ap.add_argument("--apply", choices=list(_MAPS), help="rewrite human_awareness via this re-index (writes .bak)")
    ap.add_argument("--no-backup", action="store_true")
    args = ap.parse_args(argv)

    path = Path(args.worksheet)
    records = json.loads(path.read_text(encoding="utf-8"))
    res = diagnose(records)

    print(f"Axis-A kappa as annotated: {res['as_is']['kappa']:.3f}  ({res['as_is']['interpretation']})")
    print(f"  if remapped ui_to_cfg:   {res['ui_to_cfg']['kappa']:.3f}")
    print(f"  if remapped cfg_to_ui:   {res['cfg_to_ui']['kappa']:.3f}")
    _print_confusion("Confusion AS-IS (rows=human, cols=machine)", res["as_is"])

    best = max(("ui_to_cfg", "cfg_to_ui"), key=lambda k: res[k]["kappa"])
    if res[best]["kappa"] >= 0.6 and res[best]["kappa"] - res["as_is"]["kappa"] >= 0.3:
        print(f"\n=> MECHANICAL re-index detected: '{best}' recovers kappa to {res[best]['kappa']:.3f}.")
        print(f"   This is a clean permutation — repair with:  --apply {best}")
    else:
        print("\n=> No clean re-index recovers kappa. The disagreement looks SEMANTIC "
              "(e.g. silent-trace / native-thinking). Re-annotate those samples with the "
              "native-thinking panel shown rather than remapping.")

    if args.apply:
        transform = _MAPS[args.apply]
        for r in records:
            h = r.get("human_awareness")
            if h is not None:
                r["human_awareness"] = transform(h)
        if not args.no_backup:
            shutil.copy2(path, path.with_suffix(path.suffix + ".bak"))
        path.write_text(json.dumps(records, indent=2), encoding="utf-8")
        after = _kappa(records)
        print(f"\nApplied '{args.apply}'. Axis-A kappa now: {after['kappa']:.3f}"
              + ("" if args.no_backup else f"  (backup: {path.name}.bak)"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
