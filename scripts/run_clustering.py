"""Fit + persist the 1A-aligned phenotype pipeline (see src/phenotype.py).

Prints the silhouette curve, the auto-selected k, and per-cluster predicted-TBWL
trajectory means (comparable to Supplementary Table S7). Loads via the standard
loader (keep-first dedup applied).
"""
import warnings

from src.data_load import load
from src.phenotype import fit_phenotypes


def main():
    warnings.simplefilter("ignore")
    df = load()
    print(f"Fitting 1A-aligned phenotypes on N={len(df)} (keep-first deduped)...")
    bundle = fit_phenotypes(df)

    print("\nk : silhouette (UMAP embedding)")
    for k, s in bundle["silhouette_by_k"].items():
        mark = "  <- selected (argmax)" if k == bundle["k"] else ""
        print(f"{k:>2} : {s:.4f}{mark}")

    print(f"\nAuto-selected k={bundle['k']} on {bundle['n_train']} patients.")
    print("\nPer-cluster mean ACTUAL TBWL% by year (ordered by ascending Preop_TBWL):")
    hdr = "cluster    n   " + "  ".join(f"yr{y}" for y in range(1, 7))
    print(hdr)
    for c in range(bundle["k"]):
        cinfo = bundle["cluster_actual_traj"][c]
        cells = []
        for y in range(1, 7):
            v = cinfo["per_year"].get(y)
            cells.append(f"{v['mean']:4.1f}" if v else "  --")
        print(f"{c:>7}  {cinfo['n']:>3}   " + "  ".join(cells))
    print("\nSaved to artifacts/phenotype_kmeans.joblib")


if __name__ == "__main__":
    main()
