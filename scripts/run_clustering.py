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

    sil = bundle["silhouette_by_k"]
    argmax_k = max(sil, key=sil.get)
    print("\nk : silhouette (UMAP embedding)")
    for k, s in sil.items():
        marks = []
        if k == argmax_k:
            marks.append("argmax")
        if k == bundle["k"]:
            marks.append("selected")
        tag = f"  <- {'/'.join(marks)}" if marks else ""
        print(f"{k:>2} : {s:.4f}{tag}")

    if bundle["k"] != argmax_k:
        print(f"\nSelected k={bundle['k']} (1A manuscript) — within {sil[argmax_k]-sil[bundle['k']]:.4f} "
              f"of the silhouette argmax k={argmax_k} (a tie); see phenotype.choose_k.")
    else:
        print(f"\nAuto-selected k={bundle['k']} (silhouette argmax) on {bundle['n_train']} patients.")
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
