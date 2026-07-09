"""Fit + persist the phenotype k-means, with k derived by silhouette (not forced).

Prints the full silhouette curve over k=2..10 and the auto-selected k.

NOTE: uses load(strict_unique_ids=False) as a TEMPORARY bypass of the duplicate-ID
guard — the dedup policy is parked pending Ioanna's cleaned -v2 CSV, and this matches
the provenance of the existing phenotype artifact. Once dedup is resolved, drop the
bypass so this runs against strict-unique data.
"""
from src.data_load import load
from src.phenotype import fit_phenotypes, select_k, TRAJ_COLS
from sklearn.preprocessing import StandardScaler


def main():
    df = load(strict_unique_ids=False)  # TEMP bypass — see module docstring
    sub = df.dropna(subset=TRAJ_COLS)
    X = StandardScaler().fit_transform(sub[TRAJ_COLS].values)

    chosen_k, sil = select_k(X)
    print(f"N = {len(sub)} (complete years 1-3)\nk : silhouette")
    for k, s in sil.items():
        marker = "  <- selected (argmax)" if k == chosen_k else ""
        print(f"{k} : {s:.4f}{marker}")

    bundle = fit_phenotypes(df)  # derives k internally, persists the frozen model
    print(f"\nAuto-selected k={bundle['k']} (silhouette-argmax) on {bundle['n_train']} "
          f"patients; saved to artifacts/phenotype_kmeans.joblib")


if __name__ == "__main__":
    main()
