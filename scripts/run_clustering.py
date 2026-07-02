"""Reproduce the blind k-means and fit + persist the phenotype model."""
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

from src.data_load import load
from src.phenotype import fit_phenotypes, TRAJ_COLS


def main():
    df = load()
    sub = df.dropna(subset=TRAJ_COLS)
    X = StandardScaler().fit_transform(sub[TRAJ_COLS].values)
    print(f"N = {len(sub)} (complete years 1-3)\nk : silhouette")
    for k in range(2, 9):
        km = KMeans(n_clusters=k, n_init=25, random_state=42).fit(X)
        print(f"{k} : {silhouette_score(X, km.labels_):.3f}")

    bundle = fit_phenotypes(df)
    print(f"\nFitted k={bundle['k']} on {bundle['n_train']} patients; "
          f"saved to artifacts/phenotype_kmeans.joblib")


if __name__ == "__main__":
    main()
