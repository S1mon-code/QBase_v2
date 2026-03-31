import numpy as np
from sklearn.decomposition import PCA


def oi_pca_features(closes: np.ndarray, oi: np.ndarray,
                    volumes: np.ndarray, period: int = 120) -> tuple:
    """PCA on OI-derived features.

    Constructs a feature matrix from OI change, OI momentum,
    OI-price correlation, and OI-volume ratio, then extracts the
    first two principal components.  Retrained every period//4 bars.

    Parameters
    ----------
    closes : np.ndarray
        Close prices.
    oi : np.ndarray
        Open interest.
    volumes : np.ndarray
        Trading volume.
    period : int
        Lookback for feature construction and PCA fitting.

    Returns
    -------
    pc1 : np.ndarray
        First principal component score.
    pc2 : np.ndarray
        Second principal component score.
    explained_ratio : np.ndarray
        Sum of explained variance ratio for PC1+PC2.
    """
    n = len(closes)
    pc1 = np.full(n, np.nan)
    pc2 = np.full(n, np.nan)
    explained_ratio = np.full(n, np.nan)

    if n < period + 1:
        return pc1, pc2, explained_ratio

    # Pre-compute raw features
    oi_chg = np.full(n, np.nan)
    oi_chg[1:] = np.diff(oi)

    price_ret = np.full(n, np.nan)
    price_ret[1:] = np.diff(closes) / np.maximum(closes[:-1], 1e-10)

    oi_vol_ratio = np.full(n, np.nan)
    for i in range(n):
        oi_vol_ratio[i] = oi[i] / max(volumes[i], 1.0)

    retrain_interval = max(period // 4, 1)
    pca_model = None
    feat_mean = None
    feat_std = None

    for i in range(period, n):
        # Retrain periodically
        if pca_model is None or (i - period) % retrain_interval == 0:
            # Build feature matrix for training window
            idx_start = i - period
            idx_end = i

            f1 = oi_chg[idx_start:idx_end]
            f2 = price_ret[idx_start:idx_end]
            f3 = oi_vol_ratio[idx_start:idx_end]

            # Rolling OI-price correlation (use rolling 20-bar)
            f4 = np.full(period, 0.0)
            corr_win = 20
            for j in range(corr_win, period):
                abs_j = idx_start + j
                p_slice = price_ret[abs_j - corr_win:abs_j]
                o_slice = oi_chg[abs_j - corr_win:abs_j]
                mask = ~np.isnan(p_slice) & ~np.isnan(o_slice)
                if np.sum(mask) > 3:
                    p_s = p_slice[mask]
                    o_s = o_slice[mask]
                    ps = np.std(p_s)
                    os_ = np.std(o_s)
                    if ps > 0 and os_ > 0:
                        f4[j] = np.corrcoef(p_s, o_s)[0, 1]

            X = np.column_stack([f1, f2, f3, f4])
            valid_rows = ~np.any(np.isnan(X), axis=1)
            X_valid = X[valid_rows]

            if len(X_valid) < 10:
                continue

            feat_mean = np.mean(X_valid, axis=0)
            feat_std = np.std(X_valid, axis=0)
            feat_std[feat_std == 0] = 1.0

            X_norm = (X_valid - feat_mean) / feat_std

            pca_model = PCA(n_components=2)
            pca_model.fit(X_norm)

        if pca_model is None or feat_mean is None:
            continue

        # Transform current bar
        corr_val = 0.0
        corr_win = 20
        if i >= corr_win + 1:
            p_s = price_ret[i - corr_win:i]
            o_s = oi_chg[i - corr_win:i]
            mask = ~np.isnan(p_s) & ~np.isnan(o_s)
            if np.sum(mask) > 3:
                ps = np.std(p_s[mask])
                os_ = np.std(o_s[mask])
                if ps > 0 and os_ > 0:
                    corr_val = np.corrcoef(p_s[mask], o_s[mask])[0, 1]

        feat = np.array([
            oi_chg[i] if not np.isnan(oi_chg[i]) else 0.0,
            price_ret[i] if not np.isnan(price_ret[i]) else 0.0,
            oi_vol_ratio[i] if not np.isnan(oi_vol_ratio[i]) else 0.0,
            corr_val,
        ]).reshape(1, -1)

        feat_norm = (feat - feat_mean) / feat_std
        scores = pca_model.transform(feat_norm)
        pc1[i] = scores[0, 0]
        pc2[i] = scores[0, 1]
        explained_ratio[i] = np.sum(pca_model.explained_variance_ratio_)

    return pc1, pc2, explained_ratio
