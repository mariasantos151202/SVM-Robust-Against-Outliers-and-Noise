def _threshold_calc(self, dataset: Dataset):
        """
        Computes a dataset-aware dominance ratio threshold for check_noise_column.

        Adjusts the base threshold of 10 using two signals:

        1. Dataset size    — small datasets are precious; raise threshold to avoid
                            aggressive column removal when n is low.
        2. Class balance   — imbalanced targets produce naturally skewed feature
                            distributions; raise threshold to avoid dropping
                            legitimate columns that merely reflect the imbalance.

        Returns a float in [10, 25].
        """
        BASE    = 10.0
        MAX     = 25.0
        penalty = 0.0

        n_rows = dataset.data.shape[0]

        # --- Signal 1: Dataset size ---
        if n_rows < 100:
            penalty += 10.0
        elif n_rows < 500:
            penalty += 5.0

        # --- Signal 2: Class imbalance ---
        target_counts = dataset.y.value_counts(normalize=True)
        imbalance_ratio = 1.0  # default if only one class visible
        if len(target_counts) >= 2:
            imbalance_ratio = target_counts.iloc[0] / target_counts.iloc[1]
            if imbalance_ratio > 5:
                penalty += 10.0
            elif imbalance_ratio > 2:
                penalty += 5.0

        threshold = min(BASE + penalty, MAX)

        logging.info(
            f"[{dataset.name}] _threshold_calc: "
            f"n_rows={n_rows}, imbalance_ratio={imbalance_ratio:.2f} "
            f"→ threshold={threshold:.1f}"
        )

        return threshold