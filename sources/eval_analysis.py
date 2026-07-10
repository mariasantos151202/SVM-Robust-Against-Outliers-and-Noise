import config
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scikit_posthocs as sp
import os
from scipy.stats import friedmanchisquare, f as f_dist, norm
from config import final_eval_file

# Paths para os ficheiros de avaliação relevantes
final_eval_path = config.final_eval_file

# Global label variables — set in main(), available to all functions
LABEL_STANDARD   = None
LABEL_MODIFIED3  = None
LABEL_MODIFIED31 = None


def parse_mean(cell: str) -> float:
    """
    Separação dos valores de cada célula em mean e variance
    Retornamos apenas a MÉDIA da métrica
    """
    return float(cell.split()[0].strip())


def load(path: str, label: str, METRIC="Balanced Accuracy") -> pd.DataFrame:
    """Abrir o ficheiro eval, fazer parsing dos valores"""
    df = pd.read_csv(path)
    df[label] = df[METRIC].apply(parse_mean)
    return df[["Dataset", label]]


def ranking(score_model_list: list, labels: list) -> tuple:
    score_matrix = np.column_stack(score_model_list)
    rank_df = pd.DataFrame(score_matrix, columns=labels).rank(
        axis=1, ascending=False, method="average"
    )

    # Média dos ranks/modelo, para todos os datasets
    average_ranks = rank_df.mean()

    rank_df.to_csv("rank_df.csv", index=False)
    return rank_df, average_ranks


def friedman(average_ranks, rankdf):
    """
    Calcula o p-valor para distribuição X^2
    """
    alpha = 0.05

    chi2, p_chi2 = friedmanchisquare(
        rankdf[LABEL_STANDARD].values,
        rankdf[LABEL_MODIFIED3].values,
        rankdf[LABEL_MODIFIED31].values
    )

    N = len(rankdf)
    k = 3
    F_F = ((N - 1) * chi2) / (N * (k - 1) - chi2)
    df1 = k - 1
    df2 = (k - 1) * (N - 1)
    p_F = 1 - f_dist.cdf(F_F, df1, df2)

    reject: bool = p_F < alpha

    if reject:
        print(f"Com {(1 - alpha) * 100} % confiança, rejeitamos a hipótese H0. "
              f"Existe pelo menos um modelo com rank suficiente distinto")
        bonferroni(avg_ranks=average_ranks, k=k, N=N)
    else:
        print(f"H0 não é rejeitado ({p_F:.6f} ≥ alpha={alpha}). "
              f"Não existem diferenças sistemáticas de rank")

    return reject, p_chi2


def bonferroni(avg_ranks, k, N):
    """
    Comparamos os ranks médios par a par de maneira a identificar se um deles
    é sistematicamente superior ao outro.
    """
    alpha = 0.05

    print("Bonferroni-Dunn\n")

    baseline = LABEL_STANDARD
    comparable_models = [LABEL_MODIFIED3, LABEL_MODIFIED31]
    n_comparisions = len(comparable_models)

    # Critical value para o standard
    q_alpha = norm.ppf(1 - alpha / (2 * n_comparisions))

    # CD - Critical Difference
    CD = q_alpha * np.sqrt((k * (k + 1) / (6 * N)))

    for model in comparable_models:
        rank_difference = abs(avg_ranks[baseline] - avg_ranks[model])
        reject: bool = rank_difference >= CD
        winner = (model if avg_ranks[model] < avg_ranks[baseline] else baseline)

        print(f"{baseline} vs {model}")
        print(f"|R(baseline) - R(challenger)| = {rank_difference:.4f}  CD={CD:.4f}")

        if reject:
            print(f"H0 foi rejeitado. No teste de pares ({baseline}, {model}), {winner} é o modelo superior: {avg_ranks[baseline]:.3f},{avg_ranks[model]:.3f}")
        else:
            print(f"Não existe diferença sistemática de rank entre o {model} e {baseline}")

def winner_by_dataset():
    LABEL_STANDARD = "SVM_STANDARD"
    LABEL_MODIFIED3 = "SVM_modified_3"
    LABEL_MODIFIED31 = "SVM_modified_3.1"

    METRICS = ["Balanced Accuracy", "F1-Score (Macro)", "Precision"]
    
    metric_series = []

    for metric in METRICS:
        df_std   = load(str(Path(final_eval_path) / "eval_SVM_STANDARD.csv"),     label=LABEL_STANDARD,   METRIC=metric)
        df_mod3  = load(str(Path(final_eval_path) / "eval_SVM_modified_3.csv"),   label=LABEL_MODIFIED3,  METRIC=metric)
        df_mod31 = load(str(Path(final_eval_path) / "eval_SVM_modified_3_1.csv"), label=LABEL_MODIFIED31, METRIC=metric)

        # Merge all three on Dataset so each row has scores from all models side by side
        merged = df_std.merge(df_mod3, on="Dataset").merge(df_mod31, on="Dataset")

        score_cols = [LABEL_STANDARD, LABEL_MODIFIED3, LABEL_MODIFIED31]

        # For each dataset row, find the column name with the highest score
        merged[f"best_model_{metric}"]  = merged[score_cols].idxmax(axis=1)
        merged[f"best_score_{metric}"]  = merged[score_cols].max(axis=1)

        # Keep only the two result columns plus the dataset index
        metric_series.append(
            merged[["Dataset", f"best_model_{metric}", f"best_score_{metric}"]]
        )

    # Merge all per-metric results into a single wide dataframe
    final_df = metric_series[0]
    for ms in metric_series[1:]:
        final_df = final_df.merge(ms, on="Dataset")

    export_path = config.base_dir/"winner_by_dataset.csv"

    final_df.to_csv(export_path, index=False)
    return final_df


def __main__():
    global LABEL_STANDARD, LABEL_MODIFIED3, LABEL_MODIFIED31

    LABEL_STANDARD   = "SVM_STANDARD"
    LABEL_MODIFIED3  = "SVM_modified_3"
    LABEL_MODIFIED31 = "SVM_modified_3.1"

    df_std   = None
    df_mod3  = None
    df_mod31 = None

    for file in os.scandir(final_eval_path):
        if file.name == "eval_SVM_STANDARD.csv":
            df_std   = load(file.path, label=LABEL_STANDARD)
        elif file.name == "eval_SVM_modified_3.csv":
            df_mod3  = load(file.path, label=LABEL_MODIFIED3)
        elif file.name == "eval_SVM_modified_3_1.csv":
            df_mod31 = load(file.path, label=LABEL_MODIFIED31)


    # Reduz à interseção dos datasets comuns aos 3 ficheiros e alinha a ordem das linhas
    all_sets = [set(df_std["Dataset"]), set(df_mod3["Dataset"]), set(df_mod31["Dataset"])]
    common = all_sets[0] & all_sets[1] & all_sets[2]

    dropped = (all_sets[0] | all_sets[1] | all_sets[2]) - common
    if dropped:
        print(f"Datasets removidos por não estarem presentes nos 3 ficheiros: {dropped}")

    df_std   = df_std[df_std["Dataset"].isin(common)].sort_values("Dataset").reset_index(drop=True)
    df_mod3  = df_mod3[df_mod3["Dataset"].isin(common)].sort_values("Dataset").reset_index(drop=True)
    df_mod31 = df_mod31[df_mod31["Dataset"].isin(common)].sort_values("Dataset").reset_index(drop=True)

    assert len(df_std) == len(df_mod3) == len(df_mod31), "Row count mismatch after alignment"
    assert list(df_std["Dataset"]) == list(df_mod3["Dataset"]) == list(df_mod31["Dataset"]), \
        "Dataset order mismatch after alignment"
    
    scores_std   = df_std[LABEL_STANDARD].values
    scores_mod3  = df_mod3[LABEL_MODIFIED3].values
    scores_mod31 = df_mod31[LABEL_MODIFIED31].values
    score_model_list = [scores_std, scores_mod3, scores_mod31]

    rank_df, average_ranks = ranking(
        score_model_list=score_model_list,
        labels=[LABEL_STANDARD, LABEL_MODIFIED3, LABEL_MODIFIED31]
    )

    rejected, p = friedman(
        average_ranks=average_ranks,
        rankdf=rank_df
    )
    
    #Se a primeira hipotese nula for rejeitada, existem diferenças sistematicas de ranking
    if rejected:
        avg_ranks_series = pd.Series({
            LABEL_STANDARD : average_ranks[LABEL_STANDARD],
            LABEL_MODIFIED3 : average_ranks[LABEL_MODIFIED3],
            LABEL_MODIFIED31 : average_ranks[LABEL_MODIFIED31],
        })

        plt.figure(figsize=(8,3))

        rank_df_long = rank_df.reset_index().melt(
            id_vars='index',
            var_name='model',
            value_name='rank'
        )

        sp.critical_difference_diagram(
            ranks = avg_ranks_series,
            sig_matrix = sp.posthoc_dunn(rank_df_long, val_col='rank', group_col='model', p_adjust='bonferroni')
        )
        plt.title("Critical Difference Diagram — Bonferroni-Dunn")
        plt.tight_layout()
        plt.savefig("cd_diagram.png", dpi=150)
        plt.show()

if __name__ == "__main__":
    __main__()
