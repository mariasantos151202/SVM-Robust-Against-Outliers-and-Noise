import config
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scikit_posthocs as sp
import stac
import matplotlib.pyplot as plt
import os
from scipy.stats import friedmanchisquare, f as f_dist, norm

# Paths para os ficheiros de avaliação relevantes 
final_eval_path= config.final_eval_file


def parse_mean(cell: str) -> float:
    """
    Separação dos valores de cada célula em mean e variance
    Retornamos apenas a MÉDIA da métrica
    """
    return float(cell.split()[0].strip())

def load(path: str, label: str, METRIC="Balanced Accuracy") -> pd.DataFrame:
    """Abrir o ficheiro eval, fazer parsing dos valores"""
    df = (pd.read_csv(path).sort_values("Dataset").reset_index(drop=True))
    df[label] = df[METRIC].apply(parse_mean)
    return df[["Dataset", label]]


def ranking(score_model_list: list, labels: list)-> pd.Dataframe: 

    score_matrix= np.column_stack(score_model_list)
    rank_df = pd.DataFrame(score_matrix, columns=labels).rank(
        axis=1, ascending=False, method="average"
    )

    #Média dos ranks/modelo, para todos os datasets
    average_ranks = rank_df.mean()

    rank_df.to_csv("rank_df.csv", index=False)
    return rank_df, average_ranks

def friedman(LABEL_STANDARD:str,
            LABEL_MODIFIED3, 
            LABEL_MODIFIED31,
            average_ranks, 
            rankdf):
    alpha = 0.05

    """
    Calcula o p-valor para distribuição X^2
    """

    chi2, p_chi2 = stats.friedmanchisquare(
        rankdf[LABEL_STANDARD].values,
        rankdf[LABEL_MODIFIED3].values,
        rankdf[LABEL_MODIFIED31].values
    ) 
    
    N = len(rankdf)
    k=3
    F_F = ((N-1)*chi2) / (N*(k-1)-chi2)
    df1 = k-1
    df2 = (k-1)*(N-1)
    p_F = 1 - f_dist.cdf(F_F, df1, df2)

    reject : bool = p_F < alpha

    if reject:
        print(f"Com {(1-alpha)*100} % confiança, rejeitamos a hipótese H0.Existe pelo menos um modelo com rank suficiente distinto")
        print(f"A rejeição permite calcular Bonferroni-Dunn post-hoc\n")
        bonferroni(avg_ranks=average_ranks, k=k, N=N, LABEL_STANDARD=LABEL_STANDARD)
    else:
        print(f"H0 não é rejeitado {p_F:.6f} ≥ alpha={alpha}). Não existem diferências sistemáticas de rank")

    return reject, p_chi2

def bonferroni(avg_ranks, k, N, LABEL_STANDARD, LABEL_MODIFIED3, LABEL_MODIFIED31):
    """
    Comparamos os ranks médios par a par de maneira a identificar se um deles é sistematicamente superior ao outro. 
    """
    alpha=0.05

    baseline = LABEL_STANDARD
    comparable_models = [LABEL_MODIFIED3, LABEL_MODIFIED31]
    n_comparisions = len(comparable_models)

    #Critical value para o standard
    q_alpha=norm.ppf(1-alpha / (2*n_comparisions))

    #CD- Critical Difference
    CD = q_alpha * np.sqrt((k*(k+1) / (6*N)))

    for model in comparable_models:
        rank_difference = abs(avg_ranks[baseline]-avg_ranks[model])
        reject: bool = rank_difference>=CD
        winner    = (model if avg_ranks[model] < avg_ranks[baseline] else baseline)

        print(f"{baseline} vs {model}")
        print(f"|R(baseline) - R(challenger)| = {rank_difference:.4f} CD={CD:.4f}")

        if reject:
            print(f"H0 foi rejeitado. No {baseline} VS {model}, {winner} é o modelo superior")

        else:
            print(f"Não existe diferença sistemática de rank entre o modelo e baseline - ")

def __main__():
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


    # Verifica se todos os csvs têm os mesmos datasets.
    assert set(df_std["Dataset"].values)  == set(df_mod3["Dataset"].values) \
        == set(df_mod31["Dataset"].values), \
        "Dataset mismatch across eval files — check for missing or misnamed datasets"

    scores_std   = df_std[LABEL_STANDARD].values
    scores_mod3  = df_mod3[LABEL_MODIFIED3].values
    scores_mod31 = df_mod31[LABEL_MODIFIED31].values
    score_model_list = [scores_std, scores_mod3, scores_mod31]

    rank_df = ranking(
        score_model_list=score_model_list,
        labels=[LABEL_STANDARD, LABEL_MODIFIED3, LABEL_MODIFIED31]
    )

    rejected, p = friedman(
        rankdf=rank_df,
        label_standard=LABEL_STANDARD,
        label_modified3=LABEL_MODIFIED3,
        label_modified31=LABEL_MODIFIED31
    )


if __name__ == "__main__":
    __main__()