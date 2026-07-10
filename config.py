#Gera os diretorios e paths de ficheiro de output utlizados no projeto.
from pathlib import Path
import sys
import json, logging, os


IN_COLAB = 'google.colab' in sys.modules

if IN_COLAB:
    from google.colab import drive
    drive.mount('/content/drive')
    base_dir = Path("/content/drive/MyDrive/AC_TRABALHO/TRABALHO_ESTRUTURADO")
    import sys
    sys.path.append('/content/drive/MyDrive/AC_TRABALHO/TRABALHO_ESTRUTURADO/sources')
else:
    base_dir = Path(".").resolve() # path do repositorio


#path datasets .csv
datasets_path = base_dir / "noise_outliers"
metafeatures_path = base_dir / "metafeatures_output" 
processed_dataset_output_path = base_dir / "data_processing.csv"
outlier_sorted_clustering = base_dir / "clustering_results" / "outlier_sorted_clustering.csv"
overlapping_sorted_clustering = base_dir / "clustering_results" / "overlapping_sorted_clustering.csv"
final_eval_file = Path(base_dir/"train_and_evaluate_dir")
