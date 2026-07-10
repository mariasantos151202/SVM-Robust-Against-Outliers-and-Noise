import pandas as pd
import os
import pathlib
from pathlib import Path
import config
#eval_1


# Com cada sessão, os dados de cache permitem ir buscar sempre o mesmo ficheiro (eval_01). Se uma alteração significativa for feita, temos de poder comparar versões.
current_output_file: Path = None
#Procura o último ficheiro de output gerado do tipo eval_*.csv e 


if not (config.metafeatures_path):
    os.makedirs(config.metafeatures_path)



def get_output_file(function_name, output_name: str) -> Path:
    counter=1
    output_folder = config.base_dir / f"{function_name}_dir"
    output_folder.mkdir(parents=True, exist_ok=True)
    
    while (output_folder / f"{output_name}_{counter}.csv").exists():
        counter +=1 
    
    output_file_path: Path = output_folder / f"{output_name}_{counter}.csv"
    return output_file_path
    


#Chamada para construção de ficheiros output da função train_evaluate (métricas de performance)
def building_table(file_path: Path, summary: pd.DataFrame) -> str:

    parent_dir = file_path.resolve().parents[1]

    if not (os.path.exists(parent_dir)):
        os.makedirs(parent_dir)

    write_header = not file_path.exists() # se ainda nao existir o ficheiro

    try:
        summary.to_csv(path_or_buf=file_path, index=False, mode="a", header=write_header)
        return "Performance evaluation successfully exported as .csv"
    except:
        return "Error in .csv export"
