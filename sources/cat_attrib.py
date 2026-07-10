import openml


def get_categorical_indicator(dataset_name: str) ->tuple[list,list,int]:
    """
    Returns a dict mapping column_name -> is_categorical (bool)
    by fetching feature metadata from OpenML using the dataset ID
    embedded in the filename.
    """
    # Extract ID from filename e.g. "dataset_38_sick.csv" -> 38
    dataset_id = int(dataset_name.split("_")[1])
    
    ds = openml.datasets.get_dataset(
        dataset_id,
        download_data=False,        
        download_qualities=False,  
        download_features_meta_data=True  
        #Não fazemos download de dados, apenas datatype.
    )
    
    # Calcular percentagem de variáveis categóricas. Usamos para evitar calcular métricas depedentes de distribuição gaussiana (skewness, kurtosis)
    num_symbolic = float(ds.qualities['NumberOfSymbolicFeatures'])
    num_features = float(ds.qualities['NumberOfFeatures'])
    pct_categorical = (num_symbolic -1 )/(num_features-1) # OpenMl tambem conta a classe alvo. Só queremos features categóricas.

    # Retornar uma lista das features nominais para cada dataset
    nominal_attributes = []
    quantitative_attributes = []
    for feature in ds.features.values():
        if feature.name == ds.default_target_attribute: #eliminar a target class das features a serem processadas.
            continue

        if feature.data_type == 'nominal':
            nominal_attributes.append(feature.name)

        if feature.data_type == "numeric":
            quantitative_attributes.append(feature.name)

    return quantitative_attributes, nominal_attributes, pct_categorical
