import pandas as pd
import os

from sklearn.preprocessing import LabelEncoder


def read_data(base_path, datasets_list=["Base"], seed=None):
    datasets_paths = {
        key : f"{base_path}{key}.parquet" if key == "Base" else f"{base_path}Variant {key.split()[-1]}.parquet" for key in datasets_list
    }
    for key in datasets_paths.keys():
        if os.path.exists(datasets_paths[key]):
            print(f"Dataset {key} already exists")
            continue
        df = pd.read_csv(datasets_paths[key].replace(".parquet", ".csv"))
        df.to_parquet(datasets_paths[key])
        print(f"Dataset {key} saved as parquet")
    datasets = {key: pd.read_parquet(path) for key, path in datasets_paths.items()}
    categorical_features = [
        "payment_type",
        "employment_status",
        "housing_status",
        "source",
        "device_os",
    ]
    train_dfs = {key: df[df["month"]<6].sample(frac=1, replace=False, random_state=seed) for key, df in datasets.items()}
    test_dfs = {key: df[df["month"]>=6].sample(frac=1, replace=False, random_state=seed)  for key, df in datasets.items()}
    for name in datasets.keys():  
        train = train_dfs[name]
        test = test_dfs[name]
        for feat in categorical_features:
            encoder = LabelEncoder()
            encoder.fit(train[feat])  
            train[feat] = encoder.transform(train[feat])  
            test[feat] = encoder.transform(test[feat])    
    return datasets_paths, datasets, train_dfs, test_dfs
