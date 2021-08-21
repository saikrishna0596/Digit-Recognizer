import pandas as pd
import numpy as np
import os
import zipfile

import yaml
from sklearn.model_selection import train_test_split


def yaml_to_dict(config_file):
    with open('config.yml', 'r') as c:
        return yaml.load(c)


def extract_zip(zip_file_path, out_dir):
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(out_dir)


def load_flat_file(filepath, format_cols=True, encoding='utf-8'):
    if filepath.endswith('.csv'):
        df = pd.read_csv(filepath,encoding=encoding)
    if filepath.endswith('.xlsx'):
        df = pd.read_excel(filepath)
    if filepath.endswith('.sas7bdat'):
        df = pd.read_sas(filepath,encoding=encoding,format='sas7bdat')
    return df


def create_out_dir(filepath):
    os.makedirs(filepath, exist_ok=True)
    

def cv_train_split(train_df, cv_perc, y_var):
    """
    Train CV split

    Args:
        train_df (pd.DataFrame): index must be set to row ID
        cv_perc (float)
        y_var (str)

    Returns:
        Train and CV datasets
    """
    X = train_df.loc[:, train_df.columns!=y_var]
    y = train_df[y_var]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=cv_perc, random_state=42)
    return X_train, X_test, y_train, y_test