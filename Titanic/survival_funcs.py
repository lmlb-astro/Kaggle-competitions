import numpy as np
import pandas as pd

####################


def trans_cols(df, col_name, new_name, trans_dict):
    for key in trans_dict.keys():
        df.loc[df[col_name] == key, new_name] = trans_dict[key]
    return df


def get_features(df, col_names):
    ## loop over the column names and append the results
    feat_list = []
    for name in col_names:
        feat_list.append(np.array(df[name]))

    ## return the list as a transposed array
    return np.array(feat_list).T