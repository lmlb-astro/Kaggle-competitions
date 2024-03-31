import numpy as np
import pandas as pd

####################

## transform the column names with non-numerical categories (i.e. enums)
## to columns with numerical values based on the information provided in a hashmap
def trans_cols(df, col_name, new_name, trans_dict):
    for key in trans_dict.keys():
        df.loc[df[col_name] == key, new_name] = trans_dict[key]
    return df


## get features for the train/fitting of the model
def get_features(df, col_names):
    ## loop over the column names and append the results
    feat_list = []
    for name in col_names:
        feat_list.append(np.array(df[name]))

    ## return the list as a transposed array
    return np.array(feat_list).T

## Get the survival rate for the groups of a given category
def get_group_probabilities(df, category):
    ## perform count and survival sum for each group
    df_count = df.groupby(category)["Survived"].count()
    df_sum = df.groupby(category)["Survived"].sum()

    ## calculate the survival rate
    surv_rate = df_sum/df_count

    ## return data frame with the count in each group and their survival rate
    return pd.DataFrame(data = {"survival rate": surv_rate, "people in group": df_count})
