import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score

#########
"""
Binning script for ChiMerge, Equal-Sized, Tree Optimal technique
"""
#########

class Calculate_WOE_IV:

    @staticmethod
    def cross_table(df, target, var):
        cross_tab = pd.crosstab(df[var], df[target]).reset_index(drop=False)
        cross_tab.rename(columns={0:"good", 1:"bad"}, inplace=True)
        cross_tab["total"] = cross_tab["good"] + cross_tab["bad"]
        cross_tab["bad_rate"] = cross_tab["bad"] / cross_tab["total"]
        cross_tab["good_rate"] = cross_tab["good"] / cross_tab["total"]
        return cross_tab
    
    @staticmethod
    def WOE_IV_table(cross_tab):
        df = cross_tab.copy()
        df["WOE"] = np.log(df["good_rate"] / df["bad_rate"])
        df["IV"] = (df["good_rate"] - df["bad_rate"]) * df["WOE"]
        return df
    
    def calculate_IV(df):
        IV = sum((df["good_rate"] - df["bad_rate"]) * df["WOE"])
        return IV

class BinningTools:

    # Equal-Sized
    @staticmethod
    def equal_sized(x, y=None, max_bins=5):
        index, bins = pd.qcut(x, max_bins, labels=False, retbins=True, duplicates='drop')
        return index, bins

    # ChiMerge
    @staticmethod
    def chisquare(arr):
        cls_cnt = np.sum(arr, axis=0, keepdims=True)
        cls_dist = cls_cnt / np.sum(cls_cnt)
        exp = np.matmul(np.sum(arr, axis=1, keepdims=True), cls_dist)
        return np.sum(np.divide((arr - exp)**2, exp, out=np.zeros_like(exp), where=exp != 0))

    @staticmethod
    def chimerge(x, y, max_bins=5):
        index, bins = equal_sized(x, y, max_bins)
        df_y = pd.get_dummies(y).groupby(index).sum()

        bins = np.append(bins[df_y.index.values], bins[-1])
        y = df_y.values

        while y.shape[0] > max_bins:
            chisq = np.array([chisquare(y[i:i+2]) for i in range(y.shape[0]-1)]) # calculate chisq value over y
            p_val = np.argmin(chisq) # select index which has lowest chisq (selected)
            y[p_val, :] += y[p_val + 1, :]
            y = np.delete(y, p_val, axis=0)
            bins = np.delete(bins, p_val + 1)
        return bins

    # Tree Optimal Binning
    @staticmethod
    def tree_optimal(x, y, max_depth=4):
        score_ls = [] # store the AUC and ROC
        score_std_ls = [] # store the std dev of AUC and ROC
        
        for tree_depth in range(1, max_depth+1):
            tree_model = DecisionTreeClassifier(max_depth=tree_depth)
            scores = cross_val_score(tree_model, x.to_frame(), y, cv=3, scoring='roc_auc')   
            score_ls.append(np.mean(scores))
            score_std_ls.append(np.std(scores))
        
        temp = pd.concat([pd.Series(range(1, max_depth+1)), pd.Series(score_ls), pd.Series(score_std_ls)], axis=1)
        temp.columns = ['depth', 'roc_auc_mean', 'roc_auc_std']
        max_depth_ = temp[temp['roc_auc_mean'] == temp['roc_auc_mean'].max()].depth.values[0]
        tree_model_ = DecisionTreeClassifier(max_depth=max_depth_)
        tree_model_.fit(x.to_frame(), y)
        var_tree = []
        var_tree = tree_model_.predict_proba(x.to_frame())[:,1]

        return np.unique(var_tree)

    # Combine all methods
    @staticmethod
    def do_binning(x, binning, max_bins=5, y=None):
        if binning == 'chimerge':
            if y is None:
                raise ValueError('y should not be empty for chimerge binning.')
            bins = chimerge(x, y, max_bins)
            return bins
        elif binning == 'equal_sized':
            index, bins = equal_sized(x, max_bins)
            return bins
        elif binning == 'tree_optimal':
            if y is None:
                raise ValueError('y should not be empty for tree_optimal binning') 
            bins = tree_optimal(x, y, max_bins)
            return bins
        else:
            raise Exception('Unknown binning method {}.'.format(binning))

        return bins