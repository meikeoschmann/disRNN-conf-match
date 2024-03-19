import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import subject_data
from subject_data import SubjectData
from scipy import stats
from scipy.io import loadmat

data_dir = '.\data\data_for_Meike.mat'
matDf = loadmat(data_dir)['data']
df = pd.DataFrame(np.squeeze(np.array(matDf.tolist())).T, columns=matDf.dtype.names).drop(columns=['label','Spcorrect1', 'Spcorrect2', 'Snoise']).sort_values(by=['subject', 'trial'])


features_prev = ['type', 'theta', 'Pconfidence',
                 'Pacc', 'Sacc', 'select', 'Reward', 'Sconfidence']
features_curr = ['type', 'theta', 'Sacc']
target = ['Sconfidence']

data = SubjectData(
    df,
    features_prev,
    features_curr,
    target,
    categorical_partners=True,
    categorical_theta=True)

ppt_df = data()

ppts = ppt_df["subject"].unique()

ppt_map = {}
for ppt in ppts:
    df = ppt_df[ppt_df["subject"] == ppt]
    blocks = df["block"].unique()
    bmap = {block: df[df["block"] == block] for block in blocks}
    ppt_map[ppt] = bmap


def ttesting_means():
    for ppt, bmap in ppt_map.items():
        for block, df in bmap.items():
            p1_df = df[(df["type"] == 1)]
            p2_df = df[(df["type"] == 2)]
            p1 = p1_df["Sconfidence"].values
            p2 = p2_df["Sconfidence"].values
            result = stats.ttest_ind(p1, p2, equal_var=False)
            if result.pvalue < 0.05:
                print(f"{int(ppt)}, {int(block)}, {result} *")
            else:
                print(f"{int(ppt)}, {int(block)}, {result}")


if __name__ == "__main__":
    ttesting_means()

