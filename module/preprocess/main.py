#%%
import torch
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


#%%
weights_dir = "/Users/wonhyung64/Github/reinforce/assets/checkpoints/2023-12-14T21-48-10"
np.log

with open(f"{weights_dir}/log", "r") as f:
    results = f.readlines()

total_log = []
for i in range(len(results)):
    result_list = results[i].split()
    result_list = [item.strip() for item in result_list]
    total_log.append(result_list)

log_df = pd.DataFrame(columns=total_log[0], data=total_log[1:])

fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(12,4))
axes[0].plot(log_df["MeanQValue"].astype(float))
axes[1].plot(log_df["MeanReward"].astype(float))
axes[2].plot(log_df["MeanLoss"].astype(float))
axes[1].set_xlabel("Episode/20")
axes[0].set_ylabel("MeanQValue")
axes[1].set_ylabel("MeanReward")
axes[2].set_ylabel("MeanLoss")
fig.set_tight_layout(True)
fig
