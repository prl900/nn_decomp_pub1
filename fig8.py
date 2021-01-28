import pandas as pd
from matplotlib import pyplot as plt

fig, ax = plt.subplots()

df = pd.read_csv("exp3_out.csv")
df = df.drop(columns=['i', 'j'])
df_nn_ss = df[df['method']=='nn_ss'].groupby(["n_comps"]).mean()
df_nn_ss.rename(columns={"mse": "NN single band (*7)"}, inplace=True)
df_nn_ms = df[df['method']=='nn_ms'].groupby(["n_comps"]).mean()
df_nn_ms.rename(columns={"mse": "NN multispectral"}, inplace=True)
df_nn_msc = df[df['method']=='nn_msc'].groupby(["n_comps"]).mean()
df_nn_msc.rename(columns={"mse": "NN Conv multispectral"}, inplace=True)
df = pd.concat([df_nn_ss,df_nn_ms,df_nn_msc], axis=1)

df.plot.bar(ax=ax, xlabel="Number of components", ylabel="MSE", title="MSE for reconstructed images per methodology")
ref = df_nn_ss.iloc[0].to_numpy()[0]
plt.axhline(y=ref, color='black', linestyle='--', linewidth=0.5)
plt.savefig("fig8.png")
plt.show()

