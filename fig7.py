import pandas as pd
from matplotlib import pyplot as plt

fig, ax = plt.subplots()

df = pd.read_csv("exp2_out.csv")
df = df.drop(columns=['i', 'j', 'pca_int_mse', 'nn_int_mse'])
df = df.groupby(["band_name"]).mean()
df = df.stack().mean(level=1)


speed = [0.1, 17.5]
lifespan = [2, 8]
index = ['Validation patches', 'Whole tile']

df = pd.DataFrame({'PCA': [df.iloc[1], df.iloc[0]],
                   'NN': [df.iloc[3], df.iloc[2]]}, index=index)

#print(df, df.iloc[0])
df.plot.bar(ax=ax, ylabel="MSE", title="MSE for missing value imputation methodologies")
plt.xticks(rotation=0)
plt.savefig("fig7.png")
plt.show()
