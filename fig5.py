import numpy as np
import xarray as xr
from matplotlib import pyplot as plt

from sklearn.decomposition import PCA


def generate_patches(patch_hsize, mask):
    i = int(np.random.randint(patch_hsize, 400-patch_hsize, size=1))
    j = int(np.random.randint(patch_hsize, 400-patch_hsize, size=1))
    t = int(np.random.randint(mask.shape[0], size=1))

    while np.count_nonzero(~mask[t, j-patch_hsize:j+patch_hsize, i-patch_hsize:i+patch_hsize]) != (2*patch_hsize)**2:
        t = int(np.random.randint(mask.shape[0], size=1))
        i = int(np.random.randint(patch_hsize, 400-patch_hsize, size=1))
        j = int(np.random.randint(patch_hsize, 400-patch_hsize, size=1))

    return t, i, j

ds = xr.open_dataset(f"/data/pca_act/000_clean.nc")
ds = ds.isel(time=(np.count_nonzero(~np.isnan(ds.nbart_blue.values), axis=(1,2)))>160000*.66)

patch_hsize = 40
mask = ds.nbart_blue.isnull().values
pmask = np.zeros(mask.shape, dtype=np.bool)

"""
for _ in range(20):
    t, ii, jj = generate_patches(patch_hsize, mask)
    pmask[t, jj-patch_hsize:jj+patch_hsize, ii-patch_hsize:ii+patch_hsize] = True
"""

# For deterministic results
pmask[10, 100:180,80:160] = True

stack = ds["nbart_red"].astype(np.float32) #/ 1e4

pstack = stack.where(~pmask)
istack = pstack.interpolate_na(dim='time')
istack = istack.interpolate_na(dim='time', method='nearest', fill_value='extrapolate')

stack = stack.values.reshape(stack.shape[0], -1)
pstack = pstack.values.reshape(stack.shape[0], -1)
istack = istack.values.reshape(stack.shape[0], -1)

refstack = np.copy(stack)

stack[~pmask.reshape(-1,160000)] = np.nan

pca = PCA(n_components=12).fit(istack)
coeffs = pca.transform(istack)
pca_decomp = pca.inverse_transform(coeffs)

fig, axs = plt.subplots(2, 2, figsize=(8, 8))

fig.suptitle('Dataset tile versions', fontsize=16, y=0.93)
axs[0,0].imshow(refstack[10].reshape(400,400), cmap='Reds_r')
axs[0,0].set_title('a) Original', y=-0.18)
axs[0,1].imshow(pstack[10].reshape(400,400), cmap='Reds_r')
axs[0,1].set_title('b) Added Patches', y=-0.18)
axs[1,0].imshow(istack[10].reshape(400,400), cmap='Reds_r')
axs[1,0].set_title('c) Output', y=-0.18)
axs[1,1].imshow(stack[10].reshape(400,400), cmap='Reds_r')
axs[1,1].set_title('d) Validation', y=-0.18)

fig.savefig('fig5.png')
