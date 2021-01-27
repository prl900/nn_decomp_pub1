import numpy as np
import xarray as xr
import pandas as pd
from matplotlib import pyplot as plt

from skimage.morphology import dilation
from skimage.morphology import disk
from skimage.morphology import remove_small_objects

from sklearn.decomposition import PCA

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

class MF(nn.Module):
    def __init__(self, n_coeffs, n_comps, n_pix):
        super(MF, self).__init__()
        self.cfs = nn.Parameter(torch.rand(n_coeffs, n_comps, requires_grad=True))
        self.cmps = nn.Parameter(torch.rand(n_comps, n_pix, requires_grad=True))

    def forward(self):
        return torch.matmul(self.cfs,self.cmps)


df = pd.DataFrame(columns=["band_name","pca_mse","nn_mse"])

for j in range(7,18):
    for i in range(10,25):

        ds = xr.open_dataset(f"/data/pca_act/{26*j+i:03d}_clean.nc")
        ds = ds.isel(time=(np.count_nonzero(~np.isnan(ds.nbart_blue.values), axis=(1,2)))>160000*.66)

        for band_name in ["nbart_red","nbart_green","nbart_blue",
                          "nbart_nir_1","nbart_nir_2", "nbart_swir_2","nbart_swir_3"]:
            results = []

            stack = ds[band_name].astype(np.float32) #/ 1e4

            stack = stack.interpolate_na(dim='time')
            stack = stack.interpolate_na(dim='time', method='nearest', fill_value='extrapolate')

            stack = stack.values.reshape(-1, 400*400)
            ncomps = 12
            
            # PCA
            pca = PCA(n_components=ncomps).fit(stack)
            coeffs = pca.transform(stack)
            pca_decomp = pca.inverse_transform(coeffs)
            results.append(np.mean(np.square(pca_decomp/1e4-stack/1e4)))

            ncoeffs = stack.shape[0]
            npix = 160000

            net = MF(ncoeffs, ncomps, npix)
            net.to(device)
            tmean = np.mean(stack, axis=0)
            target = torch.from_numpy(stack-tmean).float().to(device)
            mse = nn.MSELoss()

            opt = optim.AdamW(net.parameters(), lr=1.0)
            n_epoch  = 1000
            for epoch in range(n_epoch):
                yhat = net()
                loss = mse(yhat, target)

                net.zero_grad()
                loss.backward()
                opt.step()

            with torch.no_grad():
                net.cfs.data = net.cfs.data*torch.norm(net.cmps, dim=1).data/20
                net.cmps.data = net.cmps.data/torch.norm(net.cmps, dim=1).data[:,None]*20

            opt = optim.AdamW(net.parameters(), lr=0.001)
            n_epoch  = 1000
            for epoch in range(n_epoch):
                yhat = net()
                loss = mse(yhat, target)

                net.zero_grad()
                loss.backward()
                opt.step()

            with torch.no_grad():
                net.cfs.data = net.cfs.data*torch.norm(net.cmps, dim=1).data/20
                net.cmps.data = net.cmps.data/torch.norm(net.cmps, dim=1).data[:,None]*20
            
            opt = optim.AdamW(net.parameters(), lr=0.0001)
            n_epoch  = 1000
            for epoch in range(n_epoch):
                yhat = net()
                loss = mse(yhat, target)

                net.zero_grad()
                loss.backward()
                opt.step()

            with torch.no_grad():
                net.cfs.data = net.cfs.data*torch.norm(net.cmps, dim=1).data/20
                net.cmps.data = net.cmps.data/torch.norm(net.cmps, dim=1).data[:,None]*20


            rec = net().cpu().detach().numpy()+tmean
            results.append(np.nanmean(np.square(rec/1e4-stack/1e4)))

            df = df.append({"band_name": band_name,
                            "pca_mse":  results[0],
                            "nn_mse":  results[1]}, ignore_index=True)


df['pca_mse'] = df['pca_mse']*1e5
df['nn_mse'] = df['nn_mse']*1e5
df = df.groupby('band_name').mean()

print(df.to_latex())
