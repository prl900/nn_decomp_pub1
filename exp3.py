import numpy as np
import xarray as xr
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
        self.cmps = nn.Parameter(torch.rand(n_comps, n_pix*n_pix, requires_grad=True))

    def forward(self):
        return torch.matmul(self.cfs, self.cmps)

class MFConv(nn.Module):
    def __init__(self, n_coeffs, n_comps, n_pix):
        super(MFConv, self).__init__()
        self.n_comps = n_comps
        self.n_pix = n_pix
        self.cfs = nn.Parameter(torch.rand(n_coeffs, 2*n_comps, requires_grad=True))
        self.conv1 = nn.Conv2d(n_comps, 2*n_comps, kernel_size=5)
        self.cmps = nn.Parameter(torch.rand(1, n_comps, n_pix+4, n_pix+4, requires_grad=True))

    def forward(self):
        return torch.matmul(self.cfs, self.conv1(self.cmps).view(2*self.n_comps,self.n_pix**2))


def nan_mse_loss(output, target):
    loss = torch.mean((output[target == target] - target[target == target])**2)
    return loss


print("i,j,band_name,n_comps,nn_mse,nnconv_mse")
#for j in range(7,18):
for j in range(7,8):
    #for i in range(10,25):
    for i in range(10,11):

        ds = xr.open_dataset(f"/data/pca_act/{26*j+i:03d}_clean.nc")
        ds = ds.isel(time=(np.count_nonzero(~np.isnan(ds.nbart_blue.values), axis=(1,2)))>160000*.66)

        stack = np.empty((0,400,400))
        for band_name in ["nbart_red","nbart_green","nbart_blue",
                          "nbart_nir_1","nbart_nir_2", "nbart_swir_2","nbart_swir_3"]:
            results = []

            band = ds[band_name].astype(np.float32).values
            stack = np.append(stack, band, axis=0)
            #stack = stack.reshape(-1,400*400)#/ 1e4

        stack = stack.reshape(stack.shape[0], -1)

        # NN missing data
        for ncomps in [6,8,10,12,14,16,18,20]:
            ncoeffs = stack.shape[0]
            npix = 400

            tmean = np.nanmean(stack, axis=0)
            target = torch.from_numpy(stack-tmean).float().to(device)

            net = MF(ncoeffs, ncomps, npix)
            net.to(device)
            opt = optim.AdamW(net.parameters(), lr=1.0)
            n_epoch  = 500
            for epoch in range(n_epoch):
                yhat = net()
                loss = nan_mse_loss(yhat, target)

                net.zero_grad()
                loss.backward()
                opt.step()

            """
            with torch.no_grad():
                net.cfs.data = net.cfs.data*torch.norm(net.cmps, dim=1).data/20
                net.cmps.data = net.cmps.data/torch.norm(net.cmps, dim=1).data[:,None]*20
            """

            opt = optim.AdamW(net.parameters(), lr=0.001)
            n_epoch  = 500
            for epoch in range(n_epoch):
                yhat = net()
                loss = nan_mse_loss(yhat, target)

                net.zero_grad()
                loss.backward()
                opt.step()

            """
            with torch.no_grad():
                net.cfs.data = net.cfs.data*torch.norm(net.cmps, dim=1).data/20
                net.cmps.data = net.cmps.data/torch.norm(net.cmps, dim=1).data[:,None]*20
            """

            rec = net().cpu().detach().numpy()+tmean
            results.append(np.nanmean(np.square(rec-stack)))

            net = MFConv(ncoeffs, ncomps, npix)
            net.to(device)
            opt = optim.AdamW(net.parameters(), lr=0.1)
            n_epoch  = 1500
            for epoch in range(n_epoch):
                yhat = net()
                loss = nan_mse_loss(yhat, target)

                net.zero_grad()
                loss.backward()
                opt.step()

            """
            with torch.no_grad():
                net.cfs.data = net.cfs.data*torch.norm(net.cmps, dim=1).data/20
                net.cmps.data = net.cmps.data/torch.norm(net.cmps, dim=1).data[:,None]*20
            """

            opt = optim.AdamW(net.parameters(), lr=0.001)
            n_epoch  = 500
            for epoch in range(n_epoch):
                yhat = net()
                loss = nan_mse_loss(yhat, target)

                net.zero_grad()
                loss.backward()
                opt.step()

            """
            with torch.no_grad():
                net.cfs.data = net.cfs.data*torch.norm(net.cmps, dim=1).data/20
                net.cmps.data = net.cmps.data/torch.norm(net.cmps, dim=1).data[:,None]*20
            """

            rec = net().cpu().detach().numpy()+tmean
            results.append(np.nanmean(np.square(rec-stack)))

            print(f"{i},{j},{band_name},{n_comps},{results[0]},{results[1]}", flush=True)
