import torch
import numpy as np

from vars import Var
from pars import Par
from data import Data
from dataloader import Dataloader
from GRU import Model
from plot import plot_bicm, plot_carp, plot_enba, plot_rtms, plot_rtmo

"""
Load data root and select interested fields
"""
data_root = "../../data/par_set1/"
p = Par()
v = Var()

for i in range(63, 64):
    dC = Data(i, p, data_root)
    dL = Dataloader(dC, v, p)
    m = Model(dL, p)
    m.load()

    plot_carp(m.carp_model, dL.carp_test_loader, dL.carp_label_scaler)
    plot_bicm(m.bicm_model, p, dL.bicm_test_loader, dL.bicm_label_scaler, "hourly")
    plot_enba(m.enba_model, dL.enba_test_loader, dL.enba_label_scaler)
    plot_rtms(m.rtms_model, dL.rtms_test_loader, dL.rtms_label_scaler)
    plot_rtmo(m.rtmo_model, dL.rtmo_test_loader, dL.rtmo_label_scaler)