# -*- coding: utf-8 -*-
"""
Created on Fri Jun 24 11:57:14 2022

@author: hliu
"""
import numpy as np


# %% 1) SIF at Leaf level
def sip_cx(Kall, Kab, Cdm, Cab, phi, wle, wlf):
    kChlrel = np.zeros_like(Kall)
    kChlrel[Kall > 0] = Cab * Kab[Kall > 0] / Kall[Kall > 0]

    w0S = np.exp(-Kall / (Cdm * 600))

    fLMA = 2519.6 * Cdm
    p = 1 - (1 - np.exp(-fLMA)) / (fLMA)
    w = w0S * (1 - p) / (1 - p * w0S)
    k = 1 - w

    kChl = kChlrel * k

    sigmoid = 1 / (1 + np.exp(-wlf / 10) @ np.exp(wle / 10))

    Cab_size = 0.38 + 0.0026 * Cab

    Mf = 0.5 * Cab_size * phi @ kChl * sigmoid
    Mb = 0.5 * Cab_size * phi @ kChl * sigmoid

    M = Mf + Mb

    return M


def creat_sif_matrix(phiI, phiII, Cab, Cm, Kab, Kall):
    Kall = Kall.reshape(-1, 1)
    X_Cm = Cm

    phiI = phiI[240: 451]
    phiII = phiII[240: 451]

    wle = np.arange(400, 751).reshape(1, -1)
    wlf = np.arange(640, 851).reshape(-1, 1)

    Kall_cx = Kall[0:351].reshape(1, -1)
    Kab_cx = Kab[0:351].reshape(1, -1)

    # excitation-fluorescence matrices
    MI = sip_cx(Kall_cx, Kab_cx, Cm, Cab, phiI, wle, wlf)
    MII = sip_cx(Kall_cx, Kab_cx, Cm, Cab, phiII, wle, wlf)

    fLMA = 2765.0 * Cm
    p = 1 - (1 - np.exp(-fLMA)) / (fLMA)
    q = 1 - np.exp(-26.63 * X_Cm - 1.52)

    pL = min(p, 0.9)
    ps = max(0, (p - pL) / (1 - pL))

    w0 = np.exp(-Kall / (Cm * 600))
    w0L = w0 * (1 - ps) / (1 - ps * w0)

    w0L_diag = np.diag(w0L.flatten())
    Mf_diag = np.zeros_like(w0L_diag)

    W_diag = w0L_diag

    Mf_diag = Mf_diag[0:451, 0:451]
    W_diag = W_diag[0:451, 0:451]

    return MII, MI, W_diag, Mf_diag, pL, q


def cal_sif_leaf(d, fqe_pars):
    MII, MI, W_diag, Mf_diag, pL, q = d.MII, d.MI, d.W_diag, d.Mf_diag, d.pL, d.q
    [fqe2u, fqe1u, fqe2h, fqe1h] = fqe_pars

    output = []
    for fqe, M in zip([fqe2u, fqe1u, fqe2h, fqe1h], [MII, MI, MII, MI]):
        Mf_diag[240:451, 0:351] = M * fqe

        M_diag = W_diag + Mf_diag

        fhemi_sum = (M_diag * (1 - pL)) @ \
                    ((np.linalg.matrix_power(M_diag * pL, 100) - np.eye(451)) @ \
                     (np.linalg.inv(M_diag * pL - np.eye(451))))

        fhemi_sum_up = 1 / 2 * fhemi_sum + \
                       1 / 2 * (M_diag * (1 - pL) * q) @ \
                       ((np.linalg.matrix_power(M_diag * pL * abs(q), 100) - np.eye(451)) @ \
                        (np.linalg.inv(M_diag * pL * abs(q) - np.eye(451))))

        fhemi_sum_dn = fhemi_sum - fhemi_sum_up

        fhemi_sum_zero = (W_diag * (1 - pL)) @ \
                         ((np.linalg.matrix_power(W_diag * pL, 100) - np.eye(451)) @ \
                          (np.linalg.inv(W_diag * pL - np.eye(451))))

        fhemi_sum_up_zero = 1 / 2 * fhemi_sum_zero + \
                            1 / 2 * (W_diag * (1 - pL) * q) @ \
                            ((np.linalg.matrix_power(W_diag * pL * abs(q), 100) - np.eye(451)) @ \
                             (np.linalg.inv(W_diag * pL * abs(q) - np.eye(451))))

        fhemi_sum_dn_zero = fhemi_sum_zero - fhemi_sum_up_zero

        fhemi_sum = fhemi_sum[240:451, 0:351]
        fhemi_sum_up = fhemi_sum_up[240:451, 0:351]
        fhemi_sum_dn = fhemi_sum_dn[240:451, 0:351]

        fhemi_sum_zero = fhemi_sum_zero[240:451, 0:351]
        fhemi_sum_up_zero = fhemi_sum_up_zero[240:451, 0:351]
        fhemi_sum_dn_zero = fhemi_sum_dn_zero[240:451, 0:351]

        SIP_SIF = fhemi_sum - fhemi_sum_zero
        SIP_SIF_up = fhemi_sum_up - fhemi_sum_up_zero  # Mf
        SIP_SIF_dn = fhemi_sum_dn - fhemi_sum_dn_zero  # Mb

        output.append(SIP_SIF_up)
        output.append(SIP_SIF_dn)

    """
    output: [Mf2u, Mb2u, Mf1u, Mb1u, Mf2h, Mb2h, Mf1h, Mb1h]
    """
    return output[0:4], output[4:8]


# %% 2) SIF at Canopy level
def sif_leaf_matrix(Mb_diag, Mf_diag, Esolar):
    [Qins, Qind, fEsuno, fEskyo] = Esolar
    Qins = Qins[0:451]
    Qind = Qind[0:451]

    Mbu = Mb_diag @ Qins
    Mfu = Mf_diag @ Qins

    Mbh = Mb_diag @ Qind
    Mfh = Mf_diag @ Qind

    return [Mbu, Mfu, Mbh, Mfh]


def sif_canopy_matrix(Mf_diag, M_diag, p, aleaf_diag):
    common_term1 = np.linalg.inv(M_diag * p - np.eye(451))
    common_term2 = np.linalg.matrix_power(M_diag * p, 11)

    f_dh = M_diag @ (common_term2 - M_diag * p) @ common_term1
    f_dw = M_diag @ (common_term2 - np.eye(451)) @ common_term1
    f_fy = Mf_diag @ (common_term2 - np.eye(451)) @ common_term1
    f_ap = aleaf_diag @ (common_term2 - np.eye(451)) @ common_term1

    f_fy_s = Mf_diag @ (common_term2 - M_diag * p) @ common_term1

    return [f_dh, f_dw, f_fy, f_ap, f_fy_s]


def cal_rtm_sif(Msys, fs, Esolar, leaf, soil, canopy_pars, dir_pars, hemi_pars, dif_pars, hemi_dif_pars):
    [Mbu, Mfu, Mbh, Mfh] = Msys

    [f_dh, f_dw, f_fy, f_ap, f_fy_s] = fs

    rho_l, tau_l = leaf
    rs = soil

    [Qins, Qind, _, _] = Esolar
    Qins = Qins[0:451]
    Qind = Qind[0:451]

    [i0, iD, p, rho_obs, rho_hemi, tv, kc, kg] = canopy_pars

    [sob, sof, K] = dir_pars
    [sob_vsla, sof_vsla, kgd] = hemi_pars
    [sob_vsla_dif, sof_vsla_dif, kg_dif] = dif_pars
    [sob_vsla_hemi_dif, sof_vsla_hemi_dif, kgd_dif] = hemi_dif_pars

    t0 = 1 - i0
    td = 1 - iD

    """
    step 1: Direct radiation in BS
    """
    Qfdir_x = Qins * rho_l * (sob * kc / K) + Qins * tau_l * (sof * kc / K) + Mbu * (sob * kc / K) + Mfu * (
            sof * kc / K)
    Qfhemi_x = Qins * rho_l * sob_vsla + Qins * tau_l * sof_vsla + Mbu * sob_vsla + Mfu * sof_vsla

    Qfdir_sum = Qfdir_x + f_dh @ Qins * i0 * rho_obs
    Qfhemi_sum = Qfhemi_x + f_dh @ Qins * i0 * rho_hemi

    Qfdown_sum = f_dw @ Qins * i0 * rho_hemi
    Qfyld_sum = f_fy @ Qins * i0

    """
    step 2: Diffuse radiation in BS
    """
    Qfdir_dx = Qind * rho_l * sob_vsla_dif + Qind * tau_l * sob_vsla_dif + Mbh * sob_vsla_dif + Mfh * sob_vsla_dif
    Qfhemi_dx = Qind * rho_l * sob_vsla_hemi_dif + Qind * tau_l * sof_vsla_hemi_dif + Mbh * sof_vsla_hemi_dif + Mfh * sof_vsla_hemi_dif

    Qfdir_d_sum = Qfdir_dx + f_dh @ Qind * iD * rho_obs
    Qfhemi_d_sum = Qfhemi_dx + f_dh @ Qind * iD * rho_hemi

    Qfdown_d_sum = f_dw @ Qind * iD * rho_hemi
    Qfyld_d_sum = f_fy @ Qind * iD

    """
    step 3: Canopy-soil interaction
    """
    # the first time of interaction of soil and canopy
    Qdown_bs = Qins * t0 + Qind * td + Qfdown_sum + Qfdown_d_sum
    Qind_s = Qdown_bs * rs

    Qdown_bs_hot = Qins * t0
    Qind_s_hot = Qdown_bs_hot * rs

    Qdown_bs_d = Qind * td + Qfdown_sum + Qfdown_d_sum
    Qind_s_d = Qdown_bs_d * rs

    Qsig_s = Qind_s_hot * iD + Qind_s_d * iD
    # Qs_sg  = M_diag@(np.linalg.matrix_power(M_diag*p,11)-np.eye(451))@(np.linalg.inv(M_diag*p-np.eye(451)))@Qsig_s
    Qs_sg = f_dw @ Qsig_s

    Qfdir_sx = (Qfdown_sum + Qfdown_d_sum) * rs * tv + Qins * rs * kg + Qind * rs * kg_dif
    Qfhemi_sx = (Qfdown_sum + Qfdown_d_sum) * rs * td + Qins * rs * kgd + Qind * rs * kgd_dif

    Qfdir_s_sum = Qfdir_sx + Qs_sg * rho_obs
    Qfhemi_s_sum = Qfhemi_sx + Qs_sg * rho_hemi
    Qfdown_s_sum = Qs_sg * rho_hemi
    # Qfyld_s_sum  = Mf_diag@(np.linalg.matrix_power(M_diag*p,11)-M_diag*p)@(np.linalg.inv(M_diag*p-np.eye(451)))@Qsig_s
    Qfyld_s_sum = f_fy @ Qsig_s

    # the second time of interaction of soil and canopy
    Qind_s = Qfdown_s_sum * rs  # update

    Qsig_s = Qind_s * iD
    # Qs_sg  = M_diag@(np.linalg.matrix_power(M_diag*p,11)-np.eye(451))@(np.linalg.inv(M_diag*p-np.eye(451)))@Qsig_s
    Qs_sg = f_dw @ Qsig_s

    Qfdir_sx = Qind_s * tv
    Qfhemi_sx = Qind_s * td

    Qfdir_s_sum += Qfdir_sx + Qs_sg * rho_obs
    Qfhemi_s_sum += Qfhemi_sx + Qs_sg * rho_hemi
    Qfdown_s_sum = Qs_sg * rho_hemi
    # Qfyld_s_sum  += Mf_diag@(np.linalg.matrix_power(M_diag*p,11)-M_diag*p)@(np.linalg.inv(M_diag*p-np.eye(451)))@Qsig_s
    Qfyld_s_sum += f_fy_s @ Qsig_s

    # the third time of interaction of soil and canopy
    Qind_s = Qfdown_s_sum * rs  # update
    Qsig_s = Qind_s * iD
    # Qs_sg  = M_diag@(np.linalg.matrix_power(M_diag*p,11)-np.eye(451))@(np.linalg.inv(M_diag*p-np.eye(451)))@Qsig_s
    Qs_sg = f_dw @ Qsig_s

    Qfdir_sx = Qind_s * tv
    Qfhemi_sx = Qind_s * td

    Qfdir_s_sum += Qfdir_sx + Qs_sg * rho_obs
    Qfhemi_s_sum += Qfhemi_sx + Qs_sg * rho_hemi
    Qfdown_s_sum = Qs_sg * rho_hemi
    # Qfyld_s_sum  += Mf_diag@(np.linalg.matrix_power(M_diag*p,11)-M_diag*p)@(np.linalg.inv(M_diag*p-np.eye(451)))@Qsig_s
    Qfyld_s_sum += f_fy_s @ Qsig_s

    """
    Part 4: Output
    """
    Qfdir_bs = Qfdir_sum + Qfdir_d_sum
    Qfhemi_bs = Qfhemi_sum + Qfhemi_d_sum
    Qfyld_bs = Qfyld_sum + Qfyld_d_sum

    Qfdir_ss = Qfdir_s_sum
    Qfhemi_ss = Qfhemi_s_sum
    Qfyld_ss = Qfyld_s_sum

    Qfdir_all = Qfdir_bs + Qfdir_ss
    Qfhemi_all = Qfhemi_bs + Qfhemi_ss
    Qfyld_all = Qfyld_bs + Qfyld_ss

    return Qfdir_all, Qfhemi_all, Qfyld_all


def cal_canopy_sif(d, x, Esolar, M_pars, rtm_o_dict, hemi_dif_pars):
    rho_l, tau_l = d.leaf
    rho_l, tau_l = (rho_l[0:451, x % 365]).reshape(-1, 1), (tau_l[0:451, x % 365]).reshape(-1, 1)

    leaf = [rho_l, tau_l]
    soil = (d.soil[0:451]).reshape(-1, 1)

    [Mf2, Mb2, Mf1, Mb1] = M_pars

    d.MfII_diag[240:451, 0:351] = Mf2
    d.MbII_diag[240:451, 0:351] = Mb2

    d.MfI_diag[240:451, 0:351] = Mf1
    d.MbI_diag[240:451, 0:351] = Mb1

    d.MII_diag = d.wleaf_diag + d.MfII_diag + d.MbII_diag
    d.MI_diag = d.wleaf_diag + d.MfI_diag + d.MbI_diag
    d.MA_diag = d.wleaf_diag

    d.MII_diag_q = np.linalg.matrix_power(d.MII_diag, 2).astype(np.float32)
    d.MI_diag_q = np.linalg.matrix_power(d.MI_diag, 2).astype(np.float32)
    d.MA_diag_q = np.linalg.matrix_power(d.MA_diag, 2).astype(np.float32)

    p = rtm_o_dict['p']

    canopy_pars = [rtm_o_dict['i0'], rtm_o_dict['iD'], p, rtm_o_dict['rho2'], rtm_o_dict['rho_hemi'], rtm_o_dict['tv'],
                   rtm_o_dict['kc'], rtm_o_dict['kg']]
    dir_pars = [d.sob[x], d.sof[x], d.ko[x]]
    hemi_pars = [rtm_o_dict['sob_vsla'], rtm_o_dict['sof_vsla'], rtm_o_dict['kgd']]
    dif_pars = [rtm_o_dict['sob_vsla_dif'], rtm_o_dict['sof_vsla_dif'], rtm_o_dict['kg_dif']]

    MsysII = sif_leaf_matrix(d.MbII_diag, d.MfII_diag, Esolar)
    MsysI = sif_leaf_matrix(d.MbI_diag, d.MfI_diag, Esolar)
    MsysA = sif_leaf_matrix(d.MbA_diag, d.MfA_diag, Esolar)

    fs_II = sif_canopy_matrix(d.MfII_diag, d.MII_diag, p, d.aleaf_diag)
    fs_I = sif_canopy_matrix(d.MfI_diag, d.MI_diag, p, d.aleaf_diag)
    fs_A = sif_canopy_matrix(d.MfA_diag, d.MA_diag, p, d.aleaf_diag)

    Qfdir_II, Qfhemi_II, Qfyld_II = cal_rtm_sif(MsysII, fs_II, Esolar, leaf, soil, canopy_pars, dir_pars,
                                                hemi_pars, dif_pars, hemi_dif_pars)
    Qfdir_I, Qfhemi_I, Qfyld_I = cal_rtm_sif(MsysI, fs_I, Esolar, leaf, soil, canopy_pars, dir_pars, hemi_pars,
                                             dif_pars, hemi_dif_pars)
    Qfdir_A, Qfhemi_A, Qfyld_A = cal_rtm_sif(MsysA, fs_A, Esolar, leaf, soil, canopy_pars, dir_pars, hemi_pars,
                                             dif_pars, hemi_dif_pars)

    SRTE_Fs_fdir2 = Qfdir_II - Qfdir_A
    SRTE_Fs_fdir1 = Qfdir_I - Qfdir_A
    SRTE_Fs_fdir_all = SRTE_Fs_fdir1 + SRTE_Fs_fdir2

    SRTE_Fs_fhemi2 = Qfhemi_II - Qfhemi_A
    SRTE_Fs_fhemi1 = Qfhemi_I - Qfhemi_A
    SRTE_Fs_fhemi_all = SRTE_Fs_fhemi1 + SRTE_Fs_fhemi2

    SRTE_Fs_fyld2 = Qfyld_II
    SRTE_Fs_fyld1 = Qfyld_I
    SRTE_Fs_fyld_all = Qfyld_I + Qfyld_II

    return np.mean(SRTE_Fs_fdir1[240:450]), np.mean(SRTE_Fs_fdir2[240:450]), np.mean(SRTE_Fs_fdir_all[240:450]), \
           np.mean(SRTE_Fs_fhemi1[240:450]), np.mean(SRTE_Fs_fhemi2[240:450]), np.mean(SRTE_Fs_fhemi_all[240:450]), \
           np.mean(SRTE_Fs_fyld1[240:450]), np.mean(SRTE_Fs_fyld2[240:450]), np.mean(SRTE_Fs_fyld_all[240:450])
