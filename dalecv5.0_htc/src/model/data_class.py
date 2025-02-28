import numpy as np
import pandas as pd
import collections as col

from RTM_initial import sip_leaf, soil_spectra, atmoE
from RTM_initial import cal_lidf, weighted_sum_over_lidf_vec, CIxy
from RTM_initial import hemi_initial, dif_initial, hemi_dif_initial
from RTM_initial import calc_sun_angles
from Ebal_initial import calc_extinc_coeff_pars

xrange = range
class DalecData:
    """
    Data class for the DALEC2 model
    """
    def __init__(self, start_yr, end_yr, site, datas, pars, ci_flag, ob_str=None, k=None):
        """ Extracts data from netcdf file
        :param start_yr: year for model runs to begin as an integer (year)
        :param end_yr: year for model runs to end as an integer (year)
        :param ob_str: string containing observations that will be assimilated (Currently only NEE available)
        :param dat_file: location of csv file to extract data from
        :param k: int to repeat data multiple times
        :return:
        """
        #print(site)
        # Extract the data
        [flux_arr, vrfy_arr, brf_arr, wds_arr, tis_arr, B, rsr_red, rsr_nir, rsr_sw1, rsr_sw2, prospectpro, soil, TOCirr] = datas

        self.flux_data = flux_arr[(flux_arr['year'] >= start_yr) & (flux_arr['year'] < end_yr)]
        self.vrfy_data = vrfy_arr[(vrfy_arr['year'] >= start_yr) & (vrfy_arr['year'] < end_yr)]
        self.brf_data  = brf_arr[(brf_arr['year'] >= start_yr) & (brf_arr['year'] < end_yr)]        
        self.wds_data  = wds_arr[(wds_arr['year'] >= start_yr) & (wds_arr['year'] < end_yr)]
        
        # 'Daily temperatures degC'
        self.t_mean = self.flux_data['TA']
        self.vpd = self.flux_data['VPD']*100

        # 'Driving Data'
        self.sw = self.flux_data['SW']  
        self.par = self.flux_data['PAR_up']
        
        self.D = self.flux_data['doy']  # day of year
        self.year = self.flux_data['year']  # Year
        self.month = self.flux_data['month']  # Month
        self.date = self.flux_data['day']  # Date in month
        
        self.len_run = int(len(self.flux_data)/24)
        self.start_yr = start_yr
        self.end_yr = end_yr
        self.time_step = np.arange(self.len_run)
        
        """
        HARV forest parameters
        """
        # I.C. for carbon pools gCm-2     range
        self.clab = pars[0]              # (10,1e3)
        self.cf = pars[1]               # (10,1e3)
        self.cr = pars[2]            # (10,1e3)
        self.cw = pars[3]            # (3e3,3e4)
        self.cl = pars[4]             # (10,1e3)
        self.cs = pars[5]             # (1e3, 1e5)
        
        # Parameters for optimization  
        self.p0 = pars[6]   # theta_min, cl to cs decomp      (1e-5 - 1e-2) day-1
        self.p1 = pars[7]    # f_auto, fraction of GPP respired  (0.3 - 0.7)
        self.p2 = pars[8]   # f_fol, frac GPP to foliage        (0.01 - 0.5)
        self.p3 = pars[9]   # f_roo, frac GPP to fine roots    (0.01 - 0.5)
        self.p4 = pars[10]   # clspan, leaf lifespan               (1.0001 - 5)
        self.p5 = pars[11]   # theta_woo, wood C turnover      (2.5e-5 - 1e-3) day-1
        self.p6 = pars[12]   # theta_roo, root C turnover rate(1e-4 - 1e-2) day-1
        self.p7 = pars[13]   # theta_lit, litter C turnover     (1e-4 - 1e-2) day-1
        self.p8 = pars[14]   # theta_som, SOM C turnover       (1e-7 - 1e-3) day-1
        self.p9 = pars[15]   # Theta, temp dependence exp fact(0.018 - 0.08)
        self.p10 = pars[16]   # d_onset, clab release date       (1 - 365) (60,150)
        self.p11 = pars[17]   # f_lab, frac GPP to clab           (0.01 - 0.5)
        self.p12 = pars[18]   # cronset, clab release period      (10 - 100)
        self.p13 = pars[19]   # d_fall, date of leaf fall        (1 - 365) (242,332)
        self.p14 = pars[20]   # crfall, leaf fall period          (10 - 100)
        self.p15 = pars[21]   # Vcmax25
        self.p16 = pars[22]   # clumping index 
        self.p17 = pars[23] 
        self.p18 = np.inf
        
        """ 
        Constants of Photosynthesis 
        """    
        self.Vcmax25 = self.p15
        self.BallBerrySlope = pars[24]
        
        self.ca = 390.0  # atmospheric carbon
        self.cs = 250.0  # 
        
        self.ea = 40.0
        self.eb = 27   # 
        
        self.o = 209.0
        self.p = 970.0       
       
        """ 
        Parameters of Leaf-level Radiative Transfer Model  
        """           
        self.lma = 65.18  # clma, leaf mass per area          (81 - 120) g C m-2
        
       
        self.Cab    = np.full((1, 365+366+365), pars[25] ) #chlorophyll a+b content (mug cm-2).
        self.Car    = np.full((1, 365+366+365), pars[26])  #carotenoids content (mug cm-2).
        self.Cbrown = np.full((1, 365+366+365), pars[27]) #brown pigments concentration (unitless).
        self.Cw     = np.full((1, 365+366+365), pars[28]) #equivalent water thickness (g cm-2 or cm).
        self.Cm     = np.full((1, 365+366+365), self.p15/10000.0) #dry matter content (g cm-2).
        self.Ant    = np.full((1, 365+366+365), pars[29]) #Anthocianins concentration (mug cm-2). 
        self.Alpha  = np.full((1, 365+366+365), 160)   #constant for the the optimal size of the leaf scattering element   
        self.fLMA_k = np.full((1, 365+366+365), 2519.65) 
        self.gLMA_k = np.full((1, 365+366+365), -631.54) 
        self.gLMA_b = np.full((1, 365+366+365), 0.0086) 
        
        #self.leaf = sip_leaf_spectral(np.array(data7['refl']), np.array(data7['tran']))
        #self.Kab, self.nr, self.Kall, self.leaf = sip_leaf(self.Cab, self.Car, self.Cbrown, self.Cw, self.Cm, self.Ant, self.Alpha)
        self.leaf = sip_leaf(prospectpro, self.Cab, self.Car, self.Cbrown, self.Cw, self.Cm, self.Ant, self.Alpha, self.fLMA_k, self.gLMA_k, self.gLMA_b)

        """ 
        Parameters of soil model
        """
        self.soil = soil_spectra(soil)   
        
        """ 
        The dictionary of carbon parameters  
        """ 
        self.param_dict = col.OrderedDict([('theta_min', self.p0),
                                          ('f_auto', self.p1), ('f_fol', self.p2),
                                          ('f_roo', self.p3), ('clspan', self.p4),
                                          ('theta_woo', self.p5), ('theta_roo', self.p6),
                                          ('theta_lit', self.p7), ('theta_som', self.p8),
                                          ('Theta', self.p9), ('d_onset', self.p10), ('f_lab', self.p11),
                                          ('cronset', self.p12), ('d_fall', self.p13),
                                          ('crfall', self.p14), ('clma', self.lma),
                                          ('clab', self.clab), ('cf', self.cf),
                                          ('cr', self.cr), ('cw', self.cw), ('cl', self.cl),
                                          ('cs', self.cs)])
        self.pvals = np.array(self.param_dict.values())

        self.edinburgh_median = np.array([self.p0,    self.p1,   self.p2,
                                          self.p3,    self.p4,   self.p5,
                                          self.p6,    self.p7,   self.p8,
                                          self.p9,    self.p10,  self.p11,
                                          self.p12,   self.p13,  self.p14,
                                          self.lma,   self.clab, self.cf,    
                                          self.cr,    self.cw,   self.cl, 
                                          self.cs,    0.0,  0.0])

        self.xb = self.edinburgh_median

        # misc
        self.radconv = 365.25 / np.pi
        
        """
        The spectral response curve 
        """
        self.rsr_red = rsr_red#np.genfromtxt("../../data/parameters/rsr_red.txt")
        self.rsr_nir = rsr_nir#np.genfromtxt("../../data/parameters/rsr_nir.txt")
        self.rsr_sw1 = rsr_sw1#np.genfromtxt("../../data/parameters/rsr_swir1.txt")
        self.rsr_sw2 = rsr_sw2#np.genfromtxt("../../data/parameters/rsr_swir2.txt")
        
        """ 
        Parameters of sun's spectral curve
        """
        self.wl, self.atmoMs = atmoE(TOCirr)

        """
        Sun-sensor geometry
        """        
        lat, lon = 42.54, -72.17 #HARV
        stdlon = (int(lon/15) + -1*(1 if abs(lon)%15>7.5 else 0))*15
        
        #non-leap/leap year
        doy = np.repeat(np.arange(1,367), 24)
        ftime = np.tile(np.arange(0,24),366)
        tts_nl, _ = calc_sun_angles(lat, lon, stdlon, doy, ftime)
        tts_nl[tts_nl>90] = 90
        
        self.tts = np.concatenate([tts_nl[:8760], tts_nl, tts_nl[:8760]], axis=0)
        self.tto = np.full(24*(365+366+365), 45.0)
        self.psi = np.full(24*(365+366+365), 90)        
        self.psi[self.psi > 180] = abs(self.psi[self.psi > 180] - 360)
        
        """
        Parameters of leaf angle distribution
        """
        self.lidfa = self.p17 # float Leaf Inclination Distribution at regular angle steps. 
        self.lidfb = self.p18 # float Leaf Inclination Distribution at regular angle steps. 
        self.lidf  = cal_lidf(self.lidfa, self.lidfb)

        """
        Clumping Index (CI_flag):
            0: CI varied with the zenith angle
            1: CI as a constant 
            2: Without considering CI effect            
        """
        self.CI_thres = self.p16
        self.CI_flag = ci_flag
        print("CI_flag is {0}".format(self.CI_flag))
        self.CIs = CIxy(self.CI_flag, self.tts, self.CI_thres)
        self.CIo = CIxy(self.CI_flag, self.tto, self.CI_thres)
 
        """ 
        Parameters of Canopy-level Radiative Transfer Model 
        """
        _, _, self.ks, self.ko, _, self.sob, self.sof = weighted_sum_over_lidf_vec(self.lidf, self.tts, self.tto, self.psi)
        #self.Ps_arr, self.Po_arr, self.int_res_arr, self.nl = dir_gap_initial_vec(self.tts, self.tto, self.psi, self.ks, self.ko, self.CIs, self.CIo)
        self.hemi_pars = hemi_initial(self.CI_flag, self.tts, self.lidf, self.CI_thres)
        self.dif_pars = dif_initial(self.CI_flag, self.tto, self.lidf, self.CI_thres)      
        self.hemi_dif_pars = hemi_dif_initial(self.CI_flag, self.lidf, self.CI_thres)

        """ 
        Parameters of Reflectance-based sun-sensor geometry for Radiative Transfer Model 
        """
        self.tts_mds = np.array(self.brf_data['tts'])
        self.tto_mds = np.array(self.brf_data['tto'])
        self.psi_mds = np.array(self.brf_data['psi'])
        
        _, _, self.ks_mds, self.ko_mds, _, self.sob_mds, self.sof_mds = weighted_sum_over_lidf_vec(self.lidf, self.tts_mds, self.tto_mds, self.psi_mds)
        
        self.CIs_mds = CIxy(self.CI_flag, self.tts_mds, self.CI_thres)
        self.CIo_mds = CIxy(self.CI_flag, self.tto_mds, self.CI_thres)
        
        """
        Wind speed  
        """
        self.wds = self.wds_data['wind_u']
        
        """
        Parameters of extinction coefficient
        """
        self.extinc_k, self.extinc_sum0 = calc_extinc_coeff_pars(self.CI_flag, self.CI_thres, self.lidf)
        