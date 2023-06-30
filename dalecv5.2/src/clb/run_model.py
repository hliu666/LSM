import data_class as dc
import sys 
sys.path.append("../model")
import mod_class as mc
import numpy as np
import datetime 

def remove_outlier(arr):
    q75,q25 = np.percentile(arr,[75,25])
    intr_qr = q75-q25
    
    max = q75+(1.5*intr_qr)
    min = q25-(1.5*intr_qr)
    
    arr = arr[arr > min]
    arr = arr[arr < max]
    
    return arr.mean()

def cal_err(merge_arr):
    if np.nanmean(merge_arr[:,0]) >= 0:
        nee_err = 0.62 + 0.63*merge_arr[:,0]	
    else:
        nee_err = 1.42 + 0.19*merge_arr[:,0]	
    return nee_err
    
"""
CI_flag
    0: CI varied with the zenith angle
    1: CI as a constant 
    2: Without considering CI effect            
"""
def run_model(pars):      
    ci_flag = 1
    site = "HARV"
    
    d = dc.DalecData(2019, 2022, site, ci_flag, pars)
    m = mc.DalecModel(d)
    model_output, nee_y, lst_y, _ = m.mod_list(d.xb)
        
    cost_value = 0.0
    #print(cost_value)
    
    #define ecological constraints
    #EDC1:LAI    
    lai = model_output[:,-1]
    # The deviation of observed LAI
    d.lai_data["id"] = d.lai_data.apply(lambda x: (datetime.datetime(int(x['year']), int(x['month']), int(x['day']))-datetime.datetime(2019, 1, 1)).days, axis=1)
    d.lai_data["simLAI"] = lai[np.array(d.lai_data["id"])]
    d.lai_data["err"] = d.lai_data.apply(lambda x: ((x['LAI']-x['simLAI'])/x['errLAI'])**2, axis=1)
    cost_value += d.lai_data['err'].mean()
    print("LAI1: {0}".format(cost_value))
    
    # Set the reality error of measured LAI
    # Set the reality error of measured LAI
    for y in range(0,3):
        yid = y*365
        lai_s1,  lai_s2  = max(lai[yid+195:yid+225]),   min(lai[yid+195:yid+225])
        lai_ns1, lai_ns2 = np.nanmean(lai[yid:yid+60]), np.nanmean(lai[yid+330:yid+365])
        if (lai_s1 > 6) or (lai_s2 < 4.5):
            cost_value += (max(lai_s1-6, 4.5-lai_s2)/0.5)**2
        if (lai_ns1 > 0) or (lai_ns2 > 0):
            cost_value += (max(lai_ns1, lai_ns2)/0.4)**2
    print("LAI2: {0}".format(cost_value))
    
    if np.isnan(cost_value):
        cost_value = 10000
    else:  
        #EDC2:NEE
        nee = np.array(d.vrfy_data['nee'][0:26280])
        par = np.array(d.flux_data['PAR_up'][0:26280])
        merge = np.hstack((nee.reshape(-1,1), nee_y.reshape(-1,1), par.reshape(-1,1)))
        merge = merge[~np.isnan(merge).any(axis=1)]  
        merge_night = merge[np.where(merge[:,2] <= 0)]
        merge_day   = merge[np.where(merge[:,2] > 0)]
        nee_night_err = cal_err(merge_night)
        nee_day_err   = cal_err(merge_day)              
        rmse_night = (((merge_night[:,0] - merge_night[:,1])/nee_night_err) ** 2)*4
        rmse_day   = (((merge_day[:,0]   - merge_day[:,1])/nee_day_err) ** 2)*2
        rmse_night = remove_outlier(rmse_night)
        rmse_day   = remove_outlier(rmse_day)
        cost_value += (rmse_night + rmse_day)
        print("NEE: {0}".format(cost_value))
       
        #EDC3:Phenological constraints 
        gpp_fit = np.genfromtxt("../../data/parameters/fgpp.txt")
        gpp_sim = model_output[:,-2]
        cost_value += np.nanmean(((gpp_fit-gpp_sim)/0.5)**2)
        print("Phenology: {0}".format(cost_value))
        
        """     
        #EDC5:the GPP allocated fraction to Croo and Cfol (directly or via the labile C pool) are within a factor of 5 of each other
        ffol, froo, flab = model_output[0,2], model_output[0,3], model_output[0,11]
        ffs = ffol + flab
        if ((ffol+flab) < 0.2*froo) or ((ffol+flab) > 5*froo):
            cost_value += max((0.2*froo-ffs)/(0.2*ffs), (ffs-5*froo)/(0.2*ffs)) #*scaling_factor/10.0 
        print("EDC5: {0}".format(cost_value))
           
        #EDC6:Mean fine root and foliar pool sizes are within a factor of 5 of each other 
        Cfol_mean = np.nanmean(model_output[:,17])
        Croo_mean = np.nanmean(model_output[:,18])
        if (Croo_mean < 0.2*Cfol_mean) or (Croo_mean > 5*Cfol_mean):
            cost_value += max((0.2*Cfol_mean-Croo_mean)/Croo_mean, (Croo_mean-5*Cfol_mean)/Croo_mean) 
        print("EDC6: {0}".format(cost_value))
    
        #EDC7:Maximum growth rate for each carbon pool 
        Gmax = 0.1
        for i in range(16,22):
            bench_year = np.nanmean(model_output[0:365,i])
            for y in range(1,3):
                yid = y*365
                pool_year = np.nanmean(model_output[yid+0:yid+365,i])
                ratio = pool_year/bench_year
                if ratio >= (1+Gmax*y/10.0):
                    cost_value += (ratio-(1+Gmax*y/10.0))
        print("EDC7: {0}".format(cost_value))
    
        #EDC8:Carbon pool exponential decay trajectories  
        c0 = 365.25*3/np.log(2)
        for i in range(16,22):
            for y in range(0,2):
                yid1 = y*365
                yid2 = (y+1)*365
                deltaC0 = np.nanmean(model_output[yid2+0:yid2+365,i])-np.nanmean(model_output[yid1+0:yid1+365,i])
                deltaC1 = np.nanmean(model_output[yid2+1:yid2+365+1,i])-np.nanmean(model_output[yid1+1:yid1+365+1,i])
                c = np.log(deltaC1/deltaC0)
                if c >= -365.25*3/np.log(2):
                    cost_value += abs((c-c0)/c0)
        print("EDC8: {0}".format(cost_value))
        """
        #EDC 9-12: 
        lst_ymean = np.nanmean([np.nanmean(lst_y[m: m+1]) for m in range(0, len(lst_y), 24)])
        gpp_y = np.nanmean(model_output[:,22])   
        fauto, ffol, froo, flab = model_output[0,1], model_output[0,2], model_output[0,3], model_output[0,11]
        fwoo = 1-fauto-ffol-flab
        theta_min, theta_lit, theta_som, theta_woo = model_output[0,0], model_output[0,7], model_output[0,8], model_output[0,5]
        Theta = model_output[0,9]
        
        Csom_inf = (fwoo+(ffol+froo+flab)*theta_min)*gpp_y/((theta_min+theta_lit)*theta_som*np.exp(Theta*lst_ymean))
        Clit_inf = ((ffol+froo+flab)*gpp_y)/(theta_lit*np.exp(Theta*lst_ymean))
        Cwoo_inf = (fwoo*gpp_y)/theta_woo
        Croo_inf = (froo*gpp_y)/theta_woo
        
        Csom_0 = model_output[0,21]
        Clit_0 = model_output[0,20]
        Cwoo_0 = model_output[0,19]
        Croo_0 = model_output[0,18]
    
        if (Csom_inf < 0.1*Csom_0) or (Csom_inf > 10*Csom_0):
           cost_value += abs(max(0.1*Csom_0-Csom_inf, Csom_inf-10*Csom_0)/(0.2*Csom_inf))
           print("EDC9 Csom: {0}".format(cost_value))
        if (Clit_inf < 0.1*Clit_0) or (Clit_inf > 10*Clit_0):
           cost_value += abs(max(0.1*Clit_0-Clit_inf, Clit_inf-10*Clit_0)/(0.2*Clit_inf))
           print("EDC10 Clit: {0}".format(cost_value))
        if (Cwoo_inf < 0.1*Cwoo_0) or (Cwoo_inf > 10*Cwoo_0):
           cost_value += abs(max(0.1*Cwoo_0-Cwoo_inf, Cwoo_inf-10*Cwoo_0)/(0.2*Cwoo_inf)) 
           print("EDC11 Cwoo: {0}".format(cost_value))
        if (Croo_inf < 0.1*Croo_0) or (Croo_inf > 10*Croo_0):
           cost_value += abs(max(0.1*Croo_0-Croo_inf, Croo_inf-10*Croo_0)/(0.2*Croo_inf))
           print("EDC12 Croo: {0}".format(cost_value))
    
    nee_y = np.append(cost_value, nee_y)
    return model_output, nee_y


#import time
#start = time.time()

#import cProfile
#cProfile.run('dc.DalecData(2019, 2022, site, ci_flag, "nee")')
#d = dc.DalecData(2019, 2022, site, ci_flag, 'nee')
#m = mc.DalecModel(d)
#cProfile.run('m.mod_list(d.xb)')
"""
import spotpy
dbname = "SCEUA"
results = spotpy.analyser.load_csv_results('{0}'.format(dbname))

bestindex,bestobjf = spotpy.analyser.get_minlikeindex(results)
best_model_run = results[bestindex]
fields=['parp0', 'parp1', 'parp2', 'parp3', 'parp4', 'parp5', 'parp6', 'parp7', 'parp8', 'parp9', 'parp10', 'parp11',\
        'parp12', 'parp13', 'parp14', 'parp15', 'parp16']
pars = list(best_model_run[fields])
        
model_output, nee_y = run_model(pars)
import matplotlib.pyplot as plt
plt.figure(figsize=(8,4))
plt.title('NEE')
plt.plot(nee_y, 'k.')

"""

#end = time.time()
#print(end - start)

