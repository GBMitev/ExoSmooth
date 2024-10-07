import pandas as pd
import numpy as np

from sklearn.linear_model import HuberRegressor
from sklearn.preprocessing import StandardScaler

from tqdm import tqdm 

from multiprocessing import cpu_count
import multiprocessing as mp
import subprocess as sp
from scipy.integrate import trapezoid 

def read_stick_output(path, predicted_shifts = True):
    
    from pandas import DataFrame
        
    with open(path) as file:
        lines = file.readlines()
    n = 0
    l = lines[n]
    while "Spectrum type = ABSORPTION" not in l:
        n+=1
        l = lines[n]

    m = n+1
    while "Total intensity" not in l:
        m+=1
        l = lines[m]

    start = n+2
    end   = m-1

    lines = lines[start:end]
    rows  = []
    for l in lines:
        rows.append(l.replace("\n","").replace("<-","").split())
    if predicted_shifts == True:
        transition_columns = ["nu","I","J_upper","E_upper","J_lower","E_lower","Unc_upper","Lifetime_upper","Lande_upper","tau_upper","e/f_upper","Manifold_upper","v_upper","Lambda_upper","Sigma_upper","Omega_upper","Type_upper","E_calc_upper","Unc_lower","Lifetime_lower","Lande_lower","tau_lower","e/f_lower","Manifold_lower","v_lower","Lambda_lower","Sigma_lower","Omega_lower","Type_lower","E_calc_lower"]
    else:
        transition_columns = ["nu","I","J_upper","E_upper","J_lower","E_lower","tau_upper","e/f_upper","Manifold_upper","v_upper","Lambda_upper","Sigma_upper","Omega_upper","tau_lower","e/f_lower","Manifold_lower","v_lower","Lambda_lower","Sigma_lower","Omega_lower"]
    
    stick = DataFrame(rows, columns = transition_columns)

    stick["nu"]            = stick["nu"]            .astype("float")
    stick["I"]             = stick["I"]             .astype("float")
    stick["J_upper"]       = stick["J_upper"]       .astype("float")
    stick["E_upper"]       = stick["E_upper"]       .astype("float")
    stick["J_lower"]       = stick["J_lower"]       .astype("float")
    stick["E_lower"]       = stick["E_lower"]       .astype("float")
    stick["tau_upper"]     = stick["tau_upper"]     .astype("str")
    stick["e/f_upper"]     = stick["e/f_upper"]     .astype("str")
    stick["Manifold_upper"]= stick["Manifold_upper"].astype("str")
    stick["v_upper"]       = stick["v_upper"]       .astype("int")
    stick["Lambda_upper"]  = stick["Lambda_upper"]  .astype("float")
    stick["Sigma_upper"]   = stick["Sigma_upper"]   .astype("float")
    stick["Omega_upper"]   = stick["Omega_upper"]   .astype("float")
    stick["tau_lower"]     = stick["tau_lower"]     .astype("str")
    stick["e/f_lower"]     = stick["e/f_lower"]     .astype("str")
    stick["Manifold_lower"]= stick["Manifold_lower"].astype("str")
    stick["v_lower"]       = stick["v_lower"]       .astype("int")
    stick["Lambda_lower"]  = stick["Lambda_lower"]  .astype("float")
    stick["Sigma_lower"]   = stick["Sigma_lower"]   .astype("float")
    stick["Omega_lower"]   = stick["Omega_lower"]   .astype("float")
    
    if predicted_shifts == True:
        stick["Unc_upper"]     = stick["Unc_upper"]     .astype("float")
        stick["Unc_lower"]     = stick["Unc_lower"]     .astype("float")
        stick["Lifetime_upper"]= stick["Lifetime_upper"].astype("float")
        stick["Lifetime_lower"]= stick["Lifetime_lower"].astype("float")
        stick["Lande_upper"]   = stick["Lande_upper"]   .astype("float")
        stick["Lande_lower"]   = stick["Lande_lower"]   .astype("float")
        stick["Type_upper"]    = stick["Type_upper"]    .astype("str")
        stick["Type_lower"]    = stick["Type_lower"]    .astype("str")
        stick["E_calc_upper"]  = stick["E_calc_upper"]  .astype("float")
        stick["E_calc_lower"]  = stick["E_calc_lower"]  .astype("float")
        
    return stick

def gaussian(x:list, mu:float, sigma:float, scale_factor = None, scale_style = "integral", dx = 1):
    '''
    Returns Gaussian distribution
    
    Inputs:
        x            = dependent variable values    : list  (float)
        mu           = mean                         : value (float)  
        sigma        = standard deviation           : value (float)  
    
    Outputs:
        Gauss      = Gaussian Distribution          : list  (float)
    '''
    Gauss = 1/(sigma*np.sqrt(2*np.pi))*np.exp(-0.5*((x-mu)/sigma)**2)

    if scale_factor is not None:
        if scale_style == "integral":
            
            current_integral_value = trapezoid(Gauss, dx = dx)
            new_integral_value = scale_factor
            scale_factor = new_integral_value/current_integral_value

            Gauss = Gauss * scale_factor.reshape(-1,1)
        elif scale_style == "mult":
            Gauss = Gauss * scale_factor
    return Gauss

def fit_HuberRegressor(nu):
    nu_diff = np.diff(nu)
    nu      = nu[1:].reshape(-1,1)

    scaler = StandardScaler()
    nu_scaled = scaler.fit_transform(nu)

    model = HuberRegressor()
    model.fit(nu_scaled, nu_diff**2)

    mean_nu = scaler.mean_ [0]
    std_nu  = scaler.scale_[0]

    model.intercept_ = model.intercept_ - (model.coef_*mean_nu/std_nu)
    model.coef_ = model.coef_/std_nu

    return model

def predict_HuberRegressor(model, nu):
    if type(nu) == np.ndarray:
        pass
    elif type(nu) != np.ndarray:
        if type(nu) == list:
            nu = np.array([*nu])
        elif type(nu) == float or type(nu) == int:
            nu = np.array([nu])
    
    if len(nu) == 1:
        nu = nu.reshape(1,-1)
    elif len(nu) > 1:
        nu = nu.reshape(-1,1)

    return np.sqrt(model.predict(nu))

def get_line_profile(wavenumber_range,wavenumber, intensity, gamma=None, sigma=None, model = None):
    # sigma = gamma / (2 * np.sqrt(2 * np.log(2)))
    if gamma != None:
        sigma = gamma / (np.sqrt(2 * np.log(2)))
    elif sigma != None:
        pass
    elif model != None:
        gamma = predict_HuberRegressor(model, wavenumber)
        sigma = gamma / (np.sqrt(2 * np.log(2)))

    line_profile = gaussian(wavenumber_range, wavenumber, sigma)    
    
    line_profile *= intensity

    return line_profile

def get_stdevs(df, J_lower, Omega_lower, min_hwhm=0, min_sigma=None):
    df_min = df[
        (df["J_lower"]==J_lower)&
        (df["Omega_lower"]==Omega_lower)
        ]
    
    grouped = df_min.groupby(["v_lower","v_upper"], as_index = False).agg(nu = ("nu","mean"))
    models = {}
    widths = pd.DataFrame(columns = [*df.columns]+["Sigma"])
    min_sigma = min_hwhm / (np.sqrt(2 * np.log(2))) if min_sigma is None else min_sigma

    for v in grouped.v_lower.unique():
        curr = grouped[grouped.v_lower == v].sort_values("v_upper")
        nu = curr.nu.to_numpy()

        model = fit_HuberRegressor(nu)
        
        models[f"{v}"] = model
        curr = df[df["v_lower"]==v]

        curr["Sigma"] = predict_HuberRegressor(model, curr["nu"].to_numpy())/(np.sqrt(2*np.log(2)))
        
        # Removing inexplicable nans
        # mean_sigma = curr[curr["Sigma"].isnull()==False]["Sigma"].mean()
        mean_sigma = curr[curr["Sigma"].isnull()==False]["Sigma"].max()
        cond = [curr["Sigma"].isnull()==True,curr["Sigma"].isnull()==False]
        vals = [mean_sigma, curr["Sigma"]]
        curr["Sigma"] = np.select(cond, vals)

        cond = [curr["Sigma"]>min_sigma, curr["Sigma"]<min_sigma]
        vals = [curr["Sigma"], min_sigma]
        curr["Sigma"] = np.select(cond, vals)
    
        widths = pd.concat([widths, curr])

    return widths

def get_stdevs_states_trans(states_trans, J_i, Omega_i, min_sigma=0, min_hwhm=None):

    df = states_trans[
        (states_trans["J_i"]==J_i)&
        (states_trans["Omega_i"]==Omega_i)
        ]
    
    grouped = df.groupby(["v_i","v_f"], as_index = False).agg(nu = ("nu","mean"))
    models = {}
    widths = pd.DataFrame(columns = [*states_trans.columns]+["HWHM"])
    
    min_hwhm  = min_sigma * (np.sqrt(2 * np.log(2))) if min_hwhm is None else min_hwhm
    # return grouped
    for v in grouped.v_i.unique():
        curr = grouped[grouped.v_i == v].sort_values("v_f")
        nu = curr.nu.to_numpy()

        model = fit_HuberRegressor(nu)
        
        models[f"{v}"] = model
        curr = states_trans[states_trans["v_i"]==v]

        curr["HWHM"] = predict_HuberRegressor(model, curr["nu"].to_numpy())/(np.sqrt(2*np.log(2)))
        
        # Removing inexplicable nans
        # mean_sigma = curr[curr["Sigma"].isnull()==False]["Sigma"].mean()
        mean_sigma = curr[curr["HWHM"].isnull()==False]["HWHM"].max()
        cond = [curr["HWHM"].isnull()==True,curr["HWHM"].isnull()==False]
        vals = [mean_sigma, curr["HWHM"]]
        curr["HWHM"] = np.select(cond, vals)

        cond = [curr["HWHM"]>min_hwhm, curr["HWHM"]<min_hwhm]
        vals = [curr["HWHM"], min_hwhm]
        curr["HWHM"] = np.select(cond, vals)
        widths = pd.concat([widths, curr])

    return widths

def split_dataframe(df, n):
    components = np.array_split(df, n)
    components = {f"{num}":comp for num, comp in enumerate(components)}
    return components

def chunk_dataframe(df, chunk_size = 1000):
    for start in range(0,len(df), chunk_size):
        yield df[start:start+chunk_size]

def xsec_component(df, wavenumber_range, scale_style = "mult"):
    nu = df["nu"].to_numpy()[:,np.newaxis]
    I  = df["I"].to_numpy()[:,np.newaxis]
    sigma = df["Sigma"].to_numpy()[:,np.newaxis]
    
    dx = wavenumber_range[1]-wavenumber_range[0]

    gaussians = gaussian(wavenumber_range, nu, sigma, scale_factor = I, scale_style = scale_style, dx = dx)
                
    return np.sum(gaussians, axis = 0)

def get_xsec_single_process(df, wavenumber_range,chunk_size, write, fname, scale_style):
    xsec_chunk = 0
    total = int(len(df)/chunk_size)

    for chunk in tqdm(chunk_dataframe(df, chunk_size),total=total):
        xsec_chunk += xsec_component(chunk, wavenumber_range, scale_style = scale_style)
    
    if write == True:
        np.savetxt(f"{fname}",xsec_chunk)
    elif write == False:
        return xsec_chunk

def get_xsec_parallel_process(df, wavenumber_range, J_lower, Omega_lower,chunk_size = 1000, rm = False, min_hwhm=0, min_sigma = None, scale_style = "mult"):
    df = get_stdevs(df, J_lower, Omega_lower,min_hwhm = min_hwhm, min_sigma = min_sigma)
    
    running = sp.Popen("mkdir fds_tmp_dir", shell = True, stdout = sp.PIPE)
    running.communicate()
    
    cores = cpu_count()
    components = split_dataframe(df, cores)
    print("components made")
    processes = {}
    # if __name__=="__main__": # really should be here, but it doesn't work without it, if you want you can fix it

    for i in range(0,cores):
        fname = f"./fds_tmp_dir/xsec_{i}"
        processes[f"{i}"] = mp.Process(
            target = get_xsec_single_process,
            args = (components[f"{i}"], wavenumber_range, chunk_size, True, fname, scale_style)
        )
    for proc in processes.values():
        proc.start()
    for proc in processes.values():
        proc.join()
    
    xsec = 0
    for i in range(0,cores):
        xsec += np.loadtxt(f"./fds_tmp_dir/xsec_{i}")
    if rm == True:
        running = sp.Popen("rm -r fds_tmp_dir", shell=True, stdout = sp.PIPE)
        running.communicate()
    return xsec
    