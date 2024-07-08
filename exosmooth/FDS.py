import pandas as pd
import numpy as np

from sklearn.linear_model import HuberRegressor
from sklearn.preprocessing import StandardScaler

from binslt.distributions import gaussian
from tqdm import tqdm 

from multiprocessing import cpu_count
import multiprocessing as mp
import subprocess as sp

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
        sigma = gamma / (2 * np.sqrt(2 * np.log(2)))
    elif sigma != None:
        pass
    elif model != None:
        gamma = predict_HuberRegressor(model, wavenumber)
        sigma = gamma / (2 * np.sqrt(2 * np.log(2)))

    line_profile = gaussian(wavenumber_range, wavenumber, sigma)    
    
    line_profile *= intensity

    return line_profile

def get_stdevs(df, J_lower, Omega_lower):
    df_min = df[
        (df["J_lower"]==J_lower)&
        (df["Omega_lower"]==Omega_lower)
        ]
    
    grouped = df_min.groupby(["v_lower","v_upper"], as_index = False).agg(nu = ("nu","mean"))
    models = {}
    widths = pd.DataFrame(columns = [*df.columns]+["Sigma"])
    
    for v in grouped.v_lower.unique():
        curr = grouped[grouped.v_lower == v].sort_values("v_upper")
        nu = curr.nu.to_numpy()

        model = fit_HuberRegressor(nu)
        
        models[f"{v}"] = model
        curr = df[df["v_lower"]==v]

        curr["Sigma"] = predict_HuberRegressor(model, curr["nu"].to_numpy())/(2*np.sqrt(2*np.log(2)))
        
        widths = pd.concat([widths, curr])

    return widths

def split_dataframe(df, n):
    components = np.array_split(df, n)
    components = {f"{num}":comp for num, comp in enumerate(components)}
    return components

def chunk_dataframe(df, chunk_size = 1000):
    for start in range(0,len(df), chunk_size):
        yield df[start:start+chunk_size]

def xsec_component(df, wavenumber_range):
    nu = df["nu"].to_numpy()[:,np.newaxis]
    I  = df["I"].to_numpy()[:,np.newaxis]
    sigma = df["Sigma"].to_numpy()[:,np.newaxis]
    
    gaussians = gaussian(wavenumber_range, nu, sigma, I)

    return np.sum(gaussians, axis = 0)

def get_xsec_single_process(df, wavenumber_range,chunk_size, write, fname):
    xsec_chunk = 0
    total = int(len(df)/chunk_size)

    for chunk in tqdm(chunk_dataframe(df, chunk_size),total=total):
        xsec_chunk += xsec_component(chunk, wavenumber_range)
    
    if write == True:
        np.savetxt(f"{fname}",xsec_chunk)
    elif write == False:
        return xsec_chunk

def get_xsec_parallel_process(df, wavenumber_range, chunk_size = 1000):
    df = get_stdevs(df, 1.5, -1.5)
    
    running = sp.Popen("mkdir fds_tmp_dir", shell = True)
    running.communicate()
    
    cores = cpu_count()
    components = split_dataframe(df, cores)
    
    processes = {}
    if __name__=="__main__":
        for i in range(0,cores):
            fname = f"./fds_tmp_dir/xsec_{i}"
            processes[f"{i}"] = mp.Process(
                target = get_xsec_single_process,
                args = (components[f"{i}"], wavenumber_range, chunk_size, True, fname)
            )
        for proc in processes.values():
            proc.start()
        for proc in processes.values():
            proc.join()
    
    xsec = 0
    for i in range(0,cores):
        xsec += np.loadtxt(f"./fds_tmp_dir/xsec_{i}")

    running = sp.Popen("rm -r fds_tmp_dir", shell=True)
    running.communicate()
    return xsec
    