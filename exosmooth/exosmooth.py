import pandas as pd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from pandarallel import pandarallel, core

def read_xsec(path, as_dataframe = True):
    if as_dataframe == True:
        XSEC = pd.read_csv(path,sep = "\s+",names=["lambda","sigma"])
        return XSEC
    else:
        wavenumber, xsec = np.loadtxt(path, unpack = True)
        return wavenumber, xsec
    
def smooth_xsec(wavenumber, xsec, alpha=1000):
        
    dl = wavenumber[1] - wavenumber[0] #Finds distance between lambda data points
    sigma = alpha / np.sqrt(2*np.log(2))
    effective_sigma=sigma/dl  
    
    smoothed = gaussian_filter1d(xsec, effective_sigma) #Performs gaussian smoothing
    
    return smoothed #Returns Filtered curve cross section values

def differentiate(x, fx, sampling_frequency = 1):
    dx = x[sampling_frequency] - x[0] 
    f_prime = []
    
    for i in range(len(fx)-sampling_frequency):
        f_prime.append((fx[i+sampling_frequency]-fx[i])/dx)
        
    x = x[:(len(f_prime)-len(x))]
    return np.array([*x]), np.array([*f_prime])

def count_roots(x, function, positions = False, thresh_func = 0, thresh = 0):
    Counter = 0
    
    if thresh == 0:
        if positions == False:
            for i in range(len(function)-1):
                if np.sign(function[i]) != np.sign(function[i+1]):
                    Counter += 1
            return Counter
        else:
            Positions = []
            for i in range(len(function)-1):
                if np.sign(function[i]) != np.sign(function[i+1]):
                    Counter += 1
                    Positions.append(x[i]) 
            return Counter, Positions
        
    else:
        if positions == False:
            for i in range(len(function)-1):
                if np.sign(function[i]) != np.sign(function[i+1]) and thresh_func[i] > thresh:
                    Counter += 1
            return Counter
        else:
            Positions = []
            for i in range(len(function)-1):
                if np.sign(function[i]) != np.sign(function[i+1]) and  thresh_func[i] > thresh:
                    Counter += 1
                    Positions.append(x[i]) 
            return Counter, Positions

def smooth_xsec_count_roots(wavenumber, xsec, alpha,**kwargs):
    sampling_frequency_d1 = kwargs.get("sampling_frequency_d1", 0.01 )
    sampling_frequency_d2 = kwargs.get("sampling_frequency_d2", 0.01 )

    sampling_frequency_d1 = int(round(len(wavenumber)*sampling_frequency_d1,0)) 
    sampling_frequency_d2 = int(round(len(wavenumber)*sampling_frequency_d2,0))

    threshold             = kwargs.get("threshold"            , 0    )
    positions             = kwargs.get("positions"            , False)

    smoothed = smooth_xsec(wavenumber, xsec, alpha)

    x_1, f_prime_1 = differentiate(wavenumber, smoothed, sampling_frequency_d1)
    x_2, f_prime_2 = differentiate(x_1, f_prime_1, sampling_frequency_d2)

    num_roots_d1 = count_roots(x_1,f_prime_1, positions = positions, thresh_func=smoothed, thresh = threshold)
    num_roots_d2 = count_roots(x_2,f_prime_2, positions = positions, thresh_func=smoothed, thresh = threshold)
    return num_roots_d1, num_roots_d2

def optimize_alpha(alpha_range, wavenumber, xsec,style = "iterate",num_turning_points=1,progress_bar = True, **kwargs):
    if "i" in style.lower():
        data = []
        print("optimize alpha will iterate until optimal value found or range elapses")
        for alpha in alpha_range:
            print(f"Curent Alpha = {alpha}")
            
            num_roots_d1, num_roots_d2 = smooth_xsec_count_roots(wavenumber, xsec, alpha,**kwargs)
            data.append([alpha, num_roots_d1, num_roots_d2])
            
            if num_roots_d1 == num_turning_points and num_roots_d2 == num_turning_points+1:
                break
        alphas = pd.DataFrame(data, columns = ["alpha","num_roots_d1","num_roots_d2"])
        return alphas

    elif "f" in style.lower():
        print("optimize alpha will smooth full range")
        cores = kwargs.get("cores", core.NB_PHYSICAL_CORES)

        pandarallel.initialize(nb_workers=cores, progress_bar = progress_bar, verbose = 0)
        
        alphas = pd.DataFrame(data = [*alpha_range], columns = ["alpha"])
        alphas[["num_roots_d1","num_roots_d2"]] = alphas.parallel_apply(
            lambda x: smooth_xsec_count_roots(wavenumber, xsec, x["alpha"], **kwargs),
            axis = 1,
            result_type = "expand")
        return alphas
# %%