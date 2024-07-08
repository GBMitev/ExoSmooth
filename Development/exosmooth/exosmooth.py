# %%
from scipy.ndimage import gaussian_filter1d

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def read_xsec(path, data_type= "Pandas"):
    """
    Input:
    path: Location of file you want to input - type = .xsec or any tab_delim file
    Data: "Pandas" (default is Pandas), if anything else: will return two outputs, "lambda" and "sigma". - dtype = str
    
    Output: 
    XSEC: Data outputted in pandas DataFrame - dtype = DataFrame 
    Lambda, Sigma: if Data != "Pandas", returns "lambda" and "sigma" upacked - dtypes = np.array
    """
    
    if data_type == "Pandas":
        XSEC = pd.read_table(path,delim_whitespace=True,names=["lambda","sigma"])
        return XSEC
    else:
        Lambda, Sigma = np.loadtxt(path, unpack = True)
        return Lambda, Sigma
    
    
def apply_smoothing(lambda_vals, sigma_vals, alpha=1000):
    
    """
    Input: 
    Data_Lambda: Input cross section "lambda" (must have equal incrementations)- dtype = np.array
    Data_Sigma: Input cross section "sigma" - dtype = np.array
    alpha: Value of alpha filtering parameter, default = 1000 - dtype = float
    scalefactor: Linear multiplicative scale factor for resulting curve, default is 1 - dtype = float
    
    Output: 
    Gaussian filtered cross section 
    """
    
    dl = lambda_vals[1] - lambda_vals[0] #Finds distance between lambda data points
    
    stdev=alpha/dl  
    
    smoothed = gaussian_filter1d(sigma_vals, stdev) #Performs gaussian smoothing
    
    return smoothed #Returns Filtered curve cross section values

def differentiate(x, fx, sampling_frequency = 1):
    """
    Input: 
    x, fx: Input curve x and fx values - dtypes = np.array
    sampling frequency: number of data points between gradient pair - dtype = int
    
    Output:
    x: truncated x array accounting for lost terms - dtype = np.array
    f_prime: 1st differential f_prime(x) points - dtype = np.array
    """
    
    dx = x[sampling_frequency] - x[0] 
    f_prime = []
    
    for i in range(len(fx)-sampling_frequency):
        f_prime.append((fx[i+sampling_frequency]-fx[i])/dx)
        
    x = x[:(len(f_prime)-len(x))]
    return x, f_prime

def count_roots(x, function, positions = False, thresh_func = None, thresh = None):
    """
    Input: 
    x: x points - dtype = np.array
    function: (gradient function) f'(x) points - dtype = np.array
    positions: return location of roots in the x array (point before crossover), default = False - dtype = boolean
    ThreshFunc: Integral of "function", default = 0 - dtype = np.array
    Thresh: If ThreshFunc < Thresh, if root found, not counted, else, counted, default = 0 - dtype = float  
    
    Output:
    Counter: Number of roots in the function - dtype = int
    Positions: if positions != False Location of all roots - dtype = list 
    """
    Counter = 0

    Positions = []
    thresh_func = thresh_func if thresh_func is not None else np.ones(len(function))
    thresh      = thresh      if thresh_func is not None and thresh is not None else np.zeros(len(function))

    for i in range(len(function)-1):
        if np.sign(function[i]) != np.sign(function[1+1]) and :
            Counter +=1
            Positions.append(x[i])
        return Counter



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

def AlphaFinder(Alpha_Range, Lambda, Sigma, Num_Turn_Points=1, sampling_frequency = [0.01, 0.01], scalefactor_ = 18, Thresh_ = 0, positions = False):
    """
        Input: 
        Alpha Range: range of Alphas to test - dtype = np.array
        Lambda: lambda values for input signal - dtype = np.array
        Sigma: sigma values for input signal - dtype = np.array
        Num_Turn_Points: Number of turning points in expected cross section, found by visual inspection - dtype = Float
        sampling_frequency: sampling frequency as a percentage (of input signal length [between 0 - 1]) - dtype = list
        scalefactor_: Linear multiplicative scale factor for resulting curve, default is 1 - dtype = float
        Thresh_: Threshold value for function if root located in region where function < Thresh, root not counted. 
        
        Output:
        Optimum_Alpha: Alpha which satisfies test parameters
    """
    #Defining Test Parameters and Sampling Frequencies
    
    Test_Parameters = [Num_Turn_Points, Num_Turn_Points+1] # Turning point tests
    
    # Sampling frequency transformed from percentage to int
    Samp_1 = int(round(len(Lambda)*sampling_frequency[0],0)) 
    Samp_2 = int(round(len(Lambda)*sampling_frequency[1],0))
    
    for i in Alpha_Range:
        print(f"Curent Alpha = {i}")
        Filtered = apply_smoothing(Lambda, Sigma, i, scalefactor=scalefactor_)
        
        # differentiation
        x_1, f_prime_1 = differentiate(Lambda, Filtered, Samp_1)
        x_2, f_prime_2 = differentiate(x_1, f_prime_1, Samp_2)
        
        Roots_1 = count_roots(x_1,f_prime_1, positions = False, thresh_func=Filtered, thresh = Thresh_)
        Roots_2 = count_roots(x_2,f_prime_2, positions = False, thresh_func=Filtered, thresh = Thresh_)
        
        if Roots_1 == Test_Parameters[0] and Roots_2 == Test_Parameters[1]:
            break
            
    Optimum_Alpha = i   
    print(f"Root No First Der: {Roots_1}, Root No Second Der: {Roots_2}") 
    return Optimum_Alpha   
# %%
import numpy as np
np.ones(10)
