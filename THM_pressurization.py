# This file is not working as the greyscale is not implemented for pressurization test


import os, sys, pickle
import numpy as np

import matplotlib.pyplot as plt
from time import perf_counter
import shutil
from skimage import io
from PIL import Image

from solvers_Biot import (
    local_operator,
    adaptive_eyre_milton,
    adaptive_eyre_milton_2,
)

Microstructures = ["sintered_sample_inv"]

images = range(0, 5)

plt.rcParams["figure.figsize"] = (6, 6)

def convert_to_tiff(field):
    if len(field.shape) == 3:
        return np.asarray(np.swapaxes(field, 0, 2), dtype=np.float32)
    else:
        return np.asarray(field, dtype=np.float32)

def minimg(x):
    x = np.asarray(x)
    out = np.copy(x)
    out[x > 0.5] -= 1
    out[x < -0.5] += 1
    return out

def first_lame_coefficient(μ, ν):
    return 2.0 * μ * ν / (1.0 - 2.0 * ν)


def bulk_modulus(μ, ν, dim):
    λ = first_lame_coefficient(μ, ν)
    return λ + 2.0 * μ / dim

def shear_modulus(E, ν):
    return E / (2 * (1 + ν))

def Biots_module(k_s, phi0, biot):
    return (biot - phi0)/k_s


def alpha_phi(alpha, phi0, biot):
    return alpha * (biot - phi0)

params = {
    "E": [0, 1, 1],
    "μ": [None, None, None],
    "ν": [0., 0.3, 0.3],
    "biot": [1.0, 0.0, 0.0],
    "e": 0.002,
    "alpha": [0, 0.05, 0.05],
    "N": [0.0, 0.0, 0.0],
    "alpha_phi": [0.0, 0.0, 0.0],
    "Heat_cap": [1.47e6, 0.0, 0],
}
for i in range(3):
    params["μ"][i] = shear_modulus(params["E"][i], params["ν"][i])
      
solver = adaptive_eyre_milton_2
model = solver.__name__

# load case
params["p_macro"] = [0.0]
params["T_macro"] = [1.0]
params["ε_macro"] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]


for micro in Microstructures:

    for im in images:
        t1 = perf_counter()

        sys.modules['numpy._core.numeric'] = np.core.numeric

        if im < 10:
            with open(f"{micro}/dict_fft_0{im}", 'rb') as handle:
                dict_fft = pickle.load(handle)
        else:   
            with open(f"{micro}/dict_fft_{im}", 'rb') as handle:
                dict_fft = pickle.load(handle)

        del sys.modules['numpy._core.numeric']

        M_solid = dict_fft['M_microstructure']

        # adapt
        params["φ"] = M_solid
        params["μ"] = np.array(params["μ"])
        params["ν"] = np.array(params["ν"])
        params["p_macro"] = np.array(params["p_macro"])
        params["T_macro"] = np.array(params["T_macro"])
        params["ε_macro"] = np.array(params["ε_macro"])
        params["κ"] = bulk_modulus(params["μ"], params["ν"], dim=params["φ"].ndim)

        # create the prefix for the save
        if params["p_macro"] != [0.0] and params["T_macro"] == [0.0]:
            prefix = f"SIMULATIONS/{micro}/dict_fft_{im}/pressP/"
        elif params["p_macro"] == [0.0] and params["T_macro"] != [0.0]:
            prefix = f"SIMULATIONS/{micro}/dict_fft_{im}/pressT/"
        elif params["p_macro"] != [0.0] and params["T_macro"] != [0.0]:
            prefix = f"SIMULATIONS/{micro}/dict_fft_{im}/pressPT/"
        else: 
            raise ValueError('Apply pressurization')
                
        
        os.makedirs(prefix, exist_ok=True)
        out_AEM, sigma, phi, grid_sig, grid_eps = solver(**params)

        np.savetxt(prefix + "AEM.csv", out_AEM, delimiter=",")
        np.savetxt(prefix + "sigma.csv", sigma, delimiter=",")
        np.savetxt(prefix + "phi-phi0.csv", phi, delimiter=",")
        # np.savetxt(prefix + "S_s.csv", S, delimiter=",")

        # import image
        #image = convert_to_tiff((grid_sig[..., 0]+grid_sig[..., 1]+grid_sig[..., 2])/3)
        #io.imsave(prefix + f"sig_p.tiff", image)
        #image = convert_to_tiff((grid_eps[..., 0]+grid_eps[..., 1]+grid_eps[..., 2]))
        #io.imsave(prefix + f"eps_v.tiff", image)

        t2 = perf_counter()

        print(f"{micro} | image {im} | Total computation time (s): {t2-t1:.4g}")
        