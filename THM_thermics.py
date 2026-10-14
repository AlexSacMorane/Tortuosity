import os
import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt
from time import perf_counter
import shutil
from skimage import io, util, exposure, transform, color
import skimage as ski
from PIL import Image

from solversLastversion import (
    local_operator,
    adaptive_eyre_milton_IC,
    adaptive_eyre_milton_H_IC,
    adaptive_eyre_milton_T_IC,
)


Microstructures = ["Imperial"]

images = [5]

plt.rcParams["figure.figsize"] = (6, 6)


t1 = perf_counter()


#Input parameters
c = np.inf
params = {
    "E": [0.0, 1.0, 1.0],
    "μ": [None, None, None],
    "ν": [0.0, 0.3, 0.3],
    "k": [1.0, 0.0, 0.0],
    "lmbda": [0.0, 1.0, 0.0],
    "visin_fluid": 0.001,
    "flu_denkgm3": 1000,
    "e": 0.002,
}

def convert_to_tiff(field):
    if len(field.shape) == 3:
        return np.asarray(np.swapaxes(field, 0, 2), dtype=np.float32)
    else:
        return np.asarray(field, dtype=np.float32)


def first_lame_coefficient(μ, ν):
    return 2.0 * μ * ν / (1.0 - 2.0 * ν)


def bulk_modulus(μ, ν, dim):
    λ = first_lame_coefficient(μ, ν)
    return λ + 2.0 * μ / dim


def shear_modulus(E, ν):
    return E / (2 * (1 + ν))

for i in range(3):
    params["μ"][i] = shear_modulus(params["E"][i], params["ν"][i])

average = "Voigt"
params["average"] = average
params["μ"] = np.array(params["μ"])
params["ν"] = np.array(params["ν"])
####### MECHANICS ##########

solver = adaptive_eyre_milton_T_IC
model = solver.__name__


load_cases = [
    ([1.0, 0.0, 0.0], "xx"),
    ([0.0, 1.0, 0.0], "yy"),
    ([0.0, 0.0, 1.0], "zz"),
]

for micro in Microstructures:

    for im in images:

        sys.modules['numpy._core.numeric'] = np.core.numeric

        if im < 10:
            with open(f"{micro}/dict_fft_0{im}", 'rb') as handle:
                dict_fft = pickle.load(handle)
        else:   
            with open(f"{micro}/dict_fft_{im}", 'rb') as handle:
                dict_fft = pickle.load(handle)

        del sys.modules['numpy._core.numeric']

        M_solid = dict_fft['M_microstructure']

        params["φ"] = M_solid
        params["κ"] = bulk_modulus(params["μ"], params["ν"], dim=params["φ"].ndim)

        for t_macro_, direction in load_cases:
            print(f"Running simulation for Microstructure = {micro}, sample = {im}, direction = {direction}")
            params["t_macro"] = t_macro_

            prefix = f"SIMULATIONS/{micro}/dict_fft_{im}/the/{direction}/"
            
            os.makedirs(prefix, exist_ok=True)
            if params["t_macro"] != 0.0:
                out_AEM, flux, grid_q = solver(**params)

                np.savetxt(prefix + "AEM.csv", out_AEM, delimiter=",")
                np.savetxt(prefix + "flux.csv", flux, delimiter=",")
                
                # change here if you want an image
                for i in range(len(params["t_macro"])):
                     image = convert_to_tiff(grid_q[..., i])
                     io.imsave(prefix + f"flux{i}.tiff", image)

            t2 = perf_counter()
            print(f"Total computation time (s): {t2 - t1:.4g}")
            np.savetxt(prefix + "time.csv", [t2 - t1], delimiter=",", fmt="%.4g")
