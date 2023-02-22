# -*- coding: utf-8 -*-
"""
Created on Wed Feb 22 01:43:19AM 2023

@author: Mustafa Hammood
"""
import numpy as np
import matplotlib.pyplot as plt


class mzi:
    def __init__(self, wavl_start, wavl_stop, resolution, lam, n1, n2, n3, L1, L2, alpha, kappa):
        self.wavl_start = wavl_start  # spectrum start wavelength (m)
        self.wavl_stop = wavl_stop  # spectrum stop wavelength (m)
        self.resolution = resolution  # resolution step (nm)
        self.lam = lam  # effective index model fit's wavelength
        self.n1 = n1  # effective index model fit, 1st term
        self.n2 = n2  # effective index model fit, 2nd term
        self.n3 = n3  # effective index model fit, 3rd term
        self.L1 = L1  # MZI's 1st arm length (µm)
        self.L2 = L2  # MZI's 2nd arm length (µm)
        self.alpha = alpha  # propagation loss (/µm)
        self.kappa = kappa  # coupling coefficient (power, NOT field)

    @property
    def lam0(self):
        """Linear space of the wavelength range."""
        import numpy as np
        return np.linspace(self.wavl_start, self.wavl_stop, round(
            (self.wavl_stop-self.wavl_start)*1e9/self.resolution))*1e6

    @property
    def neff(self):
        """2nd order polynomial fit for a dispersive effective index model."""
        return self.n1 + self.n2*(self.lam0-self.lam) + self.n3*(self.lam0-self.lam)**2

    @property
    def beta(self):
        """Propagation constant (including the complex term)."""
        import numpy as np
        return 2*np.pi*self.neff/self.lam0 - 1j*self.alpha/2*np.ones(np.size(self.lam0))

    @property
    def extinction_ratio(self):
        """Maximum attainable extinction ratio of the device."""
        import numpy as np
        T = 10*np.log10([np.abs(i**2) for i in self.TMM_MZI()[0][0]])
        return np.abs(np.max(T)-np.min(T))

    def TMM_MZI(self):
        """Transfer matrix model of the MZI using two directional couplers."""
        import numpy as np
        alpha = np.sqrt(self.kappa/2)
        beta = np.sqrt((1-self.kappa)/2)
        M_dc = np.array([[1, self.kappa], [1-self.kappa, 1]])
        M_wg = np.array([[0, np.exp(-1j*self.beta*self.L1)],
                        [np.exp(-1j*self.beta*self.L2), 0]])
        M = np.dot(np.dot(M_dc, M_wg), M_dc)

        return np.dot(M, np.array([[1], [0]]))

    def plot_wavl(self):
        import matplotlib.pyplot as plt
        import numpy as np
        lam0 = np.linspace(self.wavl_start, self.wavl_stop, round(
            (self.wavl_stop-self.wavl_start)*1e9/self.resolution))*1e6
        fig, ax = plt.subplots()
        ax.plot(lam0, 10*np.log10([np.abs(i**2) for i in self.TMM_MZI()[0][0]]), color='b')
        ax.set_xlabel('Wavelength [µm]')
        ax.set_ylabel('Transmission [dB]')
        ax.set_title(f'Transmission spectrum of the MZI with kappa={self.kappa}')


# %% example of an MZI with 50:50 splitter
device = mzi(wavl_start=1.5e-6, wavl_stop=1.6e-6, resolution=0.001, lam=1.55,
             n1=2.4, n2=-1, n3=0, L1=100, L2=135, alpha=1e-3, kappa=.5)
device.plot_wavl()

# %% example of the MZI transfer function as a function of splitting ratio
kappa = np.linspace(0.3, 0.5, 5)

fig, ax = plt.subplots()
for k in kappa:
    device = mzi(wavl_start=1.53e-6, wavl_stop=1.57e-6, resolution=0.001, lam=1.55,
                 n1=2.4, n2=-1, n3=0, L1=100, L2=135, alpha=1e-3, kappa=k)
    ax.plot(device.lam0, 10*np.log10([np.abs(i**2)
            for i in device.TMM_MZI()[0][0]]), label=f"k={k}")
ax.set_xlabel('Wavelength [µm]')
ax.set_ylabel('Transmission [dB]')
ax.set_title('Transmission spectrum of the MZI')
fig.legend(bbox_to_anchor=(0.4, 0.5))

# %% example of the MZI transfer function as a function of effective index variance
dneff = np.linspace(-0.005, 0.005, 6)

fig, ax = plt.subplots()
for d in dneff:
    device = mzi(wavl_start=1.53e-6, wavl_stop=1.57e-6, resolution=0.001, lam=1.55,
                 n1=2.4+d, n2=-1, n3=0, L1=100, L2=135, alpha=1e-3, kappa=0.5)
    ax.plot(device.lam0, 10*np.log10([np.abs(i**2)
            for i in device.TMM_MZI()[0][0]]), label=f"Δn_eff={d}")
ax.set_xlabel('Wavelength [µm]')
ax.set_ylabel('Transmission [dB]')
ax.set_title('Transmission spectrum of the MZI')
fig.legend(bbox_to_anchor=(0.4, 0.5))

# %% example of the MZI transfer function as a function of waveguide loss
loss = np.linspace(0, 5e-3, 6)

fig, ax = plt.subplots()
for l in loss:
    device = mzi(wavl_start=1.53e-6, wavl_stop=1.57e-6, resolution=0.001, lam=1.55,
                 n1=2.4, n2=-1, n3=0, L1=100, L2=135, alpha=l, kappa=0.5)
    ax.plot(device.lam0, 10*np.log10([np.abs(i**2)
            for i in device.TMM_MZI()[0][0]]), label=f"Loss={l} /µm")
ax.set_xlabel('Wavelength [µm]')
ax.set_ylabel('Transmission [dB]')
ax.set_title('Transmission spectrum of the MZI')
fig.legend(bbox_to_anchor=(0.4, 0.5))
