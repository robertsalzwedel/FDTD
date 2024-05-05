import numpy as np
import numba
from numba.experimental import jitclass
from parameters import *


# the @numba.jitclass decorator allows us to pass our custom
# class into a `numba.jit`ed function
@jitclass(
    [
        ("x", numba.float32[:, :, :]),
        ("y", numba.float32[:, :, :]),
        ("z", numba.float32[:, :, :]),
    ]
)
class Field(object):
    """This class creates a field in three directions.

    Attributes:
        x: Field strength in the x-direction
        y: Field strength in the y-direction
        z: Field strength in the z-direction
    """

    def __init__(self, dimensions, initial_value):
        """Construct a Field object with an
            *x*, *y*, and *z* component

        Size of array generated is x_cells by y_cells by z_cells
        with given initial_value
        """

        self.x = (
            np.ones((dimensions.x, dimensions.y, dimensions.z), dtype=np.float32)
            * initial_value
        )
        self.y = (
            np.ones((dimensions.x, dimensions.y, dimensions.z), dtype=np.float32)
            * initial_value
        )
        self.z = (
            np.ones((dimensions.x, dimensions.y, dimensions.z), dtype=np.float32)
            * initial_value
        )

    # this would have be thought through in more detail
    # def __add__(self, other):
    #     return self.x + other.x, self.y + other.y, self.z + other.z

    # def __str__(self):
    #     return str(self.x)


@jitclass(
    [
        ("x", numba.complex64[:, :, :, :]),
        ("y", numba.complex64[:, :, :, :]),
        ("z", numba.complex64[:, :, :, :]),
    ]
)
class DFT_Field_3D(object):
    """This class creates a field in three directions.

    Attributes:
        x: Field strength in the x-direction
        y: Field strength in the y-direction
        z: Field strength in the z-direction
    """

    def __init__(self, dimensions, initial_value, iwdim):
        """Construct a Field object with an
            *x*, *y*, and *z* component

        Size of array generated is x_cells by y_cells by z_cells
        with given initial_value
        """

        self.x = (
            np.ones(
                (iwdim + 1, dimensions.x, dimensions.y, dimensions.z),
                dtype=np.complex64,
            )
            * initial_value
        )
        self.y = (
            np.ones(
                (iwdim + 1, dimensions.x, dimensions.y, dimensions.z),
                dtype=np.complex64,
            )
            * initial_value
        )
        self.z = (
            np.ones(
                (iwdim + 1, dimensions.x, dimensions.y, dimensions.z),
                dtype=np.complex64,
            )
            * initial_value
        )


@jitclass(
    [
        ("x", numba.complex64[:, :, :]),
        ("y", numba.complex64[:, :, :]),
        ("z", numba.complex64[:, :, :]),
    ]
)
class DFT_Field_2D(object):
    """This class creates a field in three directions.

    Attributes:
        x: Field strength in the x-direction
        y: Field strength in the y-direction
        z: Field strength in the z-direction
    """

    def __init__(self, dim1, dim2, initial_value, iwdim):
        """Construct a Field object with an
            *x*, *y*, and *z* component

        Size of array generated is x_cells by y_cells by z_cells
        with given initial_value
        """

        self.x = np.ones((iwdim + 1, dim1, dim2), dtype=np.complex64) * initial_value
        self.y = np.ones((iwdim + 1, dim1, dim2), dtype=np.complex64) * initial_value
        self.z = np.ones((iwdim + 1, dim1, dim2), dtype=np.complex64) * initial_value

    def magnitude(self):
        return (np.abs(self.x) ** 2 + np.abs(self.y) ** 2 + np.abs(self.z) ** 2) ** (
            1 / 2
        )

    def surface_magnitude(self):
        data = self.magnitude()
        return np.sum(np.sum(data, axis=2), axis=1)


class Pulse:
    def __init__(self, width, delay, energy, dt, ddx, eps_in):
        self.width = width
        self.spread = self.width * 1e-15 / dt
        self.t0 = self.spread * delay
        self.energy = energy  # energy in eV
        self.freq_0 = 2 * np.pi * (energy * eC / h_planck)
        self.lam_0 = 2 * np.pi * c / self.freq_0
        self.ppw = self.lam_0 / ddx
        self.ppw_0 = int(self.ppw)
        self.ppwn = int(self.lam_0 / ddx / eps_in**0.5)
        self.amplitude = 1

    def print_parameters(self):
        print("pulse width = ", int(self.spread), " timesteps")
        print("delay t0 =", int(self.t0), " timesteps")
        print("lam0 =", int(self.lam_0 / nm), "nm")
        print("points per wavelength in media ", self.ppwn, "should be > 15")

    def update_value(self, t, dt):
        return (
            self.amplitude
            * np.exp(-0.5 * (t - self.t0) ** 2 / self.spread**2)
            * (np.cos(t * self.freq_0 * dt))
        )


class DFT:
    def __init__(self, dt, iwdim, pulse_spread, emin, emax):
        self.iwdim = iwdim
        self.tstart = int(1.8 * pulse_spread)
        self.omega = np.zeros([iwdim + 1], float)
        self.lam = np.zeros([iwdim + 1], float)
        self.energy = np.zeros([iwdim + 1], float)
        self.nu = np.zeros([iwdim + 1], float)
        self.emin = emin
        self.emax = emax
        self.energy = np.linspace(e_min, e_max, iwdim + 1)
        for ifreq in range(iwdim + 1):
            self.omega[ifreq] = (
                dt * eC / hbar * (e_min + ifreq * (e_max - e_min) / iwdim)
            )
            self.nu[ifreq] = (
                eC * (e_min + ifreq * (e_max - e_min) / iwdim) / h_planck / tera
            )
            self.lam[ifreq] = c / self.nu[ifreq] / tera

    def print_parameters(self):
        print("tDFTstart = ", self.tstart)
