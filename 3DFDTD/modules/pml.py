from collections import namedtuple
import numpy as np

PerfectlyMatchedLayer = namedtuple(
    "PerfectlyMatchedLayer",
    (
        "fi1",
        "fi2",
        "fi3",
        "fj1",
        "fj2",
        "fj3",
        "fk1",
        "fk2",
        "fk3",
        "gi1",
        "gi2",
        "gi3",
        "gj1",
        "gj2",
        "gj3",
        "gk1",
        "gk2",
        "gk3",
    ),
)


def calculate_pml_params(dims, npml, BOUNDARY_FLAG):
    """Creates the Perfectly Matched Layer object"""
    fj1, gj2, gj3 = calculate_pml_slice(dims.y, 0, npml)
    gj1, fj2, fj3 = calculate_pml_slice(dims.y, 0.5, npml)

    if BOUNDARY_FLAG == "PML":
        fi1, gi2, gi3 = calculate_pml_slice(dims.x, 0, npml)
        fk1, gk2, gk3 = calculate_pml_slice(dims.z, 0, npml)
        gi1, fi2, fi3 = calculate_pml_slice(dims.x, 0.5, npml)
        gk1, fk2, fk3 = calculate_pml_slice(dims.z, 0.5, npml)
    elif BOUNDARY_FLAG == "PBC":
        fi1 = np.zeros(dims.x, float)
        fk1 = np.zeros(dims.z, float)
        gi1 = np.zeros(dims.x, float)
        gk1 = np.zeros(dims.z, float)
        fi2 = np.ones(dims.x, float)
        fk2 = np.ones(dims.z, float)
        gi2 = np.ones(dims.x, float)
        gk2 = np.ones(dims.z, float)
        fi3 = np.ones(dims.x, float)
        fk3 = np.ones(dims.z, float)
        gi3 = np.ones(dims.x, float)
        gk3 = np.ones(dims.z, float)

    else:
        raise ValueError("You did not implement any boundaries.")

    pml = PerfectlyMatchedLayer(
        fi1=fi1,
        fi2=fi2,
        fi3=fi3,
        fj1=fj1,
        fj2=fj2,
        fj3=fj3,
        fk1=fk1,
        fk2=fk2,
        fk3=fk3,
        gi1=gi1,
        gi2=gi2,
        gi3=gi3,
        gj1=gj1,
        gj2=gj2,
        gj3=gj3,
        gk1=gk1,
        gk2=gk2,
        gk3=gk3,
    )
    return pml


def calculate_pml_slice(size, offset, pml_cells):
    """This initializes arrays and calculates a slice of
    the PML parameters
    (three of the parameters along one direction that use
    the same offset).
    fx1, gx2, gx3: offset = 0
    gx1, fx2, fx3: offset = 0.5"""
    distance = np.arange(pml_cells, 0, -1)
    xxn = (distance - offset) / pml_cells
    xn = 0.33 * (xxn**3)
    p1 = np.zeros(size, float)
    p2 = np.ones(size, float)
    p3 = np.ones(size, float)
    p1[:pml_cells] = xn
    p1[size - pml_cells - int(2 * offset) : size - int(2 * offset)] = np.flip(xn, 0)
    p2[:pml_cells] = 1.0 / (1.0 + xn)
    p2[size - pml_cells - int(2 * offset) : size - int(2 * offset)] = 1.0 / (
        1.0 + np.flip(xn, 0)
    )
    p3[:pml_cells] = (1.0 - xn) / (1.0 + xn)
    p3[size - pml_cells - int(2 * offset) : size - int(2 * offset)] = (
        1.0 - np.flip(xn, 0)
    ) / (1.0 + np.flip(xn, 0))
    return p1, p2, p3
