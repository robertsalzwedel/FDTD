from pyexpat import EXPAT_VERSION
import numpy as np
import numba
from scipy.constants import nano as nm

# import warnings
# import solve
# import grid
# import constants_jonas as cj

# warnings.filterwarnings('ignore')

# @numba.jit(nopython=True)
# def update_micro(ddx,micro_time,offset,diameter,j,f_global,e,sphere, *args):
#     #*args = E_FIELD_FLAG, ep_in_a, ep_in_e, ep_out_a, ep_out_e, kp, lm_k, lm_p, lm_t, e_k, e_p, e_t, fd_p, fd_t
#     t_max = micro_time*1e15

#     print('Micro code working')
#     t_steps = 2
#     for i in range (diameter):
#         for l in range (diameter):
#             for k in range (diameter):
#                 ii = i + offset
#                 ll = l + offset
#                 kk = k + offset
#                 dist = np.sqrt((ii*ddx-sphere.x)**2+(ll*ddx-sphere.y)**2+(kk*ddx-sphere.z)**2)
#                 if (dist <= sphere.R): # hard boundary condition without substaggering
#                     E_field = 1e11*cj.unitchange_E * np.array([e.x[ii,ll,kk],e.y[ii,ll,kk],e.z[ii,ll,kk]])   #E field from modified SI to semicondutcor units
#                     #print('E',E_field,e.x[ii,ll,kk])
#                     f = f_global[i,l,k,:,:,:]

#                     # current = np.full((3,t_steps),0.) #'this should be defined outside the loop
#                     f = np.reshape(f.copy(), grid.n_kmax*grid.n_phimax*grid.n_thetamax)   #Initial value has to be 1D

#                     #RK2
#                     current_density, wignermatrix = solve.solve_boltzmann_equation_with_rk2(t_steps, t_max, f, E_field, *args)
#                     #print('curr dens', current_density)
#                     #RK4
#                     #current_density, wignermatrix = solve.solve_boltzmann_equation_with_rk4(t_steps, t_max, f, E_field, *args)

#                     j.x[ii,ll,kk] = float(1/cj.unitchange_j * current_density[0,t_steps-1])  #Current from semiconductor units to modified SI
#                     j.y[ii,ll,kk] = float(1/cj.unitchange_j * current_density[1,t_steps-1])
#                     j.z[ii,ll,kk] = float(1/cj.unitchange_j * current_density[2,t_steps-1])
#                     f_global[i,l,k,:,:,:] = wignermatrix[t_steps-1,:,:,:]

#                     #print('j in micro', j.x[ii,ll,kk] ,j.y[ii,ll,kk] ,j.z[ii,ll,kk],current_density[0,t_steps-1],current_density[1,t_steps-1],current_density[2,t_steps-1] )

#     print('Current:', j.x[offset+int(diameter/2),offset+int(diameter/2),offset+int(diameter/2)], j.y[offset+int(diameter/2),offset+int(diameter/2),offset+int(diameter/2)], j.z[offset+int(diameter/2),offset+int(diameter/2),offset+int(diameter/2)])

#     return j, f_global

# @numba.jit(nopython=True)
# def update_polarization_micro(offset,diameter,dt,p,j,j_tmp):
#     for i in range (offset,offset+diameter):
#         for m in range (offset,offset+diameter):
#             for n in range (offset,offset+diameter):
#                 p.x[i,m,n] += 0.5*dt*(j.x[i,m,n]+j_tmp.x[i,m,n])
#                 p.y[i,m,n] += 0.5*dt*(j.y[i,m,n]+j_tmp.y[i,m,n])
#                 p.z[i,m,n] += 0.5*dt*(j.z[i,m,n]+j_tmp.z[i,m,n])
#     return p


@numba.jit(nopython=True)
def calculate_polarization(dims, sphere, ddx, p, p_tmp, e, f1, f2, f3, OBJECT_FLAG):
    # p1 = p_tmp
    # p_tmp = p
    if OBJECT_FLAG == 0:
        return p, p_tmp
    elif OBJECT_FLAG == 1:
        for i in range(
            int((sphere.x - sphere.R) / ddx) - 2, int((sphere.x + sphere.R) / ddx) + 2
        ):
            for j in range(
                int((sphere.y - sphere.R) / ddx) - 2,
                int((sphere.y + sphere.R) / ddx) + 2,
            ):
                for k in range(
                    int((sphere.z - sphere.R) / ddx) - 2,
                    int((sphere.z + sphere.R) / ddx) + 2,
                ):
                    px, py, pz, ptmpx, ptmpy, ptmpz = update_polarization_point(
                        i, j, k, p, p_tmp, e, f1, f2, f3
                    )
                    p.x[i, j, k] = px
                    p.y[i, j, k] = py
                    p.z[i, j, k] = pz
                    p_tmp.x[i, j, k] = ptmpx
                    p_tmp.y[i, j, k] = ptmpy
                    p_tmp.z[i, j, k] = ptmpz
        return p, p_tmp

    elif OBJECT_FLAG == 2:
        for i in range(0, dims.x):
            for j in range(0, dims.y):  # this can be optimized
                for k in range(0, dims.z):
                    px, py, pz, ptmpx, ptmpy, ptmpz = update_polarization_point(
                        i, j, k, p, p_tmp, e, f1, f2, f3
                    )
                    p.x[i, j, k] = px
                    p.y[i, j, k] = py
                    p.z[i, j, k] = pz
                    p_tmp.x[i, j, k] = ptmpx
                    p_tmp.y[i, j, k] = ptmpy
                    p_tmp.z[i, j, k] = ptmpz
        return p, p_tmp


@numba.jit(nopython=True)
def calculate_polarization_etch(
    dims, sphere, ddx, p, p_tmp, e, e1, f1, f2, f3, f4, OBJECT_FLAG
):
    # p1 = p_tmp
    # p_tmp = p
    if OBJECT_FLAG == 0:
        return p, p_tmp
    elif OBJECT_FLAG == 1:
        for i in range(
            int((sphere.x - sphere.R) / ddx) - 2, int((sphere.x + sphere.R) / ddx) + 2
        ):
            for j in range(
                int((sphere.y - sphere.R) / ddx) - 2,
                int((sphere.y + sphere.R) / ddx) + 2,
            ):
                for k in range(
                    int((sphere.z - sphere.R) / ddx) - 2,
                    int((sphere.z + sphere.R) / ddx) + 2,
                ):
                    (
                        p.x[i, j, k],
                        p.y[i, j, k],
                        p.z[i, j, k],
                        p_tmp.x[i, j, k],
                        p_tmp.y[i, j, k],
                        p_tmp.z[i, j, k],
                    ) = update_polarization_four(
                        i, j, k, p, p_tmp, e, e1, f1, f2, f3, f4
                    )
        return p, p_tmp

    elif OBJECT_FLAG == 2:
        for i in range(0, dims.x):
            for j in range(0, dims.y):  # this can be optimized
                for k in range(0, dims.z):
                    (
                        p.x[i, j, k],
                        p.y[i, j, k],
                        p.z[i, j, k],
                        p_tmp.x[i, j, k],
                        p_tmp.y[i, j, k],
                        p_tmp.z[i, j, k],
                    ) = update_polarization_four(
                        i, j, k, p, p_tmp, e, e1, f1, f2, f3, f4
                    )
        return p, p_tmp


@numba.jit(nopython=True)
def update_polarization_four(i, j, k, p, p_tmp, e, e1, f1, f2, f3, f4):
    p1x = p_tmp.x[i, j, k]
    p1y = p_tmp.y[i, j, k]
    p1z = p_tmp.z[i, j, k]
    ptmpx = p.x[i, j, k]
    ptmpy = p.y[i, j, k]
    ptmpz = p.z[i, j, k]

    # update polarization
    px = (
        f1.x[i, j, k] * p.x[i, j, k]
        + f2.x[i, j, k] * p1x
        + f3.x[i, j, k] * e.x[i, j, k]
        + f4.x[i, j, k] * e1.x[i, j, k]
    )
    py = (
        f1.y[i, j, k] * p.y[i, j, k]
        + f2.y[i, j, k] * p1y
        + f3.y[i, j, k] * e.y[i, j, k]
        + f4.y[i, j, k] * e1.y[i, j, k]
    )
    pz = (
        f1.z[i, j, k] * p.z[i, j, k]
        + f2.z[i, j, k] * p1z
        + f3.z[i, j, k] * e.z[i, j, k]
        + f4.z[i, j, k] * e1.z[i, j, k]
    )
    return px, py, pz, ptmpx, ptmpy, ptmpz


@numba.jit(nopython=True)
def update_polarization_point(i, j, k, p, p_tmp, e, f1, f2, f3):
    p1x = p_tmp.x[i, j, k]
    p1y = p_tmp.y[i, j, k]
    p1z = p_tmp.z[i, j, k]
    ptmpx = p.x[i, j, k]
    ptmpy = p.y[i, j, k]
    ptmpz = p.z[i, j, k]

    # update polarization
    px = (
        f1.x[i, j, k] * p.x[i, j, k]
        + f2.x[i, j, k] * p1x
        + f3.x[i, j, k] * e.x[i, j, k]
    )
    py = (
        f1.y[i, j, k] * p.y[i, j, k]
        + f2.y[i, j, k] * p1y
        + f3.y[i, j, k] * e.y[i, j, k]
    )
    pz = (
        f1.z[i, j, k] * p.z[i, j, k]
        + f2.z[i, j, k] * p1z
        + f3.z[i, j, k] * e.z[i, j, k]
    )
    return px, py, pz, ptmpx, ptmpy, ptmpz


@numba.jit(nopython=True)
def create_rectangle_PBC(dims, y_low, y_high, dt, eps_in, wp, gamma, ga, d1, d2, d3):
    for jj in range(y_low, y_high):
        for ii in range(0, dims.x):
            for kk in range(0, dims.z):
                # x components
                ga.x[ii, jj, kk] = 1 / eps_in
                d1.x[ii, jj, kk] = 4.0 / (2.0 + gamma * dt)
                d2.x[ii, jj, kk] = (gamma * dt - 2.0) / (gamma * dt + 2.0)
                d3.x[ii, jj, kk] = 2.0 * wp**2 * dt**2 / (2.0 + gamma * dt)

                # y components
                ga.y[ii, jj, kk] = 1 / eps_in
                d1.y[ii, jj, kk] = 4.0 / (2.0 + gamma * dt)
                d2.y[ii, jj, kk] = (gamma * dt - 2.0) / (gamma * dt + 2.0)
                d3.y[ii, jj, kk] = 2.0 * wp**2 * dt**2 / (2.0 + gamma * dt)

                # z components
                ga.z[ii, jj, kk] = 1 / eps_in
                d1.z[ii, jj, kk] = 4.0 / (2.0 + gamma * dt)
                d2.z[ii, jj, kk] = (gamma * dt - 2.0) / (gamma * dt + 2.0)
                d3.z[ii, jj, kk] = 2.0 * wp**2 * dt**2 / (2.0 + gamma * dt)
    return ga, d1, d2, d3


@numba.jit(nopython=True)
def create_rectangle_PBC_lorentz(
    dims, y_low, y_high, dt, eps_in, wl, gamma_l, delta_eps, ga, l1, l2, l3
):
    for jj in range(y_low, y_high):
        for ii in range(0, dims.x):
            for kk in range(0, dims.z):
                # x components
                ga.x[ii, jj, kk] = 1 / eps_in
                l1.x[ii, jj, kk] = (2.0 - wl**2 * dt**2) / (1.0 + gamma_l * dt / 2.0)
                l2.x[ii, jj, kk] = (gamma_l * dt / 2.0 - 1.0) / (
                    1.0 + gamma_l * dt / 2.0
                )
                l3.x[ii, jj, kk] = (
                    delta_eps * wl**2 * dt**2 / (1.0 + gamma_l * dt / 2.0)
                )

                # y components
                ga.y[ii, jj, kk] = 1 / eps_in
                l1.y[ii, jj, kk] = (2.0 - wl**2 * dt**2) / (1.0 + gamma_l * dt / 2.0)
                l2.y[ii, jj, kk] = (gamma_l * dt / 2.0 - 1.0) / (
                    1.0 + gamma_l * dt / 2.0
                )
                l3.y[ii, jj, kk] = (
                    delta_eps * wl**2 * dt**2 / (1.0 + gamma_l * dt / 2.0)
                )

                # z components
                ga.z[ii, jj, kk] = 1 / eps_in
                l1.z[ii, jj, kk] = (2.0 - wl**2 * dt**2) / (1.0 + gamma_l * dt / 2.0)
                l2.z[ii, jj, kk] = (gamma_l * dt / 2.0 - 1.0) / (
                    1.0 + gamma_l * dt / 2.0
                )
                l3.z[ii, jj, kk] = (
                    delta_eps * wl**2 * dt**2 / (1.0 + gamma_l * dt / 2.0)
                )
    return ga, l1, l2, l3


@numba.jit(nopython=True)
def create_rectangle_PBC_etch(
    dims,
    y_low,
    y_high,
    dt,
    c1,
    c2,
    w1,
    gamma1,
    w2,
    gamma2,
    f1_et1,
    f2_et1,
    f3_et1,
    f4_et1,
    f1_et2,
    f2_et2,
    f3_et2,
    f4_et2,
):
    for jj in range(y_low, y_high):
        for ii in range(0, dims.x):
            for kk in range(0, dims.z):
                # x components
                f1_et1.x[ii, jj, kk] = (2.0 - w1**2 * dt**2) / (1.0 + gamma1 * dt)
                f2_et1.x[ii, jj, kk] = (gamma1 * dt - 1.0) / (1.0 + gamma1 * dt)
                f3_et1.x[ii, jj, kk] = (
                    c1 * ((w1 + gamma1) * dt**2 + 2 * dt) / (2 * (1.0 + gamma1 * dt))
                )
                f4_et1.x[ii, jj, kk] = (
                    c1 * ((w1 + gamma1) * dt**2 - 2 * dt) / (2 * (1.0 + gamma1 * dt))
                )

                f1_et2.x[ii, jj, kk] = (2.0 - w2**2 * dt**2) / (1.0 + gamma2 * dt)
                f2_et2.x[ii, jj, kk] = (gamma2 * dt - 1) / (1 + gamma2 * dt)
                f3_et2.x[ii, jj, kk] = (
                    c2 * ((w2 + gamma2) * dt**2 + 2 * dt) / (2 * (1 + gamma2 * dt))
                )
                f4_et2.x[ii, jj, kk] = (
                    c2 * ((w2 + gamma2) * dt**2 - 2 * dt) / (2 * (1 + gamma2 * dt))
                )

                # y components
                f1_et1.y[ii, jj, kk] = (2.0 - w1**2 * dt**2) / (1.0 + gamma1 * dt)
                f2_et1.y[ii, jj, kk] = (gamma1 * dt - 1) / (1 + gamma1 * dt)
                f3_et1.y[ii, jj, kk] = (
                    c1 * ((w1 + gamma1) * dt**2 + 2 * dt) / (2 * (1 + gamma1 * dt))
                )
                f4_et1.y[ii, jj, kk] = (
                    c1 * ((w1 + gamma1) * dt**2 - 2 * dt) / (2 * (1 + gamma1 * dt))
                )

                f1_et2.y[ii, jj, kk] = (2.0 - w2**2 * dt**2) / (1.0 + gamma2 * dt)
                f2_et2.y[ii, jj, kk] = (gamma2 * dt - 1) / (1 + gamma2 * dt)
                f3_et2.y[ii, jj, kk] = (
                    c2 * ((w2 + gamma2) * dt**2 + 2 * dt) / (2 * (1 + gamma2 * dt))
                )
                f4_et2.y[ii, jj, kk] = (
                    c2 * ((w2 + gamma2) * dt**2 - 2 * dt) / (2 * (1 + gamma2 * dt))
                )

                # z components
                f1_et1.z[ii, jj, kk] = (2.0 - w1**2 * dt**2) / (1.0 + gamma1 * dt)
                f2_et1.z[ii, jj, kk] = (gamma1 * dt - 1) / (1 + gamma1 * dt)
                f3_et1.z[ii, jj, kk] = (
                    c1 * ((w1 + gamma1) * dt**2 + 2 * dt) / (2 * (1 + gamma1 * dt))
                )
                f4_et1.z[ii, jj, kk] = (
                    c1 * ((w1 + gamma1) * dt**2 - 2 * dt) / (2 * (1 + gamma1 * dt))
                )

                f1_et2.z[ii, jj, kk] = (2.0 - w2**2 * dt**2) / (1.0 + gamma2 * dt)
                f2_et2.z[ii, jj, kk] = (gamma2 * dt - 1) / (1 + gamma2 * dt)
                f3_et2.z[ii, jj, kk] = (
                    c2 * ((w2 + gamma2) * dt**2 + 2 * dt) / (2 * (1 + gamma2 * dt))
                )
                f4_et2.z[ii, jj, kk] = (
                    c2 * ((w2 + gamma2) * dt**2 - 2 * dt) / (2 * (1 + gamma2 * dt))
                )

    return f1_et1, f2_et1, f3_et1, f4_et1, f1_et2, f2_et2, f3_et2, f4_et2


def create_sphere_drude(sphere, nsub, ddx, dt, wp, gamma, d1, d2, d3):

    # gax
    for x in range(
        int((sphere.x - sphere.R) / ddx) - 2, int((sphere.x + sphere.R) / ddx) + 2
    ):
        for y in range(
            int((sphere.y - sphere.R) / ddx) - 2, int((sphere.y + sphere.R) / ddx) + 2
        ):
            for z in range(
                int((sphere.z - sphere.R) / ddx) - 2,
                int((sphere.z + sphere.R) / ddx) + 2,
            ):
                p_xmin = x * ddx
                p_ymin = (y - 0.5) * ddx
                p_zmin = (z - 0.5) * ddx
                p_xmax = (x + 1) * ddx
                p_ymax = (y + 0.5) * ddx
                p_zmax = (z + 0.5) * ddx
                p_corners = [
                    (p_xmin, p_ymin, p_zmin),
                    (p_xmin, p_ymax, p_zmin),
                    (p_xmax, p_ymin, p_zmin),
                    (p_xmax, p_ymax, p_zmin),
                    (p_xmin, p_ymin, p_zmax),
                    (p_xmin, p_ymax, p_zmax),
                    (p_xmax, p_ymin, p_zmax),
                    (p_xmax, p_ymax, p_zmax),
                ]

                corner_in_box = np.full((8), False)
                i = 0
                for corner in p_corners:
                    dist = np.sqrt(
                        (corner[0] - sphere.x) ** 2
                        + (corner[1] - sphere.y) ** 2
                        + (corner[2] - sphere.z) ** 2
                    )
                    if dist <= sphere.R + 0.0001 * nm:
                        corner_in_box[i] = True
                    i += 1

                if all(corner_in_box):
                    d1.x[x, y, z] = 4.0 / (2.0 + gamma * dt)
                    d2.x[x, y, z] = (gamma * dt - 2.0) / (gamma * dt + 2.0)
                    d3.x[x, y, z] = 2.0 * wp**2 * dt**2 / (2.0 + gamma * dt)

                else:
                    nin = 0
                    nout = 0
                    xi = 0
                    yj = 0
                    zk = 0
                    for xi in range(0, nsub):
                        for yj in range(0, nsub):
                            for zk in range(0, nsub):
                                xsub = p_xmin + (xi + 0.5) / nsub / ddx
                                ysub = p_ymin + (yj + 0.5) / nsub / ddx
                                zsub = p_zmin + (zk + 0.5) / nsub / ddx
                                distsub = np.sqrt(
                                    (xsub - sphere.x) ** 2
                                    + (ysub - sphere.y) ** 2
                                    + (zsub - sphere.z) ** 2
                                )
                                if distsub <= sphere.R + 0.0001 * nm:
                                    nin += 1
                                else:
                                    nout += 1
                    d1.x[x, y, z] = (
                        4.0 / (2.0 + gamma * dt) * nin / (nout + nin)
                    )  # + 0.5*nout/(nout+nin)
                    d2.x[x, y, z] = (
                        (gamma * dt - 2.0) / (gamma * dt + 2.0) * nin / (nout + nin)
                    )  # + 0.5*nout/(nout+nin)
                    d3.x[x, y, z] = (
                        2.0 * wp**2 * dt**2 / (2.0 + gamma * dt) * nin / (nout + nin)
                    )

    # gay
    for x in range(
        int((sphere.x - sphere.R) / ddx) - 2, int((sphere.x + sphere.R) / ddx) + 2
    ):
        for y in range(
            int((sphere.y - sphere.R) / ddx) - 2, int((sphere.y + sphere.R) / ddx) + 2
        ):
            for z in range(
                int((sphere.z - sphere.R) / ddx) - 2,
                int((sphere.z + sphere.R) / ddx) + 2,
            ):
                p_xmin = (x - 0.5) * ddx
                p_ymin = y * ddx
                p_zmin = (z - 0.5) * ddx
                p_xmax = (x + 0.5) * ddx
                p_ymax = (y + 1.0) * ddx
                p_zmax = (z + 0.5) * ddx
                p_corners = [
                    (p_xmin, p_ymin, p_zmin),
                    (p_xmin, p_ymax, p_zmin),
                    (p_xmax, p_ymin, p_zmin),
                    (p_xmax, p_ymax, p_zmin),
                    (p_xmin, p_ymin, p_zmax),
                    (p_xmin, p_ymax, p_zmax),
                    (p_xmax, p_ymin, p_zmax),
                    (p_xmax, p_ymax, p_zmax),
                ]

                corner_in_box = np.full((8), False)
                i = 0
                for corner in p_corners:
                    dist = np.sqrt(
                        (corner[0] - sphere.x) ** 2
                        + (corner[1] - sphere.y) ** 2
                        + (corner[2] - sphere.z) ** 2
                    )
                    if dist <= sphere.R + 0.0001 * nm:
                        corner_in_box[i] = True
                    i += 1

                if all(corner_in_box):
                    d1.y[x, y, z] = 4.0 / (2.0 + gamma * dt)
                    d2.y[x, y, z] = (gamma * dt - 2.0) / (gamma * dt + 2.0)
                    d3.y[x, y, z] = 2.0 * wp**2 * dt**2 / (2.0 + gamma * dt)
                else:
                    nin = 0
                    nout = 0
                    xi = 0
                    yj = 0
                    zk = 0
                    for xi in range(0, nsub):
                        for yj in range(0, nsub):
                            for zk in range(0, nsub):
                                xsub = p_xmin + (xi + 0.5) / nsub / ddx
                                ysub = p_ymin + (yj + 0.5) / nsub / ddx
                                zsub = p_zmin + (zk + 0.5) / nsub / ddx
                                distsub = np.sqrt(
                                    (xsub - sphere.x) ** 2
                                    + (ysub - sphere.y) ** 2
                                    + (zsub - sphere.z) ** 2
                                )
                                if distsub <= sphere.R + 0.0001 * nm:
                                    nin += 1
                                else:
                                    nout += 1
                    d1.y[x, y, z] = (
                        4.0 / (2.0 + gamma * dt) * nin / (nout + nin)
                    )  # + 0.5*nout/(nout+nin)
                    d2.y[x, y, z] = (
                        (gamma * dt - 2.0) / (gamma * dt + 2.0) * nin / (nout + nin)
                    )  # + 0.5*nout/(nout+nin)
                    d3.y[x, y, z] = (
                        2.0 * wp**2 * dt**2 / (2.0 + gamma * dt) * nin / (nout + nin)
                    )

    # gaz
    for x in range(
        int((sphere.x - sphere.R) / ddx) - 2, int((sphere.x + sphere.R) / ddx) + 2
    ):
        for y in range(
            int((sphere.y - sphere.R) / ddx) - 2, int((sphere.y + sphere.R) / ddx) + 2
        ):
            for z in range(
                int((sphere.z - sphere.R) / ddx) - 2,
                int((sphere.z + sphere.R) / ddx) + 2,
            ):
                p_xmin = (x - 0.5) * ddx
                p_ymin = (y - 0.5) * ddx
                p_zmin = z * ddx
                p_xmax = (x + 0.5) * ddx
                p_ymax = (y + 0.5) * ddx
                p_zmax = (z + 1.0) * ddx
                p_corners = [
                    (p_xmin, p_ymin, p_zmin),
                    (p_xmin, p_ymax, p_zmin),
                    (p_xmax, p_ymin, p_zmin),
                    (p_xmax, p_ymax, p_zmin),
                    (p_xmin, p_ymin, p_zmax),
                    (p_xmin, p_ymax, p_zmax),
                    (p_xmax, p_ymin, p_zmax),
                    (p_xmax, p_ymax, p_zmax),
                ]

                corner_in_box = np.full((8), False)
                i = 0
                for corner in p_corners:
                    dist = np.sqrt(
                        (corner[0] - sphere.x) ** 2
                        + (corner[1] - sphere.y) ** 2
                        + (corner[2] - sphere.z) ** 2
                    )
                    if dist <= sphere.R + 0.0001 * nm:
                        corner_in_box[i] = True
                    i += 1

                if all(corner_in_box):
                    d1.z[x, y, z] = 4.0 / (2.0 + gamma * dt)
                    d2.z[x, y, z] = (gamma * dt - 2.0) / (gamma * dt + 2.0)
                    d3.z[x, y, z] = 2.0 * wp**2 * dt**2 / (2.0 + gamma * dt)
                else:
                    nin = 0
                    nout = 0
                    xi = 0
                    yj = 0
                    zk = 0
                    for xi in range(0, nsub):
                        for yj in range(0, nsub):
                            for zk in range(0, nsub):
                                xsub = p_xmin + (xi + 0.5) / nsub / ddx
                                ysub = p_ymin + (yj + 0.5) / nsub / ddx
                                zsub = p_zmin + (zk + 0.5) / nsub / ddx
                                distsub = np.sqrt(
                                    (xsub - sphere.x) ** 2
                                    + (ysub - sphere.y) ** 2
                                    + (zsub - sphere.z) ** 2
                                )
                                if distsub <= sphere.R + 0.0001 * nm:
                                    nin += 1
                                else:
                                    nout += 1
                    d1.z[x, y, z] = (
                        4.0 / (2.0 + gamma * dt) * nin / (nout + nin)
                    )  # + 0.5*nout/(nout+nin)
                    d2.z[x, y, z] = (
                        (gamma * dt - 2.0) / (gamma * dt + 2.0) * nin / (nout + nin)
                    )  # + 0.5*nout/(nout+nin)
                    d3.z[x, y, z] = (
                        2.0 * wp**2 * dt**2 / (2.0 + gamma * dt) * nin / (nout + nin)
                    )

    return d1, d2, d3


def create_sphere_etch(
    sphere,
    nsub,
    ddx,
    dt,
    c1,
    c2,
    w1,
    w2,
    gamma1,
    gamma2,
    f1_et1,
    f2_et1,
    f3_et1,
    f4_et1,
    f1_et2,
    f2_et2,
    f3_et2,
    f4_et2,
):

    # gax
    for x in range(
        int((sphere.x - sphere.R) / ddx) - 2, int((sphere.x + sphere.R) / ddx) + 2
    ):
        for y in range(
            int((sphere.y - sphere.R) / ddx) - 2, int((sphere.y + sphere.R) / ddx) + 2
        ):
            for z in range(
                int((sphere.z - sphere.R) / ddx) - 2,
                int((sphere.z + sphere.R) / ddx) + 2,
            ):
                p_xmin = x * ddx
                p_ymin = (y - 0.5) * ddx
                p_zmin = (z - 0.5) * ddx
                p_xmax = (x + 1) * ddx
                p_ymax = (y + 0.5) * ddx
                p_zmax = (z + 0.5) * ddx
                p_corners = [
                    (p_xmin, p_ymin, p_zmin),
                    (p_xmin, p_ymax, p_zmin),
                    (p_xmax, p_ymin, p_zmin),
                    (p_xmax, p_ymax, p_zmin),
                    (p_xmin, p_ymin, p_zmax),
                    (p_xmin, p_ymax, p_zmax),
                    (p_xmax, p_ymin, p_zmax),
                    (p_xmax, p_ymax, p_zmax),
                ]

                corner_in_box = np.full((8), False)
                i = 0
                for corner in p_corners:
                    dist = np.sqrt(
                        (corner[0] - sphere.x) ** 2
                        + (corner[1] - sphere.y) ** 2
                        + (corner[2] - sphere.z) ** 2
                    )
                    if dist <= sphere.R + 0.0001 * nm:
                        corner_in_box[i] = True
                    i += 1

                if all(corner_in_box):
                    # x components
                    f1_et1.x[x, y, z] = (2.0 - w1**2 * dt**2) / (1.0 + gamma1 * dt)
                    f2_et1.x[x, y, z] = (gamma1 * dt - 1) / (1 + gamma1 * dt)
                    f3_et1.x[x, y, z] = (
                        c1 * ((w1 + gamma1) * dt**2 + 2 * dt) / (2 * (1 + gamma1 * dt))
                    )
                    f4_et1.x[x, y, z] = (
                        c1 * ((w1 + gamma1) * dt**2 - 2 * dt) / (2 * (1 + gamma1 * dt))
                    )

                    f1_et2.x[x, y, z] = (2.0 - w2**2 * dt**2) / (1.0 + gamma2 * dt)
                    f2_et2.x[x, y, z] = (gamma2 * dt - 1) / (1 + gamma2 * dt)
                    f3_et2.x[x, y, z] = (
                        c2 * ((w2 + gamma2) * dt**2 + 2 * dt) / (2 * (1 + gamma2 * dt))
                    )
                    f4_et2.x[x, y, z] = (
                        c2 * ((w2 + gamma2) * dt**2 - 2 * dt) / (2 * (1 + gamma2 * dt))
                    )

                else:
                    nin = 0
                    nout = 0
                    xi = 0
                    yj = 0
                    zk = 0
                    for xi in range(0, nsub):
                        for yj in range(0, nsub):
                            for zk in range(0, nsub):
                                xsub = p_xmin + (xi + 0.5) / nsub / ddx
                                ysub = p_ymin + (yj + 0.5) / nsub / ddx
                                zsub = p_zmin + (zk + 0.5) / nsub / ddx
                                distsub = np.sqrt(
                                    (xsub - sphere.x) ** 2
                                    + (ysub - sphere.y) ** 2
                                    + (zsub - sphere.z) ** 2
                                )
                                if distsub <= sphere.R + 0.0001 * nm:
                                    nin += 1
                                else:
                                    nout += 1
                    f1_et1.x[x, y, z] = (
                        ((2.0 - w1**2 * dt**2) / (1.0 + gamma1 * dt))
                        * nin
                        / (nout + nin)
                    )  # + 0.5*nout/(nout+nin)
                    f2_et1.x[x, y, z] = (
                        ((gamma1 * dt - 1) / (1 + gamma1 * dt)) * nin / (nout + nin)
                    )  # + 0.5*nout/(nout+nin)
                    f3_et1.x[x, y, z] = (
                        (
                            c1
                            * ((w1 + gamma1) * dt**2 + 2 * dt)
                            / (2 * (1 + gamma1 * dt))
                        )
                        * nin
                        / (nout + nin)
                    )
                    f4_et1.x[x, y, z] = (
                        (
                            c1
                            * ((w1 + gamma1) * dt**2 - 2 * dt)
                            / (2 * (1 + gamma1 * dt))
                        )
                        * nin
                        / (nout + nin)
                    )

                    f1_et2.x[x, y, z] = (
                        ((2.0 - w2**2 * dt**2) / (1.0 + gamma2 * dt))
                        * nin
                        / (nout + nin)
                    )
                    f2_et2.x[x, y, z] = (
                        ((gamma2 * dt - 1) / (1 + gamma2 * dt)) * nin / (nout + nin)
                    )
                    f3_et2.x[x, y, z] = (
                        (
                            c2
                            * ((w2 + gamma2) * dt**2 + 2 * dt)
                            / (2 * (1 + gamma2 * dt))
                        )
                        * nin
                        / (nout + nin)
                    )
                    f4_et2.x[x, y, z] = (
                        (
                            c2
                            * ((w2 + gamma2) * dt**2 - 2 * dt)
                            / (2 * (1 + gamma2 * dt))
                        )
                        * nin
                        / (nout + nin)
                    )
                # gay
    for x in range(
        int((sphere.x - sphere.R) / ddx) - 2, int((sphere.x + sphere.R) / ddx) + 2
    ):
        for y in range(
            int((sphere.y - sphere.R) / ddx) - 2, int((sphere.y + sphere.R) / ddx) + 2
        ):
            for z in range(
                int((sphere.z - sphere.R) / ddx) - 2,
                int((sphere.z + sphere.R) / ddx) + 2,
            ):
                p_xmin = (x - 0.5) * ddx
                p_ymin = y * ddx
                p_zmin = (z - 0.5) * ddx
                p_xmax = (x + 0.5) * ddx
                p_ymax = (y + 1.0) * ddx
                p_zmax = (z + 0.5) * ddx
                p_corners = [
                    (p_xmin, p_ymin, p_zmin),
                    (p_xmin, p_ymax, p_zmin),
                    (p_xmax, p_ymin, p_zmin),
                    (p_xmax, p_ymax, p_zmin),
                    (p_xmin, p_ymin, p_zmax),
                    (p_xmin, p_ymax, p_zmax),
                    (p_xmax, p_ymin, p_zmax),
                    (p_xmax, p_ymax, p_zmax),
                ]

                corner_in_box = np.full((8), False)
                i = 0
                for corner in p_corners:
                    dist = np.sqrt(
                        (corner[0] - sphere.x) ** 2
                        + (corner[1] - sphere.y) ** 2
                        + (corner[2] - sphere.z) ** 2
                    )
                    if dist <= sphere.R + 0.0001 * nm:
                        corner_in_box[i] = True
                    i += 1

                if all(corner_in_box):
                    f1_et1.y[x, y, z] = (2.0 - w1**2 * dt**2) / (1.0 + gamma1 * dt)
                    f2_et1.y[x, y, z] = (gamma1 * dt - 1) / (1 + gamma1 * dt)
                    f3_et1.y[x, y, z] = (
                        c1 * ((w1 + gamma1) * dt**2 + 2 * dt) / (2 * (1 + gamma1 * dt))
                    )
                    f4_et1.y[x, y, z] = (
                        c1 * ((w1 + gamma1) * dt**2 - 2 * dt) / (2 * (1 + gamma1 * dt))
                    )

                    f1_et2.y[x, y, z] = (2.0 - w2**2 * dt**2) / (1.0 + gamma2 * dt)
                    f2_et2.y[x, y, z] = (gamma2 * dt - 1) / (1 + gamma2 * dt)
                    f3_et2.y[x, y, z] = (
                        c2 * ((w2 + gamma2) * dt**2 + 2 * dt) / (2 * (1 + gamma2 * dt))
                    )
                    f4_et2.y[x, y, z] = (
                        c2 * ((w2 + gamma2) * dt**2 - 2 * dt) / (2 * (1 + gamma2 * dt))
                    )

                else:
                    nin = 0
                    nout = 0
                    xi = 0
                    yj = 0
                    zk = 0
                    for xi in range(0, nsub):
                        for yj in range(0, nsub):
                            for zk in range(0, nsub):
                                xsub = p_xmin + (xi + 0.5) / nsub / ddx
                                ysub = p_ymin + (yj + 0.5) / nsub / ddx
                                zsub = p_zmin + (zk + 0.5) / nsub / ddx
                                distsub = np.sqrt(
                                    (xsub - sphere.x) ** 2
                                    + (ysub - sphere.y) ** 2
                                    + (zsub - sphere.z) ** 2
                                )
                                if distsub <= sphere.R + 0.0001 * nm:
                                    nin += 1
                                else:
                                    nout += 1
                    f1_et1.y[x, y, z] = (
                        ((2.0 - w1**2 * dt**2) / (1.0 + gamma1 * dt))
                        * nin
                        / (nout + nin)
                    )  # + 0.5*nout/(nout+nin)
                    f2_et1.y[x, y, z] = (
                        ((gamma1 * dt - 1) / (1 + gamma1 * dt)) * nin / (nout + nin)
                    )  # + 0.5*nout/(nout+nin)
                    f3_et1.y[x, y, z] = (
                        (
                            c1
                            * ((w1 + gamma1) * dt**2 + 2 * dt)
                            / (2 * (1 + gamma1 * dt))
                        )
                        * nin
                        / (nout + nin)
                    )
                    f4_et1.y[x, y, z] = (
                        (
                            c1
                            * ((w1 + gamma1) * dt**2 - 2 * dt)
                            / (2 * (1 + gamma1 * dt))
                        )
                        * nin
                        / (nout + nin)
                    )

                    f1_et2.y[x, y, z] = (
                        ((2.0 - w2**2 * dt**2) / (1.0 + gamma2 * dt))
                        * nin
                        / (nout + nin)
                    )
                    f2_et2.y[x, y, z] = (
                        ((gamma2 * dt - 1) / (1 + gamma2 * dt)) * nin / (nout + nin)
                    )
                    f3_et2.y[x, y, z] = (
                        (
                            c2
                            * ((w2 + gamma2) * dt**2 + 2 * dt)
                            / (2 * (1 + gamma2 * dt))
                        )
                        * nin
                        / (nout + nin)
                    )
                    f4_et2.y[x, y, z] = (
                        (
                            c2
                            * ((w2 + gamma2) * dt**2 - 2 * dt)
                            / (2 * (1 + gamma2 * dt))
                        )
                        * nin
                        / (nout + nin)
                    )

    # gaz
    for x in range(
        int((sphere.x - sphere.R) / ddx) - 2, int((sphere.x + sphere.R) / ddx) + 2
    ):
        for y in range(
            int((sphere.y - sphere.R) / ddx) - 2, int((sphere.y + sphere.R) / ddx) + 2
        ):
            for z in range(
                int((sphere.z - sphere.R) / ddx) - 2,
                int((sphere.z + sphere.R) / ddx) + 2,
            ):
                p_xmin = (x - 0.5) * ddx
                p_ymin = (y - 0.5) * ddx
                p_zmin = z * ddx
                p_xmax = (x + 0.5) * ddx
                p_ymax = (y + 0.5) * ddx
                p_zmax = (z + 1.0) * ddx
                p_corners = [
                    (p_xmin, p_ymin, p_zmin),
                    (p_xmin, p_ymax, p_zmin),
                    (p_xmax, p_ymin, p_zmin),
                    (p_xmax, p_ymax, p_zmin),
                    (p_xmin, p_ymin, p_zmax),
                    (p_xmin, p_ymax, p_zmax),
                    (p_xmax, p_ymin, p_zmax),
                    (p_xmax, p_ymax, p_zmax),
                ]

                corner_in_box = np.full((8), False)
                i = 0
                for corner in p_corners:
                    dist = np.sqrt(
                        (corner[0] - sphere.x) ** 2
                        + (corner[1] - sphere.y) ** 2
                        + (corner[2] - sphere.z) ** 2
                    )
                    if dist <= sphere.R + 0.0001 * nm:
                        corner_in_box[i] = True
                    i += 1

                if all(corner_in_box):
                    f1_et1.y[x, y, z] = (2.0 - w1**2 * dt**2) / (1.0 + gamma1 * dt)
                    f2_et1.y[x, y, z] = (gamma1 * dt - 1) / (1 + gamma1 * dt)
                    f3_et1.y[x, y, z] = (
                        c1 * ((w1 + gamma1) * dt**2 + 2 * dt) / (2 * (1 + gamma1 * dt))
                    )
                    f4_et1.y[x, y, z] = (
                        c1 * ((w1 + gamma1) * dt**2 - 2 * dt) / (2 * (1 + gamma1 * dt))
                    )

                    f1_et2.y[x, y, z] = (2.0 - w2**2 * dt**2) / (1.0 + gamma2 * dt)
                    f2_et2.y[x, y, z] = (gamma2 * dt - 1) / (1 + gamma2 * dt)
                    f3_et2.y[x, y, z] = (
                        c2 * ((w2 + gamma2) * dt**2 + 2 * dt) / (2 * (1 + gamma2 * dt))
                    )
                    f4_et2.y[x, y, z] = (
                        c2 * ((w2 + gamma2) * dt**2 - 2 * dt) / (2 * (1 + gamma2 * dt))
                    )
                else:
                    nin = 0
                    nout = 0
                    xi = 0
                    yj = 0
                    zk = 0
                    for xi in range(0, nsub):
                        for yj in range(0, nsub):
                            for zk in range(0, nsub):
                                xsub = p_xmin + (xi + 0.5) / nsub / ddx
                                ysub = p_ymin + (yj + 0.5) / nsub / ddx
                                zsub = p_zmin + (zk + 0.5) / nsub / ddx
                                distsub = np.sqrt(
                                    (xsub - sphere.x) ** 2
                                    + (ysub - sphere.y) ** 2
                                    + (zsub - sphere.z) ** 2
                                )
                                if distsub <= sphere.R + 0.0001 * nm:
                                    nin += 1
                                else:
                                    nout += 1
                    f1_et1.z[x, y, z] = (
                        ((2.0 - w1**2 * dt**2) / (1.0 + gamma1 * dt))
                        * nin
                        / (nout + nin)
                    )  # + 0.5*nout/(nout+nin)
                    f2_et1.z[x, y, z] = (
                        ((gamma1 * dt - 1) / (1 + gamma1 * dt)) * nin / (nout + nin)
                    )  # + 0.5*nout/(nout+nin)
                    f3_et1.z[x, y, z] = (
                        (
                            c1
                            * ((w1 + gamma1) * dt**2 + 2 * dt)
                            / (2 * (1 + gamma1 * dt))
                        )
                        * nin
                        / (nout + nin)
                    )
                    f4_et1.z[x, y, z] = (
                        (
                            c1
                            * ((w1 + gamma1) * dt**2 - 2 * dt)
                            / (2 * (1 + gamma1 * dt))
                        )
                        * nin
                        / (nout + nin)
                    )

                    f1_et2.z[x, y, z] = (
                        ((2.0 - w2**2 * dt**2) / (1.0 + gamma2 * dt))
                        * nin
                        / (nout + nin)
                    )
                    f2_et2.z[x, y, z] = (
                        ((gamma2 * dt - 1) / (1 + gamma2 * dt)) * nin / (nout + nin)
                    )
                    f3_et2.z[x, y, z] = (
                        (
                            c2
                            * ((w2 + gamma2) * dt**2 + 2 * dt)
                            / (2 * (1 + gamma2 * dt))
                        )
                        * nin
                        / (nout + nin)
                    )
                    f4_et2.z[x, y, z] = (
                        (
                            c2
                            * ((w2 + gamma2) * dt**2 - 2 * dt)
                            / (2 * (1 + gamma2 * dt))
                        )
                        * nin
                        / (nout + nin)
                    )

    return f1_et1, f2_et1, f3_et1, f4_et1, f1_et2, f2_et2, f3_et2, f4_et2


def create_sphere_eps(sphere, nsub, ddx, dt, eps_in, eps_out, ga):

    # gax
    for x in range(
        int((sphere.x - sphere.R) / ddx) - 2, int((sphere.x + sphere.R) / ddx) + 2
    ):
        for y in range(
            int((sphere.y - sphere.R) / ddx) - 2, int((sphere.y + sphere.R) / ddx) + 2
        ):
            for z in range(
                int((sphere.z - sphere.R) / ddx) - 2,
                int((sphere.z + sphere.R) / ddx) + 2,
            ):
                # dielectric part
                p_xmin = x * ddx
                p_ymin = (y - 0.5) * ddx
                p_zmin = (z - 0.5) * ddx
                p_xmax = (x + 1) * ddx
                p_ymax = (y + 0.5) * ddx
                p_zmax = (z + 0.5) * ddx
                p_corners = [
                    (p_xmin, p_ymin, p_zmin),
                    (p_xmin, p_ymax, p_zmin),
                    (p_xmax, p_ymin, p_zmin),
                    (p_xmax, p_ymax, p_zmin),
                    (p_xmin, p_ymin, p_zmax),
                    (p_xmin, p_ymax, p_zmax),
                    (p_xmax, p_ymin, p_zmax),
                    (p_xmax, p_ymax, p_zmax),
                ]

                corner_in_box = np.full((8), False)
                i = 0
                for corner in p_corners:
                    dist = np.sqrt(
                        (corner[0] - sphere.x) ** 2
                        + (corner[1] - sphere.y) ** 2
                        + (corner[2] - sphere.z) ** 2
                    )
                    if dist <= sphere.R + 0.0001 * nm:
                        corner_in_box[i] = True
                    i += 1

                if all(corner_in_box):
                    ga.x[x, y, z] = 1.0 / eps_in

                elif not any(corner_in_box):
                    ga.x[x, y, z] = 1.0 / eps_out
                else:
                    nin = 0
                    nout = 0
                    xi = 0
                    yj = 0
                    zk = 0
                    for xi in range(0, nsub):
                        for yj in range(0, nsub):
                            for zk in range(0, nsub):
                                xsub = p_xmin + (xi + 0.5) / nsub / ddx
                                ysub = p_ymin + (yj + 0.5) / nsub / ddx
                                zsub = p_zmin + (zk + 0.5) / nsub / ddx
                                distsub = np.sqrt(
                                    (xsub - sphere.x) ** 2
                                    + (ysub - sphere.y) ** 2
                                    + (zsub - sphere.z) ** 2
                                )
                                if distsub <= sphere.R + 0.0001 * nm:
                                    nin += 1
                                else:
                                    nout += 1
                    ga.x[x, y, z] = (nout + nin) / (eps_in * nin + eps_out * nout)

    # gay
    for x in range(
        int((sphere.x - sphere.R) / ddx) - 2, int((sphere.x + sphere.R) / ddx) + 2
    ):
        for y in range(
            int((sphere.y - sphere.R) / ddx) - 2, int((sphere.y + sphere.R) / ddx) + 2
        ):
            for z in range(
                int((sphere.z - sphere.R) / ddx) - 2,
                int((sphere.z + sphere.R) / ddx) + 2,
            ):
                p_xmin = (x - 0.5) * ddx
                p_ymin = y * ddx
                p_zmin = (z - 0.5) * ddx
                p_xmax = (x + 0.5) * ddx
                p_ymax = (y + 1.0) * ddx
                p_zmax = (z + 0.5) * ddx
                p_corners = [
                    (p_xmin, p_ymin, p_zmin),
                    (p_xmin, p_ymax, p_zmin),
                    (p_xmax, p_ymin, p_zmin),
                    (p_xmax, p_ymax, p_zmin),
                    (p_xmin, p_ymin, p_zmax),
                    (p_xmin, p_ymax, p_zmax),
                    (p_xmax, p_ymin, p_zmax),
                    (p_xmax, p_ymax, p_zmax),
                ]

                corner_in_box = np.full((8), False)
                i = 0
                for corner in p_corners:
                    dist = np.sqrt(
                        (corner[0] - sphere.x) ** 2
                        + (corner[1] - sphere.y) ** 2
                        + (corner[2] - sphere.z) ** 2
                    )
                    if dist <= sphere.R + 0.0001 * nm:
                        corner_in_box[i] = True
                    i += 1

                if all(corner_in_box):
                    ga.y[x, y, z] = 1.0 / eps_in

                elif not any(corner_in_box):
                    ga.y[x, y, z] = 1.0 / eps_out
                else:
                    nin = 0
                    nout = 0
                    xi = 0
                    yj = 0
                    zk = 0
                    for xi in range(0, nsub):
                        for yj in range(0, nsub):
                            for zk in range(0, nsub):
                                xsub = p_xmin + (xi + 0.5) / nsub / ddx
                                ysub = p_ymin + (yj + 0.5) / nsub / ddx
                                zsub = p_zmin + (zk + 0.5) / nsub / ddx
                                distsub = np.sqrt(
                                    (xsub - sphere.x) ** 2
                                    + (ysub - sphere.y) ** 2
                                    + (zsub - sphere.z) ** 2
                                )
                                if distsub <= sphere.R + 0.0001 * nm:
                                    nin += 1
                                else:
                                    nout += 1
                    ga.y[x, y, z] = (nout + nin) / (eps_in * nin + eps_out * nout)

    # gaz
    for x in range(
        int((sphere.x - sphere.R) / ddx) - 2, int((sphere.x + sphere.R) / ddx) + 2
    ):
        for y in range(
            int((sphere.y - sphere.R) / ddx) - 2, int((sphere.y + sphere.R) / ddx) + 2
        ):
            for z in range(
                int((sphere.z - sphere.R) / ddx) - 2,
                int((sphere.z + sphere.R) / ddx) + 2,
            ):
                p_xmin = (x - 0.5) * ddx
                p_ymin = (y - 0.5) * ddx
                p_zmin = z * ddx
                p_xmax = (x + 0.5) * ddx
                p_ymax = (y + 0.5) * ddx
                p_zmax = (z + 1.0) * ddx
                p_corners = [
                    (p_xmin, p_ymin, p_zmin),
                    (p_xmin, p_ymax, p_zmin),
                    (p_xmax, p_ymin, p_zmin),
                    (p_xmax, p_ymax, p_zmin),
                    (p_xmin, p_ymin, p_zmax),
                    (p_xmin, p_ymax, p_zmax),
                    (p_xmax, p_ymin, p_zmax),
                    (p_xmax, p_ymax, p_zmax),
                ]

                corner_in_box = np.full((8), False)
                i = 0
                for corner in p_corners:
                    dist = np.sqrt(
                        (corner[0] - sphere.x) ** 2
                        + (corner[1] - sphere.y) ** 2
                        + (corner[2] - sphere.z) ** 2
                    )
                    if dist <= sphere.R + 0.0001 * nm:
                        corner_in_box[i] = True
                    i += 1

                if all(corner_in_box):
                    ga.z[x, y, z] = 1.0 / eps_in
                elif not any(corner_in_box):
                    ga.z[x, y, z] = 1.0 / eps_out
                else:
                    nin = 0
                    nout = 0
                    xi = 0
                    yj = 0
                    zk = 0
                    for xi in range(0, nsub):
                        for yj in range(0, nsub):
                            for zk in range(0, nsub):
                                xsub = p_xmin + (xi + 0.5) / nsub / ddx
                                ysub = p_ymin + (yj + 0.5) / nsub / ddx
                                zsub = p_zmin + (zk + 0.5) / nsub / ddx
                                distsub = np.sqrt(
                                    (xsub - sphere.x) ** 2
                                    + (ysub - sphere.y) ** 2
                                    + (zsub - sphere.z) ** 2
                                )
                                if distsub <= sphere.R + 0.0001 * nm:
                                    nin += 1
                                else:
                                    nout += 1
                    ga.z[x, y, z] = (nout + nin) / (eps_in * nin + eps_out * nout)
    return ga


def create_sphere_drude_eps(
    sphere, nsub, ddx, dt, eps_in, eps_out, wp, gamma, ga, d1, d2, d3
):

    # gax
    for x in range(
        int((sphere.x - sphere.R) / ddx) - 2, int((sphere.x + sphere.R) / ddx) + 2
    ):
        for y in range(
            int((sphere.y - sphere.R) / ddx) - 2, int((sphere.y + sphere.R) / ddx) + 2
        ):
            for z in range(
                int((sphere.z - sphere.R) / ddx) - 2,
                int((sphere.z + sphere.R) / ddx) + 2,
            ):
                # dielectric part
                p_xmin = x * ddx
                p_ymin = (y - 0.5) * ddx
                p_zmin = (z - 0.5) * ddx
                p_xmax = (x + 1) * ddx
                p_ymax = (y + 0.5) * ddx
                p_zmax = (z + 0.5) * ddx
                p_corners = [
                    (p_xmin, p_ymin, p_zmin),
                    (p_xmin, p_ymax, p_zmin),
                    (p_xmax, p_ymin, p_zmin),
                    (p_xmax, p_ymax, p_zmin),
                    (p_xmin, p_ymin, p_zmax),
                    (p_xmin, p_ymax, p_zmax),
                    (p_xmax, p_ymin, p_zmax),
                    (p_xmax, p_ymax, p_zmax),
                ]

                corner_in_box = np.full((8), False)
                i = 0
                for corner in p_corners:
                    dist = np.sqrt(
                        (corner[0] - sphere.x) ** 2
                        + (corner[1] - sphere.y) ** 2
                        + (corner[2] - sphere.z) ** 2
                    )
                    if dist <= sphere.R + 0.0001 * nm:
                        corner_in_box[i] = True
                    i += 1

                if all(corner_in_box):
                    ga.x[x, y, z] = 1.0 / eps_in
                    d1.x[x, y, z] = 4.0 / (2.0 + gamma * dt)
                    d2.x[x, y, z] = (gamma * dt - 2.0) / (gamma * dt + 2.0)
                    d3.x[x, y, z] = 2.0 * wp**2 * dt**2 / (2.0 + gamma * dt)

                elif not any(corner_in_box):
                    ga.x[x, y, z] = 1.0 / eps_out
                else:
                    nin = 0
                    nout = 0
                    xi = 0
                    yj = 0
                    zk = 0
                    for xi in range(0, nsub):
                        for yj in range(0, nsub):
                            for zk in range(0, nsub):
                                xsub = p_xmin + (xi + 0.5) / nsub / ddx
                                ysub = p_ymin + (yj + 0.5) / nsub / ddx
                                zsub = p_zmin + (zk + 0.5) / nsub / ddx
                                distsub = np.sqrt(
                                    (xsub - sphere.x) ** 2
                                    + (ysub - sphere.y) ** 2
                                    + (zsub - sphere.z) ** 2
                                )
                                if distsub <= sphere.R + 0.0001 * nm:
                                    nin += 1
                                else:
                                    nout += 1
                    ga.x[x, y, z] = (nout + nin) / (eps_in * nin + eps_out * nout)
                    d1.x[x, y, z] = (
                        4.0 / (2.0 + gamma * dt) * nin / (nout + nin)
                    )  # + 0.5*nout/(nout+nin)
                    d2.x[x, y, z] = (
                        (gamma * dt - 2.0) / (gamma * dt + 2.0) * nin / (nout + nin)
                    )  # + 0.5*nout/(nout+nin)
                    d3.x[x, y, z] = (
                        2.0 * wp**2 * dt**2 / (2.0 + gamma * dt) * nin / (nout + nin)
                    )

    # gay
    for x in range(
        int((sphere.x - sphere.R) / ddx) - 2, int((sphere.x + sphere.R) / ddx) + 2
    ):
        for y in range(
            int((sphere.y - sphere.R) / ddx) - 2, int((sphere.y + sphere.R) / ddx) + 2
        ):
            for z in range(
                int((sphere.z - sphere.R) / ddx) - 2,
                int((sphere.z + sphere.R) / ddx) + 2,
            ):
                p_xmin = (x - 0.5) * ddx
                p_ymin = y * ddx
                p_zmin = (z - 0.5) * ddx
                p_xmax = (x + 0.5) * ddx
                p_ymax = (y + 1.0) * ddx
                p_zmax = (z + 0.5) * ddx
                p_corners = [
                    (p_xmin, p_ymin, p_zmin),
                    (p_xmin, p_ymax, p_zmin),
                    (p_xmax, p_ymin, p_zmin),
                    (p_xmax, p_ymax, p_zmin),
                    (p_xmin, p_ymin, p_zmax),
                    (p_xmin, p_ymax, p_zmax),
                    (p_xmax, p_ymin, p_zmax),
                    (p_xmax, p_ymax, p_zmax),
                ]

                corner_in_box = np.full((8), False)
                i = 0
                for corner in p_corners:
                    dist = np.sqrt(
                        (corner[0] - sphere.x) ** 2
                        + (corner[1] - sphere.y) ** 2
                        + (corner[2] - sphere.z) ** 2
                    )
                    if dist <= sphere.R + 0.0001 * nm:
                        corner_in_box[i] = True
                    i += 1

                if all(corner_in_box):
                    ga.y[x, y, z] = 1.0 / eps_in
                    d1.y[x, y, z] = 4.0 / (2.0 + gamma * dt)
                    d2.y[x, y, z] = (gamma * dt - 2.0) / (gamma * dt + 2.0)
                    d3.y[x, y, z] = 2.0 * wp**2 * dt**2 / (2.0 + gamma * dt)
                elif not any(corner_in_box):
                    ga.y[x, y, z] = 1.0 / eps_out
                else:
                    nin = 0
                    nout = 0
                    xi = 0
                    yj = 0
                    zk = 0
                    for xi in range(0, nsub):
                        for yj in range(0, nsub):
                            for zk in range(0, nsub):
                                xsub = p_xmin + (xi + 0.5) / nsub / ddx
                                ysub = p_ymin + (yj + 0.5) / nsub / ddx
                                zsub = p_zmin + (zk + 0.5) / nsub / ddx
                                distsub = np.sqrt(
                                    (xsub - sphere.x) ** 2
                                    + (ysub - sphere.y) ** 2
                                    + (zsub - sphere.z) ** 2
                                )
                                if distsub <= sphere.R + 0.0001 * nm:
                                    nin += 1
                                else:
                                    nout += 1
                    ga.y[x, y, z] = (nout + nin) / (eps_in * nin + eps_out * nout)
                    d1.y[x, y, z] = (
                        4.0 / (2.0 + gamma * dt) * nin / (nout + nin)
                    )  # + 0.5*nout/(nout+nin)
                    d2.y[x, y, z] = (
                        (gamma * dt - 2.0) / (gamma * dt + 2.0) * nin / (nout + nin)
                    )  # + 0.5*nout/(nout+nin)
                    d3.y[x, y, z] = (
                        2.0 * wp**2 * dt**2 / (2.0 + gamma * dt) * nin / (nout + nin)
                    )

    # gaz
    for x in range(
        int((sphere.x - sphere.R) / ddx) - 2, int((sphere.x + sphere.R) / ddx) + 2
    ):
        for y in range(
            int((sphere.y - sphere.R) / ddx) - 2, int((sphere.y + sphere.R) / ddx) + 2
        ):
            for z in range(
                int((sphere.z - sphere.R) / ddx) - 2,
                int((sphere.z + sphere.R) / ddx) + 2,
            ):
                p_xmin = (x - 0.5) * ddx
                p_ymin = (y - 0.5) * ddx
                p_zmin = z * ddx
                p_xmax = (x + 0.5) * ddx
                p_ymax = (y + 0.5) * ddx
                p_zmax = (z + 1.0) * ddx
                p_corners = [
                    (p_xmin, p_ymin, p_zmin),
                    (p_xmin, p_ymax, p_zmin),
                    (p_xmax, p_ymin, p_zmin),
                    (p_xmax, p_ymax, p_zmin),
                    (p_xmin, p_ymin, p_zmax),
                    (p_xmin, p_ymax, p_zmax),
                    (p_xmax, p_ymin, p_zmax),
                    (p_xmax, p_ymax, p_zmax),
                ]

                corner_in_box = np.full((8), False)
                i = 0
                for corner in p_corners:
                    dist = np.sqrt(
                        (corner[0] - sphere.x) ** 2
                        + (corner[1] - sphere.y) ** 2
                        + (corner[2] - sphere.z) ** 2
                    )
                    if dist <= sphere.R + 0.0001 * nm:
                        corner_in_box[i] = True
                    i += 1

                if all(corner_in_box):
                    ga.z[x, y, z] = 1.0 / eps_in
                    d1.z[x, y, z] = 4.0 / (2.0 + gamma * dt)
                    d2.z[x, y, z] = (gamma * dt - 2.0) / (gamma * dt + 2.0)
                    d3.z[x, y, z] = 2.0 * wp**2 * dt**2 / (2.0 + gamma * dt)
                elif not any(corner_in_box):
                    ga.z[x, y, z] = 1.0 / eps_out
                else:
                    nin = 0
                    nout = 0
                    xi = 0
                    yj = 0
                    zk = 0
                    for xi in range(0, nsub):
                        for yj in range(0, nsub):
                            for zk in range(0, nsub):
                                xsub = p_xmin + (xi + 0.5) / nsub / ddx
                                ysub = p_ymin + (yj + 0.5) / nsub / ddx
                                zsub = p_zmin + (zk + 0.5) / nsub / ddx
                                distsub = np.sqrt(
                                    (xsub - sphere.x) ** 2
                                    + (ysub - sphere.y) ** 2
                                    + (zsub - sphere.z) ** 2
                                )
                                if distsub <= sphere.R + 0.0001 * nm:
                                    nin += 1
                                else:
                                    nout += 1
                    ga.z[x, y, z] = (nout + nin) / (eps_in * nin + eps_out * nout)
                    d1.z[x, y, z] = (
                        4.0 / (2.0 + gamma * dt) * nin / (nout + nin)
                    )  # + 0.5*nout/(nout+nin)
                    d2.z[x, y, z] = (
                        (gamma * dt - 2.0) / (gamma * dt + 2.0) * nin / (nout + nin)
                    )  # + 0.5*nout/(nout+nin)
                    d3.z[x, y, z] = (
                        2.0 * wp**2 * dt**2 / (2.0 + gamma * dt) * nin / (nout + nin)
                    )
    return ga, d1, d2, d3


def create_sphere_lorentz(
    sphere, nsub, ddx, dt, eps_in, eps_out, wl, gamma_l, delta_eps, ga, l1, l2, l3
):

    # gax
    for x in range(
        int((sphere.x - sphere.R) / ddx) - 2, int((sphere.x + sphere.R) / ddx) + 2
    ):
        for y in range(
            int((sphere.y - sphere.R) / ddx) - 2, int((sphere.y + sphere.R) / ddx) + 2
        ):
            for z in range(
                int((sphere.z - sphere.R) / ddx) - 2,
                int((sphere.z + sphere.R) / ddx) + 2,
            ):
                # dielectric part
                p_xmin = x * ddx
                p_ymin = (y - 0.5) * ddx
                p_zmin = (z - 0.5) * ddx
                p_xmax = (x + 1) * ddx
                p_ymax = (y + 0.5) * ddx
                p_zmax = (z + 0.5) * ddx
                p_corners = [
                    (p_xmin, p_ymin, p_zmin),
                    (p_xmin, p_ymax, p_zmin),
                    (p_xmax, p_ymin, p_zmin),
                    (p_xmax, p_ymax, p_zmin),
                    (p_xmin, p_ymin, p_zmax),
                    (p_xmin, p_ymax, p_zmax),
                    (p_xmax, p_ymin, p_zmax),
                    (p_xmax, p_ymax, p_zmax),
                ]

                corner_in_box = np.full((8), False)
                i = 0
                for corner in p_corners:
                    dist = np.sqrt(
                        (corner[0] - sphere.x) ** 2
                        + (corner[1] - sphere.y) ** 2
                        + (corner[2] - sphere.z) ** 2
                    )
                    if dist <= sphere.R + 0.0001 * nm:
                        corner_in_box[i] = True
                    i += 1

                if all(corner_in_box):
                    ga.x[x, y, z] = 1.0 / eps_in
                    l1.x[x, y, z] = (2.0 - wl**2 * dt**2) / (1.0 + gamma_l * dt / 2.0)
                    l2.x[x, y, z] = (gamma_l * dt / 2.0 - 1.0) / (
                        1.0 + gamma_l * dt / 2.0
                    )
                    l3.x[x, y, z] = (
                        delta_eps * wl**2 * dt**2 / (1.0 + gamma_l * dt / 2.0)
                    )

                elif not any(corner_in_box):
                    ga.x[x, y, z] = 1.0 / eps_out
                else:
                    nin = 0
                    nout = 0
                    xi = 0
                    yj = 0
                    zk = 0
                    for xi in range(0, nsub):
                        for yj in range(0, nsub):
                            for zk in range(0, nsub):
                                xsub = p_xmin + (xi + 0.5) / nsub / ddx
                                ysub = p_ymin + (yj + 0.5) / nsub / ddx
                                zsub = p_zmin + (zk + 0.5) / nsub / ddx
                                distsub = np.sqrt(
                                    (xsub - sphere.x) ** 2
                                    + (ysub - sphere.y) ** 2
                                    + (zsub - sphere.z) ** 2
                                )
                                if distsub <= sphere.R + 0.0001 * nm:
                                    nin += 1
                                else:
                                    nout += 1
                    ga.x[x, y, z] = (nout + nin) / (eps_in * nin + eps_out * nout)
                    l1.x[x, y, z] = (
                        (2.0 - wl**2 * dt**2)
                        / (1.0 + gamma_l * dt / 2.0)
                        * nin
                        / (nout + nin)
                    )  # + 0.5*nout/(nout+nin)
                    l2.x[x, y, z] = (
                        (gamma_l * dt / 2.0 - 1.0)
                        / (1.0 + gamma_l * dt / 2.0)
                        * nin
                        / (nout + nin)
                    )  # + 0.5*nout/(nout+nin)
                    l3.x[x, y, z] = (
                        delta_eps
                        * wl**2
                        * dt**2
                        / (1.0 + gamma_l * dt / 2.0)
                        * nin
                        / (nout + nin)
                    )

    # gay
    for x in range(
        int((sphere.x - sphere.R) / ddx) - 2, int((sphere.x + sphere.R) / ddx) + 2
    ):
        for y in range(
            int((sphere.y - sphere.R) / ddx) - 2, int((sphere.y + sphere.R) / ddx) + 2
        ):
            for z in range(
                int((sphere.z - sphere.R) / ddx) - 2,
                int((sphere.z + sphere.R) / ddx) + 2,
            ):
                p_xmin = (x - 0.5) * ddx
                p_ymin = y * ddx
                p_zmin = (z - 0.5) * ddx
                p_xmax = (x + 0.5) * ddx
                p_ymax = (y + 1.0) * ddx
                p_zmax = (z + 0.5) * ddx
                p_corners = [
                    (p_xmin, p_ymin, p_zmin),
                    (p_xmin, p_ymax, p_zmin),
                    (p_xmax, p_ymin, p_zmin),
                    (p_xmax, p_ymax, p_zmin),
                    (p_xmin, p_ymin, p_zmax),
                    (p_xmin, p_ymax, p_zmax),
                    (p_xmax, p_ymin, p_zmax),
                    (p_xmax, p_ymax, p_zmax),
                ]

                corner_in_box = np.full((8), False)
                i = 0
                for corner in p_corners:
                    dist = np.sqrt(
                        (corner[0] - sphere.x) ** 2
                        + (corner[1] - sphere.y) ** 2
                        + (corner[2] - sphere.z) ** 2
                    )
                    if dist <= sphere.R + 0.0001 * nm:
                        corner_in_box[i] = True
                    i += 1

                if all(corner_in_box):
                    ga.y[x, y, z] = 1.0 / eps_in
                    l1.y[x, y, z] = (2.0 - wl**2 * dt**2) / (1.0 + gamma_l * dt / 2.0)
                    l2.y[x, y, z] = (gamma_l * dt / 2.0 - 1.0) / (
                        1.0 + gamma_l * dt / 2.0
                    )
                    l3.y[x, y, z] = (
                        delta_eps * wl**2 * dt**2 / (1.0 + gamma_l * dt / 2.0)
                    )
                elif not any(corner_in_box):
                    ga.y[x, y, z] = 1.0 / eps_out
                else:
                    nin = 0
                    nout = 0
                    xi = 0
                    yj = 0
                    zk = 0
                    for xi in range(0, nsub):
                        for yj in range(0, nsub):
                            for zk in range(0, nsub):
                                xsub = p_xmin + (xi + 0.5) / nsub / ddx
                                ysub = p_ymin + (yj + 0.5) / nsub / ddx
                                zsub = p_zmin + (zk + 0.5) / nsub / ddx
                                distsub = np.sqrt(
                                    (xsub - sphere.x) ** 2
                                    + (ysub - sphere.y) ** 2
                                    + (zsub - sphere.z) ** 2
                                )
                                if distsub <= sphere.R + 0.0001 * nm:
                                    nin += 1
                                else:
                                    nout += 1
                    ga.y[x, y, z] = (nout + nin) / (eps_in * nin + eps_out * nout)
                    l1.y[x, y, z] = (
                        (2.0 - wl**2 * dt**2)
                        / (1.0 + gamma_l * dt / 2.0)
                        * nin
                        / (nout + nin)
                    )  # + 0.5*nout/(nout+nin)
                    l2.y[x, y, z] = (
                        (gamma_l * dt / 2.0 - 1.0)
                        / (1.0 + gamma_l * dt / 2.0)
                        * nin
                        / (nout + nin)
                    )  # + 0.5*nout/(nout+nin)
                    l3.y[x, y, z] = (
                        delta_eps
                        * wl**2
                        * dt**2
                        / (1.0 + gamma_l * dt / 2.0)
                        * nin
                        / (nout + nin)
                    )

    # gaz
    for x in range(
        int((sphere.x - sphere.R) / ddx) - 2, int((sphere.x + sphere.R) / ddx) + 2
    ):
        for y in range(
            int((sphere.y - sphere.R) / ddx) - 2, int((sphere.y + sphere.R) / ddx) + 2
        ):
            for z in range(
                int((sphere.z - sphere.R) / ddx) - 2,
                int((sphere.z + sphere.R) / ddx) + 2,
            ):
                p_xmin = (x - 0.5) * ddx
                p_ymin = (y - 0.5) * ddx
                p_zmin = z * ddx
                p_xmax = (x + 0.5) * ddx
                p_ymax = (y + 0.5) * ddx
                p_zmax = (z + 1.0) * ddx
                p_corners = [
                    (p_xmin, p_ymin, p_zmin),
                    (p_xmin, p_ymax, p_zmin),
                    (p_xmax, p_ymin, p_zmin),
                    (p_xmax, p_ymax, p_zmin),
                    (p_xmin, p_ymin, p_zmax),
                    (p_xmin, p_ymax, p_zmax),
                    (p_xmax, p_ymin, p_zmax),
                    (p_xmax, p_ymax, p_zmax),
                ]

                corner_in_box = np.full((8), False)
                i = 0
                for corner in p_corners:
                    dist = np.sqrt(
                        (corner[0] - sphere.x) ** 2
                        + (corner[1] - sphere.y) ** 2
                        + (corner[2] - sphere.z) ** 2
                    )
                    if dist <= sphere.R + 0.0001 * nm:
                        corner_in_box[i] = True
                    i += 1

                if all(corner_in_box):
                    ga.z[x, y, z] = 1.0 / eps_in
                    l1.y[x, y, z] = (2.0 - wl**2 * dt**2) / (1.0 + gamma_l * dt / 2.0)
                    l2.y[x, y, z] = (gamma_l * dt / 2.0 - 1.0) / (
                        1.0 + gamma_l * dt / 2.0
                    )
                    l3.y[x, y, z] = (
                        delta_eps * wl**2 * dt**2 / (1.0 + gamma_l * dt / 2.0)
                    )
                elif not any(corner_in_box):
                    ga.z[x, y, z] = 1.0 / eps_out
                else:
                    nin = 0
                    nout = 0
                    xi = 0
                    yj = 0
                    zk = 0
                    for xi in range(0, nsub):
                        for yj in range(0, nsub):
                            for zk in range(0, nsub):
                                xsub = p_xmin + (xi + 0.5) / nsub / ddx
                                ysub = p_ymin + (yj + 0.5) / nsub / ddx
                                zsub = p_zmin + (zk + 0.5) / nsub / ddx
                                distsub = np.sqrt(
                                    (xsub - sphere.x) ** 2
                                    + (ysub - sphere.y) ** 2
                                    + (zsub - sphere.z) ** 2
                                )
                                if distsub <= sphere.R + 0.0001 * nm:
                                    nin += 1
                                else:
                                    nout += 1
                    ga.z[x, y, z] = (nout + nin) / (eps_in * nin + eps_out * nout)
                    l1.z[x, y, z] = (
                        (2.0 - wl**2 * dt**2)
                        / (1.0 + gamma_l * dt / 2.0)
                        * nin
                        / (nout + nin)
                    )  # + 0.5*nout/(nout+nin)
                    l2.z[x, y, z] = (
                        (gamma_l * dt / 2.0 - 1.0)
                        / (1.0 + gamma_l * dt / 2.0)
                        * nin
                        / (nout + nin)
                    )  # + 0.5*nout/(nout+nin)
                    l3.z[x, y, z] = (
                        delta_eps
                        * wl**2
                        * dt**2
                        / (1.0 + gamma_l * dt / 2.0)
                        * nin
                        / (nout + nin)
                    )
    return ga, l1, l2, l3


# def create_sphere_wo_sg(sphere):

#     #gax
#     for x in range (int((sphere.x-sphere.R)/ddx)-2,int((sphere.x+sphere.R)/ddx)+2):
#         for y in range (int((sphere.y-sphere.R)/ddx)-2,int((sphere.y+sphere.R)/ddx)+2):
#             for z in range (int((sphere.z-sphere.R)/ddx)-2,int((sphere.z+sphere.R)/ddx)+2):
#                 if(EPSILON_FLAG==1):
#                     #dielectric part
#                     loc_x = (x+0.5)*ddx
#                     loc_y = y*ddx
#                     loc_z = z*ddx

#                     dist = np.sqrt((loc_x-sphere.x)**2+(loc_y-sphere.y)**2+(loc_z-sphere.z)**2)
#                     if (dist<=sphere.R):
#                         ga.x[x,y,z]= 1./eps_in
#                         d1.x[x,y,z] = 4./(2.+gamma*dt)
#                         d2.x[x,y,z] = (gamma*dt-2.)/(gamma*dt+2.)
#                         d3.x[x,y,z] = 2.*wp**2*dt**2/(2.+gamma*dt)

#                     else:
#                         ga.x[x,y,z]= 1./eps_out


#     #gay
#     for x in range (int((sphere.x-sphere.R)/ddx)-2,int((sphere.x+sphere.R)/ddx)+2):
#         for y in range (int((sphere.y-sphere.R)/ddx)-2,int((sphere.y+sphere.R)/ddx)+2):
#             for z in range (int((sphere.z-sphere.R)/ddx)-2,int((sphere.z+sphere.R)/ddx)+2):
#                 if(EPSILON_FLAG==1):
#                     #dielectric part
#                     loc_x = x*ddx
#                     loc_y = (y+0.5)*ddx
#                     loc_z = z*ddx

#                     dist = np.sqrt((loc_x-sphere.x)**2+(loc_y-sphere.y)**2+(loc_z-sphere.z)**2)
#                     if (dist<=sphere.R):
#                         ga.y[x,y,z]= 1./eps_in
#                         d1.y[x,y,z] = 4./(2.+gamma*dt)
#                         d2.y[x,y,z] = (gamma*dt-2.)/(gamma*dt+2.)
#                         d3.y[x,y,z] = 2.*wp**2*dt**2/(2.+gamma*dt)

#                     else:
#                         ga.y[x,y,z]= 1./eps_out

#     #gaz
#     for x in range (int((sphere.x-sphere.R)/ddx)-2,int((sphere.x+sphere.R)/ddx)+2):
#         for y in range (int((sphere.y-sphere.R)/ddx)-2,int((sphere.y+sphere.R)/ddx)+2):
#             for z in range (int((sphere.z-sphere.R)/ddx)-2,int((sphere.z+sphere.R)/ddx)+2):
#                 if(EPSILON_FLAG==1):
#                     #dielectric part
#                     loc_x = x*ddx
#                     loc_y = y*ddx
#                     loc_z = (z+0.5)*ddx

#                     dist = np.sqrt((loc_x-sphere.x)**2+(loc_y-sphere.y)**2+(loc_z-sphere.z)**2)
#                     if (dist<=sphere.R):
#                         ga.z[x,y,z]= 1./eps_in
#                         d1.z[x,y,z] = 4./(2.+gamma*dt)
#                         d2.z[x,y,z] = (gamma*dt-2.)/(gamma*dt+2.)
#                         d3.z[x,y,z] = 2.*wp**2*dt**2/(2.+gamma*dt)

#                     else:
#                         ga.z[x,y,z]= 1./eps_out


# def create_sphere(ga,ddx,dt,sphere,nsub,EPSILON_FLAG,DRUDE_FLAG):

#     numin = 0
#     numout = 0
#     numhalf = 0
#     count = 0
#     #gax
#     for x in range (int((sphere.x-sphere.R)/ddx)-2,int((sphere.x+sphere.R)/ddx)+2):
#         for y in range (int((sphere.y-sphere.R)/ddx)-2,int((sphere.y+sphere.R)/ddx)+2):
#             for z in range (int((sphere.z-sphere.R)/ddx)-2,int((sphere.z+sphere.R)/ddx)+2):
#                 if(EPSILON_FLAG==1):
#                     #dielectric part
#                     p_xmin = x*ddx
#                     p_ymin = (y-0.5)*ddx
#                     p_zmin = (z-0.5)*ddx
#                     p_xmax = (x+1)*ddx
#                     p_ymax = (y+0.5)*ddx
#                     p_zmax = (z+0.5)*ddx
#                     p_corners = [
#                         (p_xmin, p_ymin,p_zmin),
#                         (p_xmin, p_ymax,p_zmin),
#                         (p_xmax, p_ymin,p_zmin),
#                         (p_xmax, p_ymax,p_zmin),
#                         (p_xmin, p_ymin,p_zmax),
#                         (p_xmin, p_ymax,p_zmax),
#                         (p_xmax, p_ymin,p_zmax),
#                         (p_xmax, p_ymax,p_zmax)]

#                     corner_in_box = np.full((8), False)
#                     i=0
#                     for corner in p_corners:
#                         dist = np.sqrt((corner[0]-sphere.x)**2+(corner[1]-sphere.y)**2+(corner[2]-sphere.z)**2)
#                         if (dist<=sphere.R):
#                             corner_in_box[i]= True
#                         i += 1

#                     if(all(corner_in_box)):
#                         ga.x[x,y,z]= 1./eps_in
#                         # d1x[x,y,z] = 4./(2.+gamma*dt)
#                         # d2x[x,y,z] = (gamma*dt-2.)/(gamma*dt+2.)
#                         # d3x[x,y,z] = 2.*wp**2*dt**2/(2.+gamma*dt)

#                     elif(not any(corner_in_box)):
#                         ga.x[x,y,z]= 1./eps_out
#                         numout +=1
#                     else:
#                         numhalf +=1
#                         nin=0
#                         nout =0
#                         xi = 0
#                         yj = 0
#                         zk = 0
#                         for xi in range (0,nsub):
#                             for yj in range (0,nsub):
#                                 for zk in range (0,nsub):
#                                     xsub = p_xmin + (xi+0.5)/nsub/ddx
#                                     ysub = p_ymin + (yj+0.5)/nsub/ddx
#                                     zsub = p_zmin + (zk+0.5)/nsub/ddx
#                                     distsub = np.sqrt((xsub-sphere.x)**2+(ysub-sphere.y)**2+(zsub-sphere.z)**2)
#                                     if(distsub<=sphere.R):
#                                         nin += 1
#                                     else:
#                                         nout += 1
#                         ga.x[x,y,z]= (nout+nin)/(eps_in*nin+eps_out*nout)
#                         # d1x[x,y,z] = 4./(2.+gamma*dt)*nin/(nout+nin)# + 0.5*nout/(nout+nin)
#                         # d2x[x,y,z] = (gamma*dt-2.)/(gamma*dt+2.)*nin/(nout+nin)#+ 0.5*nout/(nout+nin)
#                         # d3x[x,y,z] = 2.*wp**2*dt**2/(2.+gamma*dt)*nin/(nout+nin)

#                     #Drude part
#                 if(DRUDE_FLAG==1):
#                     distx = np.sqrt(((x+.5)*ddx-sphere.x)**2+(y*ddx-sphere.y)**2+(z*ddx-sphere.z)**2)
#                     if(distx <= sphere.R):
#                         count +=1
#                         d1.x[x,y,z] = 4./(2.+gamma*dt)
#                         d2.x[x,y,z] = (gamma*dt-2.)/(gamma*dt+2.)
#                         d3.x[x,y,z] = 2.*wp**2*dt**2/(2.+gamma*dt)


#     #gay
#     for x in range (int((sphere.x-sphere.R)/ddx)-2,int((sphere.x+sphere.R)/ddx)+2):
#         for y in range (int((sphere.y-sphere.R)/ddx)-2,int((sphere.y+sphere.R)/ddx)+2):
#             for z in range (int((sphere.z-sphere.R)/ddx)-2,int((sphere.z+sphere.R)/ddx)+2):
#                 if(EPSILON_FLAG==1):
#                     p_xmin = (x-0.5)*ddx
#                     p_ymin = y*ddx
#                     p_zmin = (z-0.5)*ddx
#                     p_xmax = (x+0.5)*ddx
#                     p_ymax = (y+1.)*ddx
#                     p_zmax = (z+0.5)*ddx
#                     p_corners = [
#                         (p_xmin, p_ymin,p_zmin),
#                         (p_xmin, p_ymax,p_zmin),
#                         (p_xmax, p_ymin,p_zmin),
#                         (p_xmax, p_ymax,p_zmin),
#                         (p_xmin, p_ymin,p_zmax),
#                         (p_xmin, p_ymax,p_zmax),
#                         (p_xmax, p_ymin,p_zmax),
#                         (p_xmax, p_ymax,p_zmax)]

#                     corner_in_box = np.full((8), False)
#                     i=0
#                     for corner in p_corners:
#                         dist = np.sqrt((corner[0]-sphere.x)**2+(corner[1]-sphere.y)**2+(corner[2]-sphere.z)**2)
#                         if (dist<=sphere.R):
#                             corner_in_box[i]= True
#                         i += 1

#                     if(all(corner_in_box)):
#                         ga.y[x,y,z]=1./eps_in
#                         # d1y[x,y,z] = 4./(2.+gamma*dt)
#                         # d2y[x,y,z] = (gamma*dt-2.)/(gamma*dt+2.)
#                         # d3y[x,y,z] = 2.*wp**2*dt**2/(2.+gamma*dt)
#                     elif(not any(corner_in_box)):
#                         ga.y[x,y,z]= 1./eps_out
#                     else:
#                         nin=0
#                         nout =0
#                         xi = 0
#                         yj = 0
#                         zk = 0
#                         for xi in range (0,nsub):
#                             for yj in range (0,nsub):
#                                 for zk in range (0,nsub):
#                                     xsub = p_xmin + (xi+0.5)/nsub/ddx
#                                     ysub = p_ymin + (yj+0.5)/nsub/ddx
#                                     zsub = p_zmin + (zk+0.5)/nsub/ddx
#                                     distsub = np.sqrt((xsub-sphere.x)**2+(ysub-sphere.y)**2+(zsub-sphere.z)**2)
#                                     if(distsub<=sphere.sphere.R):
#                                         nin += 1
#                                     else:
#                                         nout += 1
#                         ga.y[x,y,z]= (nout+nin)/(eps_in*nin+eps_out*nout)
#                         # d1y[x,y,z] = 4./(2.+gamma*dt)*nin/(nout+nin)#+ 0.5*nout/(nout+nin)
#                         # d2y[x,y,z] = (gamma*dt-2.)/(gamma*dt+2.)*nin/(nout+nin)#+ 0.5*nout/(nout+nin)
#                         # d3y[x,y,z] = 2.*wp**2*dt**2/(2.+gamma*dt)*nin/(nout+nin)

#                     #Drude part
#                 if(DRUDE_FLAG==1):
#                     disty = np.sqrt((x*ddx-sphere.x)**2+((y+.5)*ddx-sphere.y)**2+(z*ddx-sphere.z)**2)
#                     if(disty <= sphere.R):
#                         d1.y[x,y,z] = 4./(2.+gamma*dt)
#                         d2.y[x,y,z] = (gamma*dt-2.)/(gamma*dt+2.)
#                         d3.y[x,y,z] = 2.*wp**2*dt**2/(2.+gamma*dt)

#     #gaz
#     for x in range (int((sphere.x-sphere.R)/ddx)-2,int((sphere.x+sphere.R)/ddx)+2):
#         for y in range (int((sphere.y-sphere.R)/ddx)-2,int((sphere.y+sphere.R)/ddx)+2):
#             for z in range (int((sphere.z-sphere.R)/ddx)-2,int((sphere.z+sphere.R)/ddx)+2):
#                 if(EPSILON_FLAG==1):
#                     p_xmin = (x-0.5)*ddx
#                     p_ymin = (y-0.5)*ddx
#                     p_zmin = z*ddx
#                     p_xmax = (x+0.5)*ddx
#                     p_ymax = (y+0.5)*ddx
#                     p_zmax = (z+1.)*ddx
#                     p_corners = [
#                         (p_xmin, p_ymin,p_zmin),
#                         (p_xmin, p_ymax,p_zmin),
#                         (p_xmax, p_ymin,p_zmin),
#                         (p_xmax, p_ymax,p_zmin),
#                         (p_xmin, p_ymin,p_zmax),
#                         (p_xmin, p_ymax,p_zmax),
#                         (p_xmax, p_ymin,p_zmax),
#                         (p_xmax, p_ymax,p_zmax)]

#                     corner_in_box = np.full((8), False)
#                     i=0
#                     for corner in p_corners:
#                         dist = np.sqrt((corner[0]-sphere.x)**2+(corner[1]-sphere.y)**2+(corner[2]-sphere.z)**2)
#                         if (dist<=sphere.R):
#                             corner_in_box[i]= True
#                         i += 1

#                     if(all(corner_in_box)):
#                         ga.z[x,y,z]=1./eps_in
#                         # d1z[x,y,z] = 4./(2.+gamma*dt)
#                         # d2z[x,y,z] = (gamma*dt-2.)/(gamma*dt+2.)
#                         # d3z[x,y,z] = 2.*wp**2*dt**2/(2.+gamma*dt)
#                     elif(not any(corner_in_box)):
#                         ga.z[x,y,z]= 1./eps_out
#                     else:
#                         nin=0
#                         nout =0
#                         xi = 0
#                         yj = 0
#                         zk = 0
#                         for xi in range (0,nsub):
#                             for yj in range (0,nsub):
#                                 for zk in range (0,nsub):
#                                     xsub = p_xmin + (xi+0.5)/nsub/ddx
#                                     ysub = p_ymin + (yj+0.5)/nsub/ddx
#                                     zsub = p_zmin + (zk+0.5)/nsub/ddx
#                                     distsub = np.sqrt((xsub-sphere.x)**2+(ysub-sphere.y)**2+(zsub-sphere.z)**2)
#                                     if(distsub<=sphere.R):
#                                         nin += 1
#                                     else:
#                                         nout += 1
#                         ga.z[x,y,z]= (nout+nin)/(eps_in*nin+eps_out*nout)
#                         # d1z[x,y,z] = 4./(2.+gamma*dt)*nin/(nout+nin)#+ 0.5*nout/(nout+nin)
#                         # d2z[x,y,z] = (gamma*dt-2.)/(gamma*dt+2.)*nin/(nout+nin)#+ 0.5*nout/(nout+nin)
#                         # d3z[x,y,z] = 2.*wp**2*dt**2/(2.+gamma*dt)*nin/(nout+nin)

#                 #Drude part
#                 if(DRUDE_FLAG==1):
#                     distz = np.sqrt((x*ddx-sphere.x)**2+(y*ddx-sphere.y)**2+((z+.5)*ddx-sphere.z)**2)
#                     if(distz <= sphere.R):
#                         d1.z[x,y,z] = 4./(2.+gamma*dt)
#                         d2.z[x,y,z] = (gamma*dt-2.)/(gamma*dt+2.)
#                         d3.z[x,y,z] = 2.*wp**2*dt**2/(2.+gamma*dt)
