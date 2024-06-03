import numpy as np
from modules.fundamentals import *
import matplotlib.pyplot as plt
from modules.parameters import *

# import grid


def setup(
    args,
    fig,
    ax,
    ddx,
    dt,
    length,
    array,
    dims,
    sphere,
    pulse,
    npml,
    tfsf,
    e,
    ez_inc,
    hx_inc,
):
    "2d field plots"
    ims = []
    x, z = np.meshgrid(array.x, array.z)
    ims.append(ax[0, 1].imshow(np.zeros((dims.x, dims.y))))
    ims.append(
        ax[1, 1].imshow(
            np.zeros((dims.x, 1)),
            cmap="viridis",
            interpolation="quadric",
            origin="lower",
        )
    )
    ims.append(
        ax[2, 1].imshow(
            np.zeros((1, 1)), cmap="viridis", interpolation="quadric", origin="lower"
        )
    )

    # Labels
    ax[0, 1].set_xlabel("$x$ [nm]")
    ax[0, 1].set_ylabel("$y$ [nm]")

    ax[1, 1].set_xlabel("$x$ [nm]")
    ax[1, 1].set_ylabel("$z$ [nm]")

    ax[2, 1].set_xlabel("$y$ [nm]")
    ax[2, 1].set_ylabel("$z$ [nm]")

    # 2d fields
    for im in ims:
        im.set_clim(vmin=0, vmax=10**5)
    ims[0].set_extent((0, length.x / nm, 0, length.x / nm))
    ims[1].set_extent((0, length.x / nm, 0, length.x / nm))
    ims[2].set_extent((0, length.x / nm, 0, length.x / nm))
    cbaxes = fig.add_axes([0.35, 0.95, 0.12, 0.01])
    cbar = plt.colorbar(ims[0], cax=cbaxes, orientation="horizontal")
    cbar.ax.set_title("Field [arb. units]")

    ax[0, 0].set_axis_off()
    field_component = "Ez"

    "Information"
    text_tstep = ax[0, 0].annotate(
        "Time Step: 0",
        (0.5, 1),
        xycoords="axes fraction",
        va="center",
        ha="center",
        weight="bold",
    )

    # source information
    match args.source:
        case "TFSF":
            plot_text = "\n" + "TFSF source"
        case "Point":
            plot_text = "\n" + "Point source"
        case "None":
            plot_text = "\n" + "No source"

    # boundary information
    match args.boundary:
        case "PML":
            plot_text = "\n" + "PML active"
        case "PBC":
            plot_text = "\n" + "PBC active"
        case "None":
            plot_text = "\n" + "No boundaries active"

    # object information
    match args.object:
        case "None":
            plot_text += "\n No object implemented"
        case "Sphere":
            plot_text += (
                "\n Sphere of radius " + str(int(sphere.R / nm)) + "nm implemented"
            )
        case "Rectangle":
            plot_text += (
                "\n Rectangle of thickness" + str(int(sphere.R / nm)) + "nm implemented"
            )

    # material information
    match args.material:
        case "Drude":
            plot_text += "\n Drude model implemented"
        case "DrudeLorentz":
            plot_text += "\n DrudeLorentz model implemented"
        case "Etchegoin":
            plot_text += "\n Etchegoin model implemented"

    # pulse information
    plot_text += (
        "\n\n Pulse information\n Amplitude: "
        + str(pulse.amplitude)
        + "\n center energy: {} eV".format(np.round(pulse.energy, 2))
        + "\n wavelength: {:06.2f} nm".format(pulse.lam_0 / nm)
        + "\n dx: {}".format(ddx * 1e9)
        + " nm"
        + "\n dt: {}".format(np.round(dt * 1e18, 2))
        + " as"
    )

    # FDTD information
    plot_text += (
        "\n\n Grid Dimensions: \nx: "
        + str(dims.x)
        + ", y: "
        + str(dims.y)
        + ", z: "
        + str(dims.z)
    )
    # plot_text += '\nScatter Field: ' + str(scat_field)
    if args.boundary == "PML" or args.boundary == "PBC":
        plot_text += ", PML: " + str(npml)
    else:
        plot_text += ", PML: Off"

    ax[0, 0].annotate(
        plot_text,
        (0, 0.9),
        xycoords="axes fraction",
        va="top",
        ha="left",
    )

    ax[0, 2].set_ylabel("Field [arb. units]")
    ax[0, 2].set_xlabel("Grid Cells ($x$)")
    ax[0, 2].set_ylabel("Ez field")
    ax[0, 2].set_title("X profile")

    ax[1, 2].set_ylabel("Field [arb. units]")
    ax[1, 2].set_xlabel("Grid Cells ($y$)")
    ax[1, 2].set_ylabel("Ez field")
    ax[1, 2].set_title("Y profile")

    ax[2, 2].set_ylabel("Field [arb. units]")
    ax[2, 2].set_xlabel("Grid Cells ($z$)")
    ax[2, 2].set_ylabel("Ez field")
    ax[2, 2].set_title("Z profile")

    (xcut,) = ax[0, 2].plot(
        array.x / nm, np.abs(e.z[:, int(dims.y / 2), int(dims.z / 2)]), label="X cut"
    )
    (ycut,) = ax[1, 2].plot(
        array.x / nm, np.abs(e.z[int(dims.x / 2), :, int(dims.z / 2)]), label="Y cut"
    )
    (zcut,) = ax[2, 2].plot(
        array.x / nm, np.abs(e.z[int(dims.x / 2), int(dims.y / 2), :]), label="Z cut"
    )

    # incident field
    (incident_e,) = ax[1, 0].plot(array.y / nm, ez_inc, label="Ez_inc")
    (incident_h,) = ax[1, 0].plot(array.y / nm, hx_inc, label="Hx_inc")
    ax[1, 0].set_xlabel("Grid Cells ($y$)")
    ax[1, 0].set_ylabel("Fields")
    ax[1, 0].set_title("Incident fields")
    ax[1, 0].legend()

    # polarization

    imp = []
    imp.append(ax[0, 3].imshow(np.zeros((dims.x, dims.y))))

    imp.append(
        ax[1, 3].imshow(
            np.zeros((dims.x, 1)),
            cmap="viridis",
            interpolation="quadric",
            origin="lower",
        )
    )

    imp.append(
        ax[2, 3].imshow(
            np.zeros((1, 1)), cmap="viridis", interpolation="quadric", origin="lower"
        )
    )

    # Labels
    ax[0, 3].set_xlabel("Grid Cells ($x$)")
    ax[0, 3].set_ylabel("Grid Cells ($y$)")

    ax[1, 3].set_xlabel("Grid Cells ($x$)")
    ax[1, 3].set_ylabel("Grid Cells ($z$)")

    ax[2, 3].set_xlabel("Grid Cells ($y$)")
    ax[2, 3].set_ylabel("Grid Cells ($z$)")

    for im in ims:
        im.set_clim(vmin=0, vmax=10**5)
    imp[0].set_extent((0, dims.x, 0, dims.x))
    imp[1].set_extent((0, dims.x, 0, dims.x))
    imp[2].set_extent((0, dims.x, 0, dims.x))
    cbaxes_p = fig.add_axes([0.75, 0.95, 0.12, 0.01])
    cbar_p = plt.colorbar(imp[0], cax=cbaxes_p, orientation="horizontal")
    cbar_p.ax.set_title("Polarization [arb. units]")

    # 'add lines and circles'

    fx = plt.Circle(
        (sphere.x / nm, sphere.y / nm), sphere.R / nm, color="r", fill=False
    )
    fy = plt.Circle(
        (sphere.x / nm, sphere.y / nm), sphere.R / nm, color="r", fill=False
    )
    fz = plt.Circle(
        (sphere.x / nm, sphere.y / nm), sphere.R / nm, color="r", fill=False
    )
    px = plt.Circle(
        (sphere.x / nm, sphere.y / nm), sphere.R / nm, color="r", fill=False
    )
    py = plt.Circle(
        (sphere.x / nm, sphere.y / nm), sphere.R / nm, color="r", fill=False
    )
    pz = plt.Circle(
        (sphere.x / nm, sphere.y / nm), sphere.R / nm, color="r", fill=False
    )

    # fields
    ax[0, 1].set_aspect(1)
    ax[1, 1].set_aspect(1)
    ax[2, 1].set_aspect(1)

    ax[0, 1].add_artist(fx)
    ax[1, 1].add_artist(fy)
    ax[2, 1].add_artist(fz)

    # polarization
    ax[0, 3].set_aspect(1)
    ax[1, 3].set_aspect(1)
    ax[2, 3].set_aspect(1)

    ax[0, 3].add_artist(px)
    ax[1, 3].add_artist(py)
    ax[2, 3].add_artist(pz)

    # PML layers
    ax[0, 1].hlines(
        npml * ddx / nm, npml * ddx / nm, (dims.x - npml - 1) * ddx / nm, "b"
    )
    ax[0, 1].hlines(
        (dims.x - npml - 1) * ddx / nm,
        npml * ddx / nm,
        (dims.x - npml - 1) * ddx / nm,
        "b",
    )
    ax[0, 1].vlines(
        npml * ddx / nm, npml * ddx / nm, (dims.x - npml - 1) * ddx / nm, "b"
    )
    ax[0, 1].vlines(
        (dims.x - npml - 1) * ddx / nm,
        npml * ddx / nm,
        (dims.x - npml - 1) * ddx / nm,
        "b",
    )
    ax[1, 1].hlines(
        npml * ddx / nm, npml * ddx / nm, (dims.x - npml - 1) * ddx / nm, "b"
    )
    ax[1, 1].hlines(
        (dims.x - npml - 1) * ddx / nm,
        npml * ddx / nm,
        (dims.x - npml - 1) * ddx / nm,
        "b",
    )
    ax[1, 1].vlines(
        npml * ddx / nm, npml * ddx / nm, (dims.x - npml - 1) * ddx / nm, "b"
    )
    ax[1, 1].vlines(
        (dims.x - npml - 1) * ddx / nm,
        npml * ddx / nm,
        (dims.x - npml - 1) * ddx / nm,
        "b",
    )
    ax[2, 1].hlines(
        npml * ddx / nm, npml * ddx / nm, (dims.x - npml - 1) * ddx / nm, "b"
    )
    ax[2, 1].hlines(
        (dims.x - npml - 1) * ddx / nm,
        npml * ddx / nm,
        (dims.x - npml - 1) * ddx / nm,
        "b",
    )
    ax[2, 1].vlines(
        npml * ddx / nm, npml * ddx / nm, (dims.x - npml - 1) * ddx / nm, "b"
    )
    ax[2, 1].vlines(
        (dims.x - npml - 1) * ddx / nm,
        npml * ddx / nm,
        (dims.x - npml - 1) * ddx / nm,
        "b",
    )

    # TFSF
    ax[0, 1].hlines(
        tfsf.x_min * ddx / nm, tfsf.z_min * ddx / nm, (tfsf.z_max + 1) * ddx / nm, "r"
    )
    ax[0, 1].hlines(
        (tfsf.x_max + 1) * ddx / nm,
        tfsf.z_min * ddx / nm,
        (tfsf.z_max + 1) * ddx / nm,
        "r",
    )
    ax[0, 1].vlines(
        tfsf.z_min * ddx / nm, tfsf.x_min * ddx / nm, (tfsf.x_max + 1) * ddx / nm, "r"
    )
    ax[0, 1].vlines(
        (tfsf.z_max + 1) * ddx / nm,
        tfsf.x_min * ddx / nm,
        (tfsf.x_max + 1) * ddx / nm,
        "r",
    )
    ax[1, 1].hlines(
        tfsf.x_min * ddx / nm, tfsf.z_min * ddx / nm, (tfsf.z_max + 1) * ddx / nm, "r"
    )
    ax[1, 1].hlines(
        (tfsf.x_max + 1) * ddx / nm,
        tfsf.z_min * ddx / nm,
        (tfsf.z_max + 1) * ddx / nm,
        "r",
    )
    ax[1, 1].vlines(
        tfsf.z_min * ddx / nm, tfsf.x_min * ddx / nm, (tfsf.x_max + 1) * ddx / nm, "r"
    )
    ax[1, 1].vlines(
        (tfsf.z_max + 1) * ddx / nm,
        tfsf.x_min * ddx / nm,
        (tfsf.x_max + 1) * ddx / nm,
        "r",
    )
    ax[2, 1].hlines(
        tfsf.x_min * ddx / nm, tfsf.z_min * ddx / nm, (tfsf.z_max + 1) * ddx / nm, "r"
    )
    ax[2, 1].hlines(
        (tfsf.x_max + 1) * ddx / nm,
        tfsf.z_min * ddx / nm,
        (tfsf.z_max + 1) * ddx / nm,
        "r",
    )
    ax[2, 1].vlines(
        tfsf.z_min * ddx / nm, tfsf.x_min * ddx / nm, (tfsf.x_max + 1) * ddx / nm, "r"
    )
    ax[2, 1].vlines(
        (tfsf.z_max + 1) * ddx / nm,
        tfsf.x_min * ddx / nm,
        (tfsf.x_max + 1) * ddx / nm,
        "r",
    )

    return ims, imp, text_tstep, xcut, ycut, zcut, incident_e, incident_h


def animate(
    t,
    text_tstep,
    e,
    dims,
    ims,
    array,
    ax,
    xcut,
    ycut,
    zcut,
    incident_e,
    ez_inc,
    hx_inc,
    incident_h,
    p,
    imp,
    time_pause,
):
    text_tstep.set_text("Time Step: " + str(t))
    max = np.max(np.abs(e.z[:, :, int(dims.y / 2)]))
    # print('max',max)
    for im in ims:
        im.set_clim(vmin=0, vmax=max)
    x, y = np.meshgrid(array.x, array.y)
    ims[0].set_data(np.transpose(np.abs(e.z[:, :, int(dims.y / 2)])))
    ims[1].set_data(np.transpose(np.abs(e.z[:, int(dims.y / 2), :])))
    ims[2].set_data(np.transpose(np.abs(e.z[int(dims.y / 2), :, :])))

    # 1d plots
    ax[0, 2].set_ylim(-max, max)
    xcut.set_data(array.x / nm, np.abs(e.z[:, int(dims.y / 2), int(dims.z / 2)]))

    ax[1, 2].set_ylim(-max, max)
    ycut.set_data(array.y / nm, np.abs(e.z[int(dims.x / 2), :, int(dims.z / 2)]))

    ax[2, 2].set_ylim(-max, max)
    zcut.set_data(array.z / nm, np.abs(e.z[int(dims.x / 2), int(dims.y / 2), :]))

    # incident field
    ax[1, 0].set_ylim(-max, max)
    incident_e.set_data(array.y / nm, ez_inc)
    incident_h.set_data(array.y / nm, hx_inc)

    max_p = np.max(np.abs(p.z[:, :, int(dims.y / 2)]))
    for im in imp:
        im.set_clim(vmin=0, vmax=max_p)
    imp[0].set_data(np.transpose(np.abs(p.z[:, :, int(dims.z / 2)])))
    imp[1].set_data(np.transpose(np.abs(p.z[:, int(dims.y / 2), :])))
    imp[2].set_data(np.transpose(np.abs(p.z[int(dims.x / 2), :, :])))

    # main graph is E(z,y, time snapshops), and a small graph of E(t) as center
    # plt.clf() # close each time for new update graph/colormap
    # fig,ax = plt.subplots(3, 4, figsize=(10, 6))

    # 2d plot - several options, two examples below
    #    img = ax.imshow(Ez)
    #     x,y =np.meshgrid(X,Y)
    #     #img = ax.contourf(x,y,np.transpose(ez[int(Ymax/2),:,:]*10))
    #     #ax[0,1].contourf(x/nm,y/nm,np.transpose(np.abs(e.z[:,:,int(dims.y/2)])))
    #     axs = ax[0, 1]
    #     img = axs.contourf(x/nm,y/nm,np.transpose(np.abs(e.z[:,:,int(dims.y/2)])))
    # #    img = ax.contourf(x,y,np.transpose(np.round(ez[:,:,int(Ymax/2)],10)))
    #     cbar=plt.colorbar(img, ax=axs)
    #     cbar.set_label('$Ez$ (arb. units)')
    # ax = axs[0,1]
    # pcm = ax.pcolormesh(np.random.random((20, 20)))
    # fig.colorbar(pcm, ax=ax, shrink=0.6)

    # # add labels to axes
    #     axs.set_xlabel('Grid Cells ($x$)')
    #     axs.set_ylabel('Grid Cells ($y$)')

    # cc = plt.Circle((xc/nm, yc/nm), R/nm, color='r',fill=False)
    # ax[0,1].set_aspect( 1 )
    # ax[0,1].add_artist( cc )

    # ax[0,1].hlines(npml*ddx/nm,npml*ddx/nm,(dims.x-npml-1)*ddx/nm, 'b')
    # ax[0,1].hlines((dims.x-npml-1)*ddx/nm,npml*ddx/nm,(dims.x-npml-1)*ddx/nm, 'b')
    # ax[0,1].vlines(npml*ddx/nm,npml*ddx/nm,(dims.x-npml-1)*ddx/nm, 'b')
    # ax[0,1].vlines((dims.x-npml-1)*ddx/nm,npml*ddx/nm,(dims.x-npml-1)*ddx/nm, 'b')

    # #TFSF
    # ax[0,1].hlines(tfsf.y_min*ddx/nm,tfsf.x_min*ddx/nm,tfsf.x_max*ddx/nm, 'r')
    # ax[0,1].hlines(tfsf.y_max*ddx/nm,tfsf.x_min*ddx/nm,tfsf.x_max*ddx/nm, 'r')
    # ax[0,1].vlines(tfsf.x_min*ddx/nm,tfsf.y_min*ddx/nm,tfsf.y_max*ddx/nm, 'r')
    # ax[0,1].vlines(tfsf.x_max*ddx/nm,tfsf.y_min*ddx/nm,tfsf.y_max*ddx/nm, 'r')

    # incident field
    # axs[1,0].plot(Y/nm,ez_inc,label='Ez_inc')
    # axs[1,0].plot(Y/nm,hx_inc,label='Hx_inc')
    # #ax2.set_ylim(-1.1,1.1)
    # axs[1,0].set_xlabel('Grid Cells ($y$)')
    # axs[1,0].set_ylabel('Fields')
    # axs[1,0].set_title('Incident fields')
    # axs[1,0].legend()
    """
    ax01 = fig.add_axes([.4, .35, .2, .4])   

# 2d plot - several options, two examples below
#    img = ax.imshow(Ez)
    z,y =np.meshgrid(X,Y)
    #img = ax.contourf(x,y,np.transpose(ez[int(Ymax/2),:,:]*10))
    img = ax01.contourf(y/nm,z/nm,np.abs(e.z[int(dims.x/2),:,:]))
    cbar=plt.colorbar(img, ax=ax01)
    cbar.set_label('$Ez$ (arb. units)')

# add labels to axes
    ax01.set_xlabel('Grid Cells ($y$)')
    ax01.set_ylabel('Grid Cells ($z$)')
    
    cc = plt.Circle((yc/nm, zc/nm), R/nm, color='r',fill=False)
    ax01.set_aspect( 1 ) 
    ax01.add_artist( cc ) 

    #PML layers
    ax01.hlines(npml*ddx/nm,npml*ddx/nm,(dims.x-npml-1)*ddx/nm, 'b')
    ax01.hlines((dims.x-npml-1)*ddx/nm,npml*ddx/nm,(dims.x-npml-1)*ddx/nm, 'b')
    ax01.vlines(npml*ddx/nm,npml*ddx/nm,(dims.x-npml-1)*ddx/nm, 'b')
    ax01.vlines((dims.x-npml-1)*ddx/nm,npml*ddx/nm,(dims.x-npml-1)*ddx/nm, 'b')

    #TFSF
    ax01.hlines(tfsf.x_min*ddx/nm,tfsf.z_min*ddx/nm,tfsf.z_max*ddx/nm, 'r')
    ax01.hlines(tfsf.x_max*ddx/nm,tfsf.z_min*ddx/nm,tfsf.z_max*ddx/nm, 'r')
    ax01.vlines(tfsf.z_min*ddx/nm,tfsf.x_min*ddx/nm,tfsf.x_max*ddx/nm, 'r')
    ax01.vlines(tfsf.z_max*ddx/nm,tfsf.x_min*ddx/nm,tfsf.x_max*ddx/nm, 'r')

    axmicro = fig.add_axes([.75, .75, .2, .2])   
    axmicro.plot(grid.k_grid(grid.n_kmax),f_plot,label='Fermi distribution')
    axmicro.set_xlabel('Grid Cells ($k$)')
    axmicro.set_ylabel('Fermi')
    axmicro.set_title('Fermi dist')
    axmicro.legend()


    ax02 = fig.add_axes([.75, .35, .2, .4])   

# 2d plot - several options, two examples below
#    img = ax.imshow(Ez)
    x,z =np.meshgrid(X,Z)
    #img = ax.contourf(x,y,np.transpose(ez[int(Ymax/2),:,:]*10))
    img = ax02.contourf(z/nm,x/nm,np.transpose(np.abs(e.z[:,int(dims.y/2),:])))
    cbar=plt.colorbar(img, ax=ax02)
    cbar.set_label('$Ez$ (arb. units)')

# add labels to axes
    ax02.set_xlabel('Grid Cells ($z$)')
    ax02.set_ylabel('Grid Cells ($x$)')

    cc = plt.Circle((zc/nm, xc/nm), R/nm, color='r',fill=False)
    ax02.set_aspect( 1 ) 
    ax02.add_artist( cc ) 

    ax02.hlines(npml*ddx/nm,npml*ddx/nm,(dims.x-npml-1)*ddx/nm, 'b')
    ax02.hlines((dims.x-npml-1)*ddx/nm,npml*ddx/nm,(dims.x-npml-1)*ddx/nm, 'b')
    ax02.vlines(npml*ddx/nm,npml*ddx/nm,(dims.x-npml-1)*ddx/nm, 'b')
    ax02.vlines((dims.x-npml-1)*ddx/nm,npml*ddx/nm,(dims.x-npml-1)*ddx/nm, 'b')

    #TFSF
    ax02.hlines(tfsf.x_min*ddx/nm,tfsf.z_min*ddx/nm,tfsf.z_max*ddx/nm, 'r')
    ax02.hlines(tfsf.x_max*ddx/nm,tfsf.z_min*ddx/nm,tfsf.z_max*ddx/nm, 'r')
    ax02.vlines(tfsf.z_min*ddx/nm,tfsf.x_min*ddx/nm,tfsf.x_max*ddx/nm, 'r')
    ax02.vlines(tfsf.z_max*ddx/nm,tfsf.x_min*ddx/nm,tfsf.x_max*ddx/nm, 'r')
# add title with current simulation time step
    ax.set_title("frame time {}".format(t))
    """
    # # incident field
    #     ax2 = fig.add_axes([.05, .1, .2, .2])
    #     ax2.plot(Z,ez_inc,label='Ez_inc')
    #     ax2.plot(Z,hx_inc,label='Hx_inc')
    #     #ax2.set_ylim(-1.1,1.1)
    #     ax2.set_xlabel('Grid Cells ($y$)')
    #     ax2.set_ylabel('Fields')
    #     ax2.set_title('Incident fields')
    #     ax2.legend()
    """
# plot calculated field shortly after source position
    axx = fig.add_axes([.05, .1, .2, .2])
    axx.plot(X/nm,np.abs(e.z[:,int(dims.y/2),int(dims.z/2)]),label='Ez_inc')
    #ax3.plot(Z,ez[:,ja,int(Zmax/2)]*10,label='Ez_inc')
    axx.set_xlabel('Grid Cells ($x$)')
    axx.set_ylabel('Ez field')
    axx.set_title('X profile')

    #ax2.plot(Z,hx_inc,label='Hx_inc')


# plot calculated field shortly after source position
    axy = fig.add_axes([.4, .1, .2, .2])
    axy.plot(Y/nm,np.abs(e.z[int(dims.x/2),:,int(dims.z/2)]),label='Ez_inc')
    axy.set_xlabel('Grid Cells ($y$)')
    axy.set_ylabel('Ez field')
    axy.set_title('Y profile')


# plot calculated field shortly after source position
    axz = fig.add_axes([.75, .1, .2, .2])
    axz.plot(Z/nm,np.abs(e.z[int(dims.x/2),int(dims.y/2),:]),label='Ez_inc')
    axz.set_xlabel('Grid Cells ($z$)')
    axz.set_ylabel('Ez field')
    axz.set_title('Z profile')
    #plt.tight_layout()
    #plt.savefig('Animation/frametime{}'.format(int(t/10)))
    path = 'Plots/Plots_nkmax{}_dx{}nm_dt{}as'.format(grid.n_kmax,int(ddx/nm),np.round(dt*1e18,2))
    if not os.path.exists(path):
        os.makedirs(path)
    save_name = path+'/animation_time{}.png'.format(int(time_step/cycle))
    """
    # plt.savefig(save_name)
    plt.pause(time_pause)  # pause sensible value to watch what is happening


def setup_GIF(
    FLAG,
    fig,
    ax,
    ddx,
    dt,
    length,
    array,
    dims,
    sphere,
    pulse,
    npml,
    tfsf,
    e,
    ez_inc,
    hx_inc,
    f_plot,
):

    "2d field plots"
    ims = []
    x, z = np.meshgrid(array.x, array.z)
    ims.append(ax[1, 0].imshow(np.zeros((dims.x, dims.y))))
    ims.append(
        ax[1, 1].imshow(
            np.zeros((dims.x, 1)),
            cmap="viridis",
            interpolation="quadric",
            origin="lower",
        )
    )
    ims.append(
        ax[1, 2].imshow(
            np.zeros((1, 1)), cmap="viridis", interpolation="quadric", origin="lower"
        )
    )

    # Labels
    ax[1, 0].set_xlabel("$x$ [nm]")
    ax[1, 0].set_ylabel("$y$ [nm]")

    ax[1, 1].set_xlabel("$x$ [nm]")
    ax[1, 1].set_ylabel("$z$ [nm]")

    ax[1, 2].set_xlabel("$y$ [nm]")
    ax[1, 2].set_ylabel("$z$ [nm]")

    # 2d fields
    for im in ims:
        im.set_clim(vmin=0, vmax=10**5)
    ims[0].set_extent((0, length.x / nm, 0, length.x / nm))
    ims[1].set_extent((0, length.x / nm, 0, length.x / nm))
    ims[2].set_extent((0, length.x / nm, 0, length.x / nm))
    cbaxes = fig.add_axes([0.92, 0.4, 0.01, 0.2])
    cbar = plt.colorbar(ims[0], cax=cbaxes, orientation="vertical")
    cbar.ax.set_title("Field [arb. units]")

    ax[0, 0].set_axis_off()
    field_component = "Ez"

    "Information"
    text_tstep = ax[0, 0].annotate(
        "Time Step: 0",
        (0.5, 1),
        xycoords="axes fraction",
        va="center",
        ha="center",
        weight="bold",
    )

    # boundary condition
    if FLAG.TFSF == 1:
        plot_text = "\n" + "TFSF on"
    elif FLAG.TFSF == 2:
        plot_text = "\n" + "Periodic boundary condtions"

    # object information
    if FLAG.OBJECT == 0:
        plot_text += "\n No object implemented"
    elif FLAG.OBJECT == 1:
        plot_text += "\n Sphere of radius " + str(int(sphere.R / nm)) + "nm implemented"
    if FLAG.OBJECT == 2:
        plot_text += (
            "\n Rectangle of thickness" + str(int(sphere.R / nm)) + "nm implemented"
        )

    # material information
    if FLAG.MATERIAL == 0:
        plot_text += "\n No object implemented"
    elif FLAG.MATERIAL == 1:
        plot_text += "\n Drude model implemented"
    if FLAG.MATERIAL == 3:
        plot_text += "\n Etchegoin model used"

    # pulse information
    plot_text += (
        "\n\n Pulse information\n Amplitude: "
        + str(pulse.amplitude)
        + "\n center energy: {} eV".format(np.round(pulse.energy, 2))
        + "\n wavelength: {:06.2f} nm".format(pulse.lam_0 / nm)
        + "\n dx: {}".format(ddx * 1e9)
        + " nm"
        + "\n dt: {}".format(np.round(dt * 1e18, 2))
        + " as"
    )

    # FDTD information
    plot_text += (
        "\n\n Grid Dimensions: \nx: "
        + str(dims.x)
        + ", y: "
        + str(dims.y)
        + ", z: "
        + str(dims.z)
    )
    # plot_text += '\nScatter Field: ' + str(scat_field)
    if FLAG.PML == 1:
        plot_text += ", PML: " + str(npml)
    else:
        plot_text += ", PML: Off"

    ax[0, 0].annotate(
        plot_text,
        (0, 0.9),
        xycoords="axes fraction",
        va="top",
        ha="left",
    )

    # #incident field
    # incident_e, = ax[0,1].plot(array.y/nm,ez_inc,label='Ez_inc')
    # incident_h, = ax[0,1].plot(array.y/nm,hx_inc,label='Hx_inc')
    # ax[0,1].set_xlabel('Grid Cells ($y$)')
    # ax[0,1].set_ylabel('Fields')
    # ax[0,1].set_title('Incident fields')
    # ax[0,1].legend()

    # Jonas plot
    (plot_f,) = ax[0, 1].plot(
        grid.k_grid(grid.n_kmax),
        f_plot[:, 0, int(grid.n_thetamax / 2)],
        label="f_k(r=0)",
    )
    (plot_f2,) = ax[0, 1].plot(
        grid.k_grid(grid.n_kmax),
        f_plot[:, int(grid.n_phimax / 4), int(grid.n_thetamax / 2)],
        label="f_k(r=0)",
    )
    (plot_f3,) = ax[0, 1].plot(
        grid.k_grid(grid.n_kmax), f_plot[:, 0, 0], label="f_k(r=0)"
    )
    ax[0, 1].set_xlabel("k grid")
    ax[0, 1].set_ylabel("Wigner function")
    ax[0, 1].set_ylim(0, 1)
    ax[0, 1].set_title("Wigner function")
    ax[0, 1].legend(loc="upper right")

    time_array = np.arange(1, tsteps + 1, 1)
    ax[0, 2].plot(
        time_array * dt * 1e15, pulse.update_value(time_array, dt), label="Pulse"
    )
    (incident_loc,) = ax[0, 2].plot(
        time_array[10] * dt, pulse.update_value(time_array[10], dt), "bo"
    )
    # incident_frame, = ax[0,2].axvspan((time_array[10]*dt-length.y/(2*c))*1e15, (time_array[10]*dt+length.y/(2*c))*1e15, alpha=0.5, color='gray')
    ax[0, 2].set_title("Optical Pulse")

    # polarization

    imp = []
    imp.append(ax[2, 0].imshow(np.zeros((dims.x, dims.y))))

    imp.append(
        ax[2, 1].imshow(
            np.zeros((dims.x, 1)),
            cmap="viridis",
            interpolation="quadric",
            origin="lower",
        )
    )

    imp.append(
        ax[2, 2].imshow(
            np.zeros((1, 1)), cmap="viridis", interpolation="quadric", origin="lower"
        )
    )

    # Labels
    ax[2, 0].set_xlabel("Grid Cells ($x$)")
    ax[2, 0].set_ylabel("Grid Cells ($y$)")

    ax[2, 1].set_xlabel("Grid Cells ($x$)")
    ax[2, 1].set_ylabel("Grid Cells ($z$)")

    ax[2, 2].set_xlabel("Grid Cells ($y$)")
    ax[2, 2].set_ylabel("Grid Cells ($z$)")

    ax[0, 2].set_xlabel("Time [fs]")
    ax[0, 2].set_ylabel("Pulse")

    for im in ims:
        im.set_clim(vmin=0, vmax=10**5)
    imp[0].set_extent((0, length.x / nm, 0, length.x / nm))
    imp[1].set_extent((0, dims.x, 0, dims.x))
    imp[2].set_extent((0, dims.x, 0, dims.x))
    cbaxes_p = fig.add_axes([0.93, 0.1, 0.01, 0.2])
    cbar_p = plt.colorbar(imp[0], cax=cbaxes_p, orientation="vertical")
    cbar_p.ax.set_title("Polarization [arb. units]")

    # 'add lines and circles'

    fx = plt.Circle(
        (sphere.x / nm, sphere.y / nm), sphere.R / nm, color="r", fill=False
    )
    fy = plt.Circle(
        (sphere.x / nm, sphere.y / nm), sphere.R / nm, color="r", fill=False
    )
    fz = plt.Circle(
        (sphere.x / nm, sphere.y / nm), sphere.R / nm, color="r", fill=False
    )
    px = plt.Circle(
        (sphere.x / nm, sphere.y / nm), sphere.R / nm, color="r", fill=False
    )
    py = plt.Circle(
        (sphere.x / nm, sphere.y / nm), sphere.R / nm, color="r", fill=False
    )
    pz = plt.Circle(
        (sphere.x / nm, sphere.y / nm), sphere.R / nm, color="r", fill=False
    )

    # fields
    ax[1, 0].set_aspect(1)
    ax[1, 1].set_aspect(1)
    ax[1, 2].set_aspect(1)

    # ax[1,0].add_artist( fx )
    # ax[1,1].add_artist( fy )
    # ax[1,2].add_artist( fz )

    # polarization
    ax[2, 0].set_aspect(1)
    ax[2, 1].set_aspect(1)
    ax[2, 2].set_aspect(1)

    # ax[2,0].add_artist( px )
    # ax[2,1].add_artist( py )
    # ax[2,2].add_artist( pz )

    # PML layers
    # ax[1,0].hlines(npml*ddx/nm,npml*ddx/nm,(dims.x-npml-1)*ddx/nm, 'b')
    # ax[1,0].hlines((dims.x-npml-1)*ddx/nm,npml*ddx/nm,(dims.x-npml-1)*ddx/nm, 'b')
    # ax[1,0].vlines(npml*ddx/nm,npml*ddx/nm,(dims.x-npml-1)*ddx/nm, 'b')
    # ax[1,0].vlines((dims.x-npml-1)*ddx/nm,npml*ddx/nm,(dims.x-npml-1)*ddx/nm, 'b')
    # ax[1,1].hlines(npml*ddx/nm,npml*ddx/nm,(dims.x-npml-1)*ddx/nm, 'b')
    # ax[1,1].hlines((dims.x-npml-1)*ddx/nm,npml*ddx/nm,(dims.x-npml-1)*ddx/nm, 'b')
    # ax[1,1].vlines(npml*ddx/nm,npml*ddx/nm,(dims.x-npml-1)*ddx/nm, 'b')
    # ax[1,1].vlines((dims.x-npml-1)*ddx/nm,npml*ddx/nm,(dims.x-npml-1)*ddx/nm, 'b')
    # ax[1,2].hlines(npml*ddx/nm,npml*ddx/nm,(dims.x-npml-1)*ddx/nm, 'b')
    # ax[1,2].hlines((dims.x-npml-1)*ddx/nm,npml*ddx/nm,(dims.x-npml-1)*ddx/nm, 'b')
    # ax[1,2].vlines(npml*ddx/nm,npml*ddx/nm,(dims.x-npml-1)*ddx/nm, 'b')
    # ax[1,2].vlines((dims.x-npml-1)*ddx/nm,npml*ddx/nm,(dims.x-npml-1)*ddx/nm, 'b')

    # #TFSF
    # ax[1,0].hlines(tfsf.x_min*ddx/nm,tfsf.z_min*ddx/nm,(tfsf.z_max+1)*ddx/nm, 'r')
    # ax[1,0].hlines((tfsf.x_max+1)*ddx/nm,tfsf.z_min*ddx/nm,(tfsf.z_max+1)*ddx/nm, 'r')
    # ax[1,0].vlines(tfsf.z_min*ddx/nm,tfsf.x_min*ddx/nm,(tfsf.x_max+1)*ddx/nm, 'r')
    # ax[1,0].vlines((tfsf.z_max+1)*ddx/nm,tfsf.x_min*ddx/nm,(tfsf.x_max+1)*ddx/nm, 'r')
    # ax[1,1].hlines(tfsf.x_min*ddx/nm,tfsf.z_min*ddx/nm,(tfsf.z_max+1)*ddx/nm, 'r')
    # ax[1,1].hlines((tfsf.x_max+1)*ddx/nm,tfsf.z_min*ddx/nm,(tfsf.z_max+1)*ddx/nm, 'r')
    # ax[1,1].vlines(tfsf.z_min*ddx/nm,tfsf.x_min*ddx/nm,(tfsf.x_max+1)*ddx/nm, 'r')
    # ax[1,1].vlines((tfsf.z_max+1)*ddx/nm,tfsf.x_min*ddx/nm,(tfsf.x_max+1)*ddx/nm, 'r')
    # ax[1,2].hlines(tfsf.x_min*ddx/nm,tfsf.z_min*ddx/nm,(tfsf.z_max+1)*ddx/nm, 'r')
    # ax[1,2].hlines((tfsf.x_max+1)*ddx/nm,tfsf.z_min*ddx/nm,(tfsf.z_max+1)*ddx/nm, 'r')
    # ax[1,2].vlines(tfsf.z_min*ddx/nm,tfsf.x_min*ddx/nm,(tfsf.x_max+1)*ddx/nm, 'r')
    # ax[1,2].vlines((tfsf.z_max+1)*ddx/nm,tfsf.x_min*ddx/nm,(tfsf.x_max+1)*ddx/nm, 'r')

    return ims, imp, text_tstep, incident_loc, plot_f, plot_f2, plot_f3


def animate_GIF(
    t,
    text_tstep,
    e,
    dims,
    ims,
    array,
    ax,
    ez_inc,
    hx_inc,
    incident_loc,
    pulse,
    p,
    imp,
    time_pause,
    f_plot,
    plot_f,
    plot_f2,
    plot_f3,
):
    text_tstep.set_text("Time Step: " + str(t))
    max = np.max(np.abs(e.z[:, :, int(dims.y / 2)]))
    # print('max',max)
    for im in ims:
        im.set_clim(vmin=0, vmax=max)
    x, y = np.meshgrid(array.x, array.y)
    ims[0].set_data(np.transpose(np.abs(e.z[:, :, int(dims.y / 2)])))
    ims[1].set_data(np.transpose(np.abs(e.z[:, int(dims.y / 2), :])))
    ims[2].set_data(np.transpose(np.abs(e.z[int(dims.y / 2), :, :])))

    # #incident field
    # ax[0,1].set_ylim(-max,max)
    # incident_e.set_data(array.y/nm,ez_inc)
    # incident_h.set_data(array.y/nm,hx_inc)

    # incident field
    # ax[0,1].set_ylim(-max,max)
    plot_f.set_data(grid.k_grid(grid.n_kmax), f_plot[:, 0, int(grid.n_thetamax / 2)])
    plot_f2.set_data(
        grid.k_grid(grid.n_kmax),
        f_plot[:, int(grid.n_phimax / 4), int(grid.n_thetamax / 2)],
    )
    plot_f3.set_data(grid.k_grid(grid.n_kmax), f_plot[:, 0, int(0)])

    incident_loc.set_data(t * dt * 1e15, pulse.update_value(t, dt))
    # incident_frame.setdata((t*dt-length.y/(2*c))*1e15, (t*dt+length.y/(2*c))*1e15)

    max_p = np.max(np.abs(p.z[:, :, int(dims.y / 2)]))
    for im in imp:
        im.set_clim(vmin=0, vmax=max_p)
    imp[0].set_data(np.transpose(np.abs(p.z[:, :, int(dims.z / 2)])))
    imp[1].set_data(np.transpose(np.abs(p.z[:, int(dims.y / 2), :])))
    imp[2].set_data(np.transpose(np.abs(p.z[int(dims.x / 2), :, :])))

    # plt.savefig(save_name)
    plt.pause(time_pause)  # pause sensible value to watch what is happening


# ####################################################
# # Animation
# ####################################################

# def graph(t,e,FLAG,tfsf):
#     """This is a mess"""
# # main graph is E(z,y, time snapshops), and a small graph of E(t) as center
#     plt.clf() # close each time for new update graph/colormap
#     ax = fig.add_axes([.05, .35, .2, .4])

# # 2d plot - several options, two examples below
# #    img = ax.imshow(Ez)
#     x,y =np.meshgrid(array.x,array.y)
#     #img = ax.contourf(x,y,np.transpose(ez[int(Ymax/2),:,:]*10))
#     img = ax.contourf(x/nm,y/nm,np.transpose(np.abs(e.z[:,:,int(dims.y/2)])))
# #    img = ax.contourf(x,y,np.transpose(np.round(ez[:,:,int(Ymax/2)],10)))
#     cbar=plt.colorbar(img, ax=ax)
#     cbar.set_label('$Ez$ (arb. units)')

# # add labels to axes
#     ax.set_xlabel('Grid Cells ($x$)')
#     ax.set_ylabel('Grid Cells ($y$)')


#     cc = plt.Circle((xc/nm, yc/nm), R/nm, color='r',fill=False)
#     ax.set_aspect( 1 )
#     ax.add_artist( cc )

#     ax.hlines(npml*ddx/nm,npml*ddx/nm,(dims.x-npml-1)*ddx/nm, 'b')
#     ax.hlines((dims.x-npml-1)*ddx/nm,npml*ddx/nm,(dims.x-npml-1)*ddx/nm, 'b')
#     ax.vlines(npml*ddx/nm,npml*ddx/nm,(dims.x-npml-1)*ddx/nm, 'b')
#     ax.vlines((dims.x-npml-1)*ddx/nm,npml*ddx/nm,(dims.x-npml-1)*ddx/nm, 'b')

#     #TFSF
#     ax.hlines(tfsf.y_min*ddx/nm,tfsf.x_min*ddx/nm,tfsf.x_max*ddx/nm, 'r')
#     ax.hlines(tfsf.y_max*ddx/nm,tfsf.x_min*ddx/nm,tfsf.x_max*ddx/nm, 'r')
#     ax.vlines(tfsf.x_min*ddx/nm,tfsf.y_min*ddx/nm,tfsf.y_max*ddx/nm, 'r')
#     ax.vlines(tfsf.x_max*ddx/nm,tfsf.y_min*ddx/nm,tfsf.y_max*ddx/nm, 'r')


#     if FLAG.POINT ==1:
#             # incident field
#         ax2 = fig.add_axes([.4, .75, .2, .2])
#         ax2.plot(pulsemon_t,label='Source_mon')
#         ax2.plot(ez_source_t,label='Ez_mon')

#         ax2.set_ylim(-1.1,1.1)
#         ax2.set_xlabel('Grid Cells ($y$)')
#         ax2.set_ylabel('Fields')
#         ax2.set_title('Incident fields')
#         ax2.legend()

#     if FLAG.TFSF ==1 or FLAG.TFSF ==2:
#         # incident field
#         ax2 = fig.add_axes([.4, .75, .2, .2])
#         ax2.plot(Y/nm,ez_inc,label='Ez_inc')
#         ax2.plot(Y/nm,hx_inc,label='Hx_inc')
#         #ax2.set_ylim(-1.1,1.1)
#         ax2.set_xlabel('Grid Cells ($y$)')
#         ax2.set_ylabel('Fields')
#         ax2.set_title('Incident fields')
#         ax2.legend()

#     ax01 = fig.add_axes([.4, .35, .2, .4])

# # 2d plot - several options, two examples below
# #    img = ax.imshow(Ez)
#     z,y =np.meshgrid(X,Y)
#     #img = ax.contourf(x,y,np.transpose(ez[int(Ymax/2),:,:]*10))
#     img = ax01.contourf(y/nm,z/nm,np.abs(e.z[int(dims.x/2),:,:]))
#     cbar=plt.colorbar(img, ax=ax01)
#     cbar.set_label('$Ez$ (arb. units)')

# # add labels to axes
#     ax01.set_xlabel('Grid Cells ($y$)')
#     ax01.set_ylabel('Grid Cells ($z$)')

#     cc = plt.Circle((yc/nm, zc/nm), R/nm, color='r',fill=False)
#     ax01.set_aspect( 1 )
#     ax01.add_artist( cc )

#     #PML layers
#     ax01.hlines(npml*ddx/nm,npml*ddx/nm,(dims.x-npml-1)*ddx/nm, 'b')
#     ax01.hlines((dims.x-npml-1)*ddx/nm,npml*ddx/nm,(dims.x-npml-1)*ddx/nm, 'b')
#     ax01.vlines(npml*ddx/nm,npml*ddx/nm,(dims.x-npml-1)*ddx/nm, 'b')
#     ax01.vlines((dims.x-npml-1)*ddx/nm,npml*ddx/nm,(dims.x-npml-1)*ddx/nm, 'b')

#     #TFSF
#     ax01.hlines(tfsf.x_min*ddx/nm,tfsf.z_min*ddx/nm,tfsf.z_max*ddx/nm, 'r')
#     ax01.hlines(tfsf.x_max*ddx/nm,tfsf.z_min*ddx/nm,tfsf.z_max*ddx/nm, 'r')
#     ax01.vlines(tfsf.z_min*ddx/nm,tfsf.x_min*ddx/nm,tfsf.x_max*ddx/nm, 'r')
#     ax01.vlines(tfsf.z_max*ddx/nm,tfsf.x_min*ddx/nm,tfsf.x_max*ddx/nm, 'r')

#     if (FLAG.MICRO ==1):
#         axmicro = fig.add_axes([.75, .75, .2, .2])
#         axmicro.plot(grid.k_grid(grid.n_kmax),f_plot,label='Fermi distribution')
#         axmicro.set_xlabel('Grid Cells ($k$)')
#         axmicro.set_ylabel('Fermi')
#         axmicro.set_title('Fermi dist')
#         axmicro.legend()


#     ax02 = fig.add_axes([.75, .35, .2, .4])

# # 2d plot - several options, two examples below
# #    img = ax.imshow(Ez)
#     x,z =np.meshgrid(X,Z)
#     #img = ax.contourf(x,y,np.transpose(ez[int(Ymax/2),:,:]*10))
#     img = ax02.contourf(z/nm,x/nm,np.transpose(np.abs(e.z[:,int(dims.y/2),:])))
#     cbar=plt.colorbar(img, ax=ax02)
#     cbar.set_label('$Ez$ (arb. units)')

# # add labels to axes
#     ax02.set_xlabel('Grid Cells ($z$)')
#     ax02.set_ylabel('Grid Cells ($x$)')

#     cc = plt.Circle((zc/nm, xc/nm), R/nm, color='r',fill=False)
#     ax02.set_aspect( 1 )
#     ax02.add_artist( cc )

#     ax02.hlines(npml*ddx/nm,npml*ddx/nm,(dims.x-npml-1)*ddx/nm, 'b')
#     ax02.hlines((dims.x-npml-1)*ddx/nm,npml*ddx/nm,(dims.x-npml-1)*ddx/nm, 'b')
#     ax02.vlines(npml*ddx/nm,npml*ddx/nm,(dims.x-npml-1)*ddx/nm, 'b')
#     ax02.vlines((dims.x-npml-1)*ddx/nm,npml*ddx/nm,(dims.x-npml-1)*ddx/nm, 'b')

#     #TFSF
#     ax02.hlines(tfsf.x_min*ddx/nm,tfsf.z_min*ddx/nm,tfsf.z_max*ddx/nm, 'r')
#     ax02.hlines(tfsf.x_max*ddx/nm,tfsf.z_min*ddx/nm,tfsf.z_max*ddx/nm, 'r')
#     ax02.vlines(tfsf.z_min*ddx/nm,tfsf.x_min*ddx/nm,tfsf.x_max*ddx/nm, 'r')
#     ax02.vlines(tfsf.z_max*ddx/nm,tfsf.x_min*ddx/nm,tfsf.x_max*ddx/nm, 'r')
# # add title with current simulation time step
#     ax.set_title("frame time {}".format(t))
#     '''
# # incident field
#     ax2 = fig.add_axes([.05, .1, .2, .2])
#     ax2.plot(Z,ez_inc,label='Ez_inc')
#     ax2.plot(Z,hx_inc,label='Hx_inc')
#     #ax2.set_ylim(-1.1,1.1)
#     ax2.set_xlabel('Grid Cells ($y$)')
#     ax2.set_ylabel('Fields')
#     ax2.set_title('Incident fields')
#     ax2.legend()
#     '''
# # plot calculated field shortly after source position
#     axx = fig.add_axes([.05, .1, .2, .2])
#     axx.plot(X/nm,np.abs(e.z[:,int(dims.y/2),int(dims.z/2)]),label='Ez_inc')
#     #ax3.plot(Z,ez[:,ja,int(Zmax/2)]*10,label='Ez_inc')
#     axx.set_xlabel('Grid Cells ($x$)')
#     axx.set_ylabel('Ez field')
#     axx.set_title('X profile')

#     #ax2.plot(Z,hx_inc,label='Hx_inc')


# # plot calculated field shortly after source position
#     axy = fig.add_axes([.4, .1, .2, .2])
#     axy.plot(Y/nm,np.abs(e.z[int(dims.x/2),:,int(dims.z/2)]),label='Ez_inc')
#     axy.set_xlabel('Grid Cells ($y$)')
#     axy.set_ylabel('Ez field')
#     axy.set_title('Y profile')


# # plot calculated field shortly after source position
#     axz = fig.add_axes([.75, .1, .2, .2])
#     axz.plot(Z/nm,np.abs(e.z[int(dims.x/2),int(dims.y/2),:]),label='Ez_inc')
#     axz.set_xlabel('Grid Cells ($z$)')
#     axz.set_ylabel('Ez field')
#     axz.set_title('Z profile')
#     #plt.tight_layout()
#     #plt.savefig('Animation/frametime{}'.format(int(t/10)))
#     path = 'Plots/Plots_nkmax{}_dx{}nm_dt{}as'.format(grid.n_kmax,int(ddx/nm),np.round(dt*1e18,2))
#     if not os.path.exists(path):
#         os.makedirs(path)
#     save_name = path+'/animation_time{}.png'.format(int(time_step/cycle))
#     plt.savefig(save_name)
#     plt.pause(time_pause) # pause sensible value to watch what is happening

# def graph_new(t):
#     text_tstep.set_text('Time Step: ' + str(t))
#     max = np.max(np.abs(e.z[:,:,int(dims.y/2)]))
#     #print('max',max)
#     for im in ims:
#         im.set_clim(vmin=0, vmax=max)
#     x,y =np.meshgrid(X,Y)
#     ims[0].set_data(np.transpose(np.abs(e.z[:,:,int(dims.y/2)])))
#     ims[1].set_data(np.transpose(np.abs(e.z[:,int(dims.y/2),:])))
#     ims[2].set_data(np.transpose(np.abs(e.z[int(dims.y/2),:,:])))

#     # 1d plots
#     ax[0,2].set_ylim(-max,max)
#     xcut.set_data(X/nm,np.abs(e.z[:,int(dims.y/2),int(dims.z/2)]))

#     ax[1,2].set_ylim(-max,max)
#     ycut.set_data(Y/nm,np.abs(e.z[int(dims.x/2),:,int(dims.z/2)]))

#     ax[2,2].set_ylim(-max,max)
#     zcut.set_data(Z/nm,np.abs(e.z[int(dims.x/2),int(dims.y/2),:]))

#     #incident field
#     ax[1,0].set_ylim(-max,max)
#     incident_e.set_data(Y/nm,ez_inc)
#     incident_h.set_data(Y/nm,hx_inc)

#     max_p = np.max(np.abs(p.z[:,:,int(dims.y/2)]))
#     for im in imp:
#         im.set_clim(vmin=0, vmax=max_p)
#     imp[0].set_data(np.transpose(np.abs(p.z[:,:,int(dims.z/2)])))
#     imp[1].set_data(np.transpose(np.abs(p.z[:,int(dims.y/2),:])))
#     imp[2].set_data(np.transpose(np.abs(p.z[int(dims.x/2),:,:])))

# # main graph is E(z,y, time snapshops), and a small graph of E(t) as center
#     #plt.clf() # close each time for new update graph/colormap
#     #fig,ax = plt.subplots(3, 4, figsize=(10, 6))

# # 2d plot - several options, two examples below
# #    img = ax.imshow(Ez)
# #     x,y =np.meshgrid(X,Y)
# #     #img = ax.contourf(x,y,np.transpose(ez[int(Ymax/2),:,:]*10))
# #     #ax[0,1].contourf(x/nm,y/nm,np.transpose(np.abs(e.z[:,:,int(dims.y/2)])))
# #     axs = ax[0, 1]
# #     img = axs.contourf(x/nm,y/nm,np.transpose(np.abs(e.z[:,:,int(dims.y/2)])))
# # #    img = ax.contourf(x,y,np.transpose(np.round(ez[:,:,int(Ymax/2)],10)))
# #     cbar=plt.colorbar(img, ax=axs)
# #     cbar.set_label('$Ez$ (arb. units)')
#     # ax = axs[0,1]
#     # pcm = ax.pcolormesh(np.random.random((20, 20)))
#     # fig.colorbar(pcm, ax=ax, shrink=0.6)

# # # add labels to axes
# #     axs.set_xlabel('Grid Cells ($x$)')
# #     axs.set_ylabel('Grid Cells ($y$)')


#     # cc = plt.Circle((xc/nm, yc/nm), R/nm, color='r',fill=False)
#     # ax[0,1].set_aspect( 1 )
#     # ax[0,1].add_artist( cc )

#     # ax[0,1].hlines(npml*ddx/nm,npml*ddx/nm,(dims.x-npml-1)*ddx/nm, 'b')
#     # ax[0,1].hlines((dims.x-npml-1)*ddx/nm,npml*ddx/nm,(dims.x-npml-1)*ddx/nm, 'b')
#     # ax[0,1].vlines(npml*ddx/nm,npml*ddx/nm,(dims.x-npml-1)*ddx/nm, 'b')
#     # ax[0,1].vlines((dims.x-npml-1)*ddx/nm,npml*ddx/nm,(dims.x-npml-1)*ddx/nm, 'b')

#     # #TFSF
#     # ax[0,1].hlines(tfsf.y_min*ddx/nm,tfsf.x_min*ddx/nm,tfsf.x_max*ddx/nm, 'r')
#     # ax[0,1].hlines(tfsf.y_max*ddx/nm,tfsf.x_min*ddx/nm,tfsf.x_max*ddx/nm, 'r')
#     # ax[0,1].vlines(tfsf.x_min*ddx/nm,tfsf.y_min*ddx/nm,tfsf.y_max*ddx/nm, 'r')
#     # ax[0,1].vlines(tfsf.x_max*ddx/nm,tfsf.y_min*ddx/nm,tfsf.y_max*ddx/nm, 'r')


#     # incident field
#     # axs[1,0].plot(Y/nm,ez_inc,label='Ez_inc')
#     # axs[1,0].plot(Y/nm,hx_inc,label='Hx_inc')
#     # #ax2.set_ylim(-1.1,1.1)
#     # axs[1,0].set_xlabel('Grid Cells ($y$)')
#     # axs[1,0].set_ylabel('Fields')
#     # axs[1,0].set_title('Incident fields')
#     # axs[1,0].legend()
#     '''
#     ax01 = fig.add_axes([.4, .35, .2, .4])

# # 2d plot - several options, two examples below
# #    img = ax.imshow(Ez)
#     z,y =np.meshgrid(X,Y)
#     #img = ax.contourf(x,y,np.transpose(ez[int(Ymax/2),:,:]*10))
#     img = ax01.contourf(y/nm,z/nm,np.abs(e.z[int(dims.x/2),:,:]))
#     cbar=plt.colorbar(img, ax=ax01)
#     cbar.set_label('$Ez$ (arb. units)')

# # add labels to axes
#     ax01.set_xlabel('Grid Cells ($y$)')
#     ax01.set_ylabel('Grid Cells ($z$)')

#     cc = plt.Circle((yc/nm, zc/nm), R/nm, color='r',fill=False)
#     ax01.set_aspect( 1 )
#     ax01.add_artist( cc )

#     #PML layers
#     ax01.hlines(npml*ddx/nm,npml*ddx/nm,(dims.x-npml-1)*ddx/nm, 'b')
#     ax01.hlines((dims.x-npml-1)*ddx/nm,npml*ddx/nm,(dims.x-npml-1)*ddx/nm, 'b')
#     ax01.vlines(npml*ddx/nm,npml*ddx/nm,(dims.x-npml-1)*ddx/nm, 'b')
#     ax01.vlines((dims.x-npml-1)*ddx/nm,npml*ddx/nm,(dims.x-npml-1)*ddx/nm, 'b')

#     #TFSF
#     ax01.hlines(tfsf.x_min*ddx/nm,tfsf.z_min*ddx/nm,tfsf.z_max*ddx/nm, 'r')
#     ax01.hlines(tfsf.x_max*ddx/nm,tfsf.z_min*ddx/nm,tfsf.z_max*ddx/nm, 'r')
#     ax01.vlines(tfsf.z_min*ddx/nm,tfsf.x_min*ddx/nm,tfsf.x_max*ddx/nm, 'r')
#     ax01.vlines(tfsf.z_max*ddx/nm,tfsf.x_min*ddx/nm,tfsf.x_max*ddx/nm, 'r')

#     axmicro = fig.add_axes([.75, .75, .2, .2])
#     axmicro.plot(grid.k_grid(grid.n_kmax),f_plot,label='Fermi distribution')
#     axmicro.set_xlabel('Grid Cells ($k$)')
#     axmicro.set_ylabel('Fermi')
#     axmicro.set_title('Fermi dist')
#     axmicro.legend()


#     ax02 = fig.add_axes([.75, .35, .2, .4])

# # 2d plot - several options, two examples below
# #    img = ax.imshow(Ez)
#     x,z =np.meshgrid(X,Z)
#     #img = ax.contourf(x,y,np.transpose(ez[int(Ymax/2),:,:]*10))
#     img = ax02.contourf(z/nm,x/nm,np.transpose(np.abs(e.z[:,int(dims.y/2),:])))
#     cbar=plt.colorbar(img, ax=ax02)
#     cbar.set_label('$Ez$ (arb. units)')

# # add labels to axes
#     ax02.set_xlabel('Grid Cells ($z$)')
#     ax02.set_ylabel('Grid Cells ($x$)')

#     cc = plt.Circle((zc/nm, xc/nm), R/nm, color='r',fill=False)
#     ax02.set_aspect( 1 )
#     ax02.add_artist( cc )

#     ax02.hlines(npml*ddx/nm,npml*ddx/nm,(dims.x-npml-1)*ddx/nm, 'b')
#     ax02.hlines((dims.x-npml-1)*ddx/nm,npml*ddx/nm,(dims.x-npml-1)*ddx/nm, 'b')
#     ax02.vlines(npml*ddx/nm,npml*ddx/nm,(dims.x-npml-1)*ddx/nm, 'b')
#     ax02.vlines((dims.x-npml-1)*ddx/nm,npml*ddx/nm,(dims.x-npml-1)*ddx/nm, 'b')

#     #TFSF
#     ax02.hlines(tfsf.x_min*ddx/nm,tfsf.z_min*ddx/nm,tfsf.z_max*ddx/nm, 'r')
#     ax02.hlines(tfsf.x_max*ddx/nm,tfsf.z_min*ddx/nm,tfsf.z_max*ddx/nm, 'r')
#     ax02.vlines(tfsf.z_min*ddx/nm,tfsf.x_min*ddx/nm,tfsf.x_max*ddx/nm, 'r')
#     ax02.vlines(tfsf.z_max*ddx/nm,tfsf.x_min*ddx/nm,tfsf.x_max*ddx/nm, 'r')
# # add title with current simulation time step
#     ax.set_title("frame time {}".format(t))
#     '''
# # # incident field
# #     ax2 = fig.add_axes([.05, .1, .2, .2])
# #     ax2.plot(Z,ez_inc,label='Ez_inc')
# #     ax2.plot(Z,hx_inc,label='Hx_inc')
# #     #ax2.set_ylim(-1.1,1.1)
# #     ax2.set_xlabel('Grid Cells ($y$)')
# #     ax2.set_ylabel('Fields')
# #     ax2.set_title('Incident fields')
# #     ax2.legend()
#     '''
# # plot calculated field shortly after source position
#     axx = fig.add_axes([.05, .1, .2, .2])
#     axx.plot(X/nm,np.abs(e.z[:,int(dims.y/2),int(dims.z/2)]),label='Ez_inc')
#     #ax3.plot(Z,ez[:,ja,int(Zmax/2)]*10,label='Ez_inc')
#     axx.set_xlabel('Grid Cells ($x$)')
#     axx.set_ylabel('Ez field')
#     axx.set_title('X profile')

#     #ax2.plot(Z,hx_inc,label='Hx_inc')


# # plot calculated field shortly after source position
#     axy = fig.add_axes([.4, .1, .2, .2])
#     axy.plot(Y/nm,np.abs(e.z[int(dims.x/2),:,int(dims.z/2)]),label='Ez_inc')
#     axy.set_xlabel('Grid Cells ($y$)')
#     axy.set_ylabel('Ez field')
#     axy.set_title('Y profile')


# # plot calculated field shortly after source position
#     axz = fig.add_axes([.75, .1, .2, .2])
#     axz.plot(Z/nm,np.abs(e.z[int(dims.x/2),int(dims.y/2),:]),label='Ez_inc')
#     axz.set_xlabel('Grid Cells ($z$)')
#     axz.set_ylabel('Ez field')
#     axz.set_title('Z profile')
#     #plt.tight_layout()
#     #plt.savefig('Animation/frametime{}'.format(int(t/10)))
#     path = 'Plots/Plots_nkmax{}_dx{}nm_dt{}as'.format(grid.n_kmax,int(ddx/nm),np.round(dt*1e18,2))
#     if not os.path.exists(path):
#         os.makedirs(path)
#     save_name = path+'/animation_time{}.png'.format(int(time_step/cycle))
#     '''
#     #plt.savefig(save_name)
#     plt.pause(time_pause) # pause sensible value to watch what is happening


# plot functions for Greens function etc might be interesting but has to be reviewed
# def plotpoint(FLAG,):
#     if(FLAG.FFT==1):
#         "Plot frequency dependent 1D monitor"
#         fig = plt.figure(figsize=(14, 6))
#         for i in range(n_mon):
#             ax = fig.add_subplot(2, 3,i+1)
#             ax.plot(hbar*omega/eC,np.abs(ez_mon_om[i,:]))
#             ax.set_title("Pos: x = {0}, y= {1} ,z= {2}".format(loc_monitors[i][0],loc_monitors[i][1],loc_monitors[i][2]))
#             ax.set_xlim(0,4)
#             ax.set_xlabel('$\hbar\omega$ [eV]')
#             ax.set_ylabel('$|E_z(\omega)|$')
#         #plt.subplots_adjust(bottom=0.05, left=0.05)
#         plt.tight_layout()
#         plt.show()

#     if(FLAG.POINT==1):

#         "Plot frequency dependent 1D monitor"
#         fig,ax = plt.subplots(figsize=(9, 6))

#         ax.plot(hbar*omega_source/eC,np.imag(GFT)*1/ddx**3*1,label='GFT FDTD')
#         ax.plot(hbar*omega_source/eC,GFT_an,label='GFT freespace')
#         ax.plot(hbar*omega_source/eC,Mon*1e21,label='bandwidth')
#         ax.set_title('Free space Green function')
#         ax.set_xlim(2,3)
#         ax.set_ylim((0, 1e20))
#         ax.set_xlabel('$\hbar\omega$ [eV]')
#         ax.set_ylabel('Green fct [m$^{-3}$]')
#         ax.legend()
#         plt.tight_layout()
#         #plt.savefig('Results/greenfct_benchmark.pdf')
#         plt.show()


#         "Plot frequency dependent 1D monitor"
#         fig,ax = plt.subplots(2,2,figsize=(9, 6))

#         #time dependent  values
#         ax[0,0].plot(pulsemon_t,label='pulse')
#         ax[0,0].plot(ez_source_t,label='field')
#         ax[0,0].set_title('Time domain')
#         ax[0,0].set_xlabel('Timestep')
#         ax[0,0].set_ylabel('Electric field')
#         ax[0,0].legend()

#         # #frequency domain
#         # ax[0,1].plot(hbar*omega_source/eC,np.real(pulsemon_om),label='pulse (real)')
#         # ax[0,1].plot(hbar*omega_source/eC,np.real(ez_source_om),label='field (real)')
#         # ax[0,1].plot(hbar*omega_source/eC,Mon*20000,label='bandwidth')
#         # ax[0,1].set_title('Freq domain, real part')
#         # ax[0,1].set_xlim(0,4)
#         # ax[0,1].set_xlabel('$\hbar\omega$ [eV]')
#         # ax[0,1].set_ylabel('Electric field')
#         # ax[0,1].legend()

#         #relative values to incident field
#         ax[1,0].plot(hbar*omega_source/eC,np.imag(pulsemon_om),label='pulse (imag)')
#         ax[1,0].plot(hbar*omega_source/eC,np.imag(ez_source_om),label='field (imag)')
#         ax[1,0].plot(hbar*omega_source/eC,Mon*20000,label='bandwidth')
#         ax[1,0].set_title('Freq domain, imag part')
#         ax[1,0].set_xlim(1.5,3.5)
#         ax[1,0].set_xlabel('$\hbar\omega$ [eV]')
#         ax[1,0].set_ylabel('Electric field')
#         ax[1,0].legend()

#         #Green function
#         ax[0,1].plot(hbar*omega_source/eC,np.imag(GFT)*1/ddx**3*1,label='GFT FDTD')
#         ax[0,1].plot(hbar*omega_source/eC,GFT_an,label='GFT freespace')
#         ax[0,1].plot(hbar*omega_source/eC,Mon*1e22,label='bandwidth')
#         ax[0,1].set_title('Green function')
#         ax[0,1].set_xlim(2,3)
#         ax[0,1].set_ylim((0, 6e21))
#         ax[0,1].set_xlabel('$\hbar\omega$ [eV]')
#         ax[0,1].set_ylabel('Green fct [m$^{-3}$]')

#         ax[0,1].legend()

#         #LDOS
#         ax[1,1].plot(hbar*omega_source/eC,np.imag(GFT)/ddx**3/GFT_an,label='LDOS')
#         #ax[1,1].plot(hbar*omega_source/eC,Mon*1e21,label='bandwidth')
#         ax[1,1].set_title('LDOS/Purcell factor, not sure')
#         ax[1,1].set_xlim(2,3)
#         ax[1,1].set_ylim((0, 50))
#         ax[1,1].set_xlabel('$\hbar\omega$ [eV]')

#         ax[1,1].legend()
#         plt.tight_layout()
#         #plt.savefig('Results/greenfct_ldos.pdf')
#         plt.show()


"Plot DFT solution"
# "Plot DFT Fields if graph flag is set"
# if (DFT_FLAG == 1):

# # test one of these - can adjust as apporpriate
#     EzReDFT2=np.abs(EzReDFT[1,:,:,:]+1j*EzImDFT[1,:,:,:])
#     S = (EzReDFT2)/np.max(EzReDFT2)

#     fig = plt.figure(figsize=(8,8))
#     ax = fig.add_axes([.16, .16, .6, .6])
# # quick example of second one - add and set w as required from LDOS picture
#     cf=ax.contourf(S[npml:Xmax-npml,jb+2,npml:Zmax-npml])
#     fig.colorbar(cf, ax=ax, shrink=0.8)
#     ax.set_aspect('equal')
#     im=ax.contour(S[npml:Xmax-npml,jb+2,npml:Zmax-npml]) # add more contour line
# #    CS2 = plt.contour(im, levels=im.levels[::2],colors='r')
#     ax.set_xlabel('Grid Cells ($y$)')
#     ax.set_ylabel('Grid Cells ($x$)')
#     ax.set_title('Example DFT plot of $E_x(x,y,\omega_2)$',y=1.05)

# # dielectric box
#     plt.vlines(X1,Y1,Y2,colors='b',lw=2)
# #set_linewidth(2)
#     plt.vlines(X2,Y1,Y2,colors='b',lw=2)
#     plt.hlines(Y1,X1,X2,colors='b',lw=2)
#     plt.hlines(Y2,X1,X2,colors='b',lw=2)

# PML box
# plt.vlines(npml,npml,Ymax-npml,colors='r',lw=2)
# plt.vlines(Xmax-npml,npml,Ymax-npml,colors='r',lw=2)
# plt.hlines(npml,npml,Xmax-npml,colors='r',lw=2)
# plt.hlines(Ymax-npml,npml,Xmax-npml,colors='r',lw=2)
# plt.show()


"1D monitors"
# fig = plt.figure(figsize=(14, 6))
# for i in range(n_mon):
#     ax = fig.add_subplot(2, 3,i+1)
#     ax.plot(ez_mon[i,:])
# plt.subplots_adjust(bottom=0.05, left=0.05)
# plt.show()

# "FFT"
# fft_res = 20 # pads with zeros for better resolution
# N = tsteps*fft_res
# ex_mon_om = np.zeros([n_mon,int(N/2+1)])
# ey_mon_om = np.zeros([n_mon,int(N/2+1)])
# ez_mon_om = np.zeros([n_mon,int(N/2+1)])
# hx_mon_om = np.zeros([n_mon,int(N/2+1)])
# hy_mon_om = np.zeros([n_mon,int(N/2+1)])
# hz_mon_om = np.zeros([n_mon,int(N/2+1)])
# for i in range(n_mon):
#     ez_mon_om[i,:] = rfft(ez_mon[i,:],n=N)
# nu = rfftfreq(N, dt)
# omega = 2*np.pi*nu/tera

# fig = plt.figure(figsize=(14, 6))
# for i in range(n_mon):
#     ax = fig.add_subplot(2, 3,i+1)
#     ax.plot(np.abs(ez_mon_om[i,:]))
#     ax.set_xlim(0,250)
# plt.subplots_adjust(bottom=0.05, left=0.05)
# plt.show()
