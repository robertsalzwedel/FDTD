from argparse import ArgumentParser


def get_user_input():
    parser = ArgumentParser()
    parser.add_argument(
        "--pulse",
        "-p",
        help="Choice: [THz, Optical]",
        choices=["THz", "Optical"],
        default="Optical",
    )

    parser.add_argument(
        "-ddx",
        help="Provide spatial resolution in nm.",
        type=int,
        default=10,
    )

    parser.add_argument(
        "-o",
        "--object",
        help="Choose object.",
        choices=["Sphere", "Rectangle", "None"],
        default="Sphere",
    )

    parser.add_argument(
        "-m",
        "--material",
        help="Choose Material.",
        choices=["Drude", "DrudeLorentz", "Etchegoin"],
        default="Drude",
    )

    parser.add_argument(
        "-r",
        "--radius",
        help="Define Radius of Sphere or Diameter of Rectangle",
        type=int,
        default=150,
    )

    parser.add_argument("-dim", help="Number of Grid Cells", type=int, default=50)

    parser.add_argument(
        "-npml", help="Number of PML Layers", choices=[8, 10, 12], type=int, default=8
    )

    parser.add_argument(
        "-eps_out",
        help="Permittivity surrounding the structure",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "-nsub",
        help="Number of Subgridding Cells",
        type=int,
        choices=[5, 7, 9],
        default=5,
    )

    parser.add_argument(
        "-t", "--tsteps", help="Number of time steps", type=int, default=5000
    )

    parser.add_argument(
        "-dft3d",
        help="Should the 3D DFT Monitor be activated",
        type=bool,
        default=False,
    )

    parser.add_argument(
        "-dft2d",
        help="Should the 2D DFT Monitor be activated",
        type=bool,
        default=False,
    )

    parser.add_argument(
        "--cross",
        help="Should the Cross Sections be monitored",
        type=bool,
        default=False,
    )

    parser.add_argument(
        "--boundary",
        "-b",
        help="What kind of boundary conditions should be implemented?",
        choices=["None", "PML", "PBC"],
        default="PML",
    )

    parser.add_argument(
        "--source",
        "-s",
        help="What source should be implemented?",
        choices=["None", "Point", "TFSF"],
        default="TFSF",
    )

    parser.add_argument(
        "--fft",
        help="Fast Fourier Transform",
        type=bool,
        default=False,
    )

    parser.add_argument(
        "--framerate",
        help="Number of Time Steps displayed in the Animation.",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--animate",
        help="Should a movie be displayed?",
        type=bool,
        default=True,
    )

    return parser.parse_args()
