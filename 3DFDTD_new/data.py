import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import json

from modules.parameters import *


def store_point(
    FLAG,
    sphere,
    ddx,
    dt,
    tsteps,
    dims,
    pulse,
    dft,
    npml,
    eps_in,
    eps_out,
    source,
    pulsemon_t,
    ez_source_t,
    start,
    stop,
):

    filename = "point_object{}_r{}_dx{}_T{}_X{}_lam{}_wdt{}_nfreq{}_xs{}_ys{}_zs{}_npml{}_eps{}_final_new".format(
        FLAG.OBJECT,
        int(sphere.R / nm),
        int(ddx / nm),
        tsteps,
        dims.x,
        int(pulse.lam_0 / nm),
        int(pulse.width),
        dft.iwdim,
        source.x,
        source.y,
        source.z,
        npml,
        eps_in,
    )

    time = np.arange(0, tsteps * dt, dt)
    pointmonitors = pd.DataFrame(columns=["time", "pulse", "field"])

    pointmonitors["time"] = time
    pointmonitors["pulse"] = pulsemon_t
    pointmonitors["field"] = ez_source_t
    pointmonitors.to_pickle("Results/" + filename + ".pkl")

    custom_meta_content = {
        "object": FLAG.OBJECT,
        "sphere": [sphere.R, sphere.x, sphere.y, sphere.z],
        "dx": ddx,
        "dt": dt,
        "timesteps": tsteps,
        "grid": dims.x,
        "eps_in": eps_in,
        "eps_out": eps_out,
        "pulsewidth": pulse.width,
        "delay": pulse.t0,
        "lambda": pulse.lam_0 / nm,
        "nfreq": dft.iwdim,
        "source_loc": [source.x, source.y, source.z],
        "npml": npml,
        "runtime": stop - start,
    }

    custom_meta_key = "pointsource.iot"
    table = pa.Table.from_pandas(pointmonitors)
    custom_meta_json = json.dumps(custom_meta_content)
    existing_meta = table.schema.metadata
    combined_meta = {
        custom_meta_key.encode(): custom_meta_json.encode(),
        **existing_meta,
    }
    table = table.replace_schema_metadata(combined_meta)
    pq.write_table(table, "Results/" + filename + ".parquet", compression="GZIP")


def store_cross(
    FLAG,
    sphere,
    ddx,
    dt,
    tsteps,
    dims,
    pulse,
    dft,
    npml,
    eps_in,
    eps_out,
    start,
    stop,
    S_scat_total,
    S_abs_total,
    SourceReDFT,
    SourceImDFT,
    wp,
    gamma,
    tfsf_dist,
):

    filename = (
        "TFSF_object{}_r{}_dx{}_T{}_X{}_lam{}_wdt{}_nfreq{}_npml{}_eps{}_tfsf{}".format(
            FLAG.OBJECT,
            int(sphere.R / nm),
            int(ddx / nm),
            tsteps,
            dims.x,
            int(pulse.lam_0 / nm),
            int(pulse.width),
            dft.iwdim,
            npml,
            eps_in,
            tfsf_dist,
        )
    )

    crosssections = pd.DataFrame(columns=["omega", "sigma_scat", "sigma_abs"])

    crosssections["omega"] = dft.omega
    crosssections["lambda"] = dft.lam
    crosssections["nu"] = dft.nu
    crosssections["sigma_scat"] = S_scat_total
    crosssections["sigma_abs"] = S_abs_total
    # crosssections["bandwidth"] = Source
    crosssections["source_re"] = SourceReDFT
    crosssections["source_im"] = SourceImDFT
    # only if we want a pure pickle file
    # crosssections.to_pickle('Results/'+filename+'.pkl')

    custom_meta_content = {
        "object": FLAG.OBJECT,
        "sphere": [sphere.R, sphere.x, sphere.y, sphere.z],
        "dx": ddx,
        "dt": dt,
        "timesteps": tsteps,
        "grid": dims.x,
        "eps_in": eps_in,
        "eps_out": eps_out,
        "wp": wp,
        "gamma": gamma,
        "pulsewidth": pulse.width,
        "delay": pulse.t0,
        "lambda": pulse.lam_0 / nm,
        "nfreq": dft.iwdim,
        "npml": npml,
        "tfsf_dist": tfsf_dist,
        "runtime": stop - start,
    }

    custom_meta_key = "TFSFsource.iot"
    table = pa.Table.from_pandas(crosssections)
    custom_meta_json = json.dumps(custom_meta_content)
    existing_meta = table.schema.metadata
    combined_meta = {
        custom_meta_key.encode(): custom_meta_json.encode(),
        **existing_meta,
    }
    table = table.replace_schema_metadata(combined_meta)
    pq.write_table(table, "Results/" + filename + ".parquet", compression="GZIP")


def store_periodic(
    FLAG,
    sphere,
    ddx,
    dt,
    tsteps,
    dims,
    pulse,
    dft,
    npml,
    eps_in,
    eps_out,
    start,
    stop,
    SourceReDFT,
    SourceImDFT,
    wp,
    gamma,
    tfsf_dist,
    transmission,
    reflection,
):

    filename = "periodic_object{}_material{}_r{}_dx{}_T{}_X{}_lam{}_wdt{}_nfreq{}_npml{}_tfsf{}".format(
        FLAG.OBJECT,
        FLAG.MATERIAL,
        int(sphere.R / nm),
        int(ddx / nm),
        tsteps,
        dims.x,
        int(pulse.lam_0 / nm),
        int(pulse.width),
        dft.iwdim,
        npml,
        tfsf_dist,
    )

    periodic = pd.DataFrame(columns=["omega", "ref", "trans"])

    periodic["omega"] = dft.omega
    periodic["lambda"] = dft.lam
    periodic["nu"] = dft.nu
    periodic["trans"] = transmission
    periodic["ref"] = reflection
    periodic["source_re"] = SourceReDFT
    periodic["source_im"] = SourceImDFT
    # only if we want a pure pickle file
    # periodic.to_pickle('Results/'+filename+'.pkl')

    custom_meta_content = {
        "object": FLAG.OBJECT,
        "sphere": [sphere.R, sphere.x, sphere.y, sphere.z],
        "dx": ddx,
        "dt": dt,
        "timesteps": tsteps,
        "grid": dims.x,
        "eps_in": eps_in,
        "eps_out": eps_out,
        "wp": wp,
        "gamma": gamma,
        "pulsewidth": pulse.width,
        "delay": pulse.t0,
        "lambda": pulse.lam_0 / nm,
        "nfreq": dft.iwdim,
        "npml": npml,
        "tfsf_dist": tfsf_dist,
        "runtime": stop - start,
    }

    custom_meta_key = "periodicsource.iot"
    table = pa.Table.from_pandas(periodic)
    custom_meta_json = json.dumps(custom_meta_content)
    existing_meta = table.schema.metadata
    combined_meta = {
        custom_meta_key.encode(): custom_meta_json.encode(),
        **existing_meta,
    }
    table = table.replace_schema_metadata(combined_meta)
    pq.write_table(table, "Results/" + filename + ".parquet", compression="GZIP")


def store_periodic_micro(
    FLAG,
    sphere,
    ddx,
    dt,
    tsteps,
    dims,
    pulse,
    dft,
    npml,
    eps_in,
    eps_out,
    start,
    stop,
    SourceReDFT,
    SourceImDFT,
    tfsf_dist,
    transmission,
    reflection,
):

    filename = "periodic_micro_object{}_material{}_r{}_dx{}_T{}_X{}_lam{}_wdt{}_nfreq{}_npml{}_tfsf{}".format(
        FLAG.OBJECT,
        FLAG.MATERIAL,
        int(sphere.R / nm),
        int(ddx / nm),
        tsteps,
        dims.x,
        int(pulse.lam_0 / nm),
        int(pulse.width),
        dft.iwdim,
        npml,
        tfsf_dist,
    )

    periodic = pd.DataFrame(columns=["omega", "ref", "trans"])

    periodic["omega"] = dft.omega
    periodic["lambda"] = dft.lam
    periodic["nu"] = dft.nu
    periodic["trans"] = transmission
    periodic["ref"] = reflection
    periodic["source_re"] = SourceReDFT
    periodic["source_im"] = SourceImDFT
    # only if we want a pure pickle file
    # periodic.to_pickle('Results/'+filename+'.pkl')

    custom_meta_content = {
        "object": FLAG.OBJECT,
        "sphere": [sphere.R, sphere.x, sphere.y, sphere.z],
        "dx": ddx,
        "dt": dt,
        "timesteps": tsteps,
        "grid": dims.x,
        "eps_in": eps_in,
        "eps_out": eps_out,
        "pulsewidth": pulse.width,
        "delay": pulse.t0,
        "lambda": pulse.lam_0 / nm,
        "nfreq": dft.iwdim,
        "npml": npml,
        "tfsf_dist": tfsf_dist,
        "runtime": stop - start,
    }

    custom_meta_key = "periodicsource.iot"
    table = pa.Table.from_pandas(periodic)
    custom_meta_json = json.dumps(custom_meta_content)
    existing_meta = table.schema.metadata
    combined_meta = {
        custom_meta_key.encode(): custom_meta_json.encode(),
        **existing_meta,
    }
    table = table.replace_schema_metadata(combined_meta)
    pq.write_table(table, "Results/" + filename + ".parquet", compression="GZIP")
