# ==========================================================================
# This code contains the functions for the preprocessing of the MSSA method
# author: artemis zr
# started: 15/07/2026
# ==========================================================================


import numpy as np
import xarray as xr
from eofs.xarray import Eof as eof
import pandas as pd
from windspharm.xarray import VectorWind
from scipy.linalg import hankel





def compute_helmholtz(u, v):
    """
    Compute streamfunction (psi) and velocity potential (phi)
    from wind components using spherical harmonics.
    """
    w = VectorWind(u, v) # can also be used to plot maps with the wind vectors? 
    psi, phi = w.sfvp()
    return psi, phi




def load_variable(cfg, NL, SL, WL, EL, start_year, end_year, ds_target_grid=None):
    ds = xr.open_dataset(cfg["path"])

    # --- Rename time, lat and lon if needed
    if "valid_time" in ds.dims:
        ds = ds.rename({"valid_time": "time"})
    if "latitude" in ds.dims:
        ds = ds.rename({"latitude": "lat"})
    if "longitude" in ds.dims:
        ds = ds.rename({"longitude": "lon"})

    # ensure lat lon similar to ERA5 (that is positive southward (decreasing latitudes) and positive eastward (increasing longitudes))
    if ds.lat.values[0] < ds.lat.values[-1]:
        ds = ds.sortby("lat", ascending=False)
    if ds.lon.values[0] > ds.lon.values[-1]:
        ds = ds.sortby("lon")

    # ensure longitudes in -180 --> 180 instead of 0 --> 360
    ds = ds.assign_coords(
        lon=(((ds.lon + 180) % 360) - 180)
    ).sortby("lon")

    # select time first
    ds = ds.sel(time=slice(f"{start_year}", f"{end_year}"))
    ds['time'] = pd.to_datetime(ds['time'].dt.year.astype(str)) # ensure the time format are also matching across variables

    if cfg.get("helmholtz"):

        u = ds[cfg["u"]]
        v = ds[cfg["v"]]

        psi, phi = compute_helmholtz(u, v) # HERE I HAVE THE POSSIBILITY TO ALSO USE PHI ! 

        psi = psi.sel(lat = slice(NL, SL), lon = slice(WL, EL))
        da = psi

    elif cfg.get("net_calculation"): 
        da = ds.avg_slhtf+ds.avg_ishf+ds.avg_snswrf+ds.avg_snlwrf
        
    else:
        da = ds[cfg["field"]]
        #da = da.sel(time=slice(f"{start_year}", f"{end_year}"), lat = slice(NL, SL), lon = slice(WL, EL))

    # subset region
    da = da.sel(lat=slice(NL, SL), lon=slice(WL, EL))

    # this is to change the grids of ERA5 in a 1° grid copied on the OHC dataset 
    if cfg.get("fix_grid"):
        if ds_target_grid is not None:
            target = ds_target_grid.sel(lat=slice(NL, SL), lon=slice(WL, EL))
            print("TARGET SHAPE:", target.lat.size, target.lon.size)

            # ensure no time dimension
            if "time" in target.dims:
                target = target.isel(time=0, drop=True)
            if "time" in target.coords:
                target = target.drop_vars("time")
            da = da.coarsen(lat=4, lon=4, boundary="trim").mean()
            #da, target = xr.align(da, target, join="exact")
            #da = da.interp(lat=target.lat, lon=target.lon)
            da = da.reindex(
                lat=target.lat,
                lon=target.lon,
                method="nearest",
                tolerance=0.6
            )
            # Force coordinates to match target grid
            da = da.assign_coords(lat=target.lat, lon=target.lon)
        else:
            print("ds_target_grid is missing")

    return da



def check_common_grid(data_arrays):
    ref = data_arrays[0]
    for i, da in enumerate(data_arrays[1:], start=1):
        if not np.array_equal(ref.lat.values, da.lat.values):
            raise ValueError(
                f"Latitude mismatch between variable 0 and variable {i}"
            )

        if not np.array_equal(ref.lon.values, da.lon.values):
            raise ValueError(
                f"Longitude mismatch between variable 0 and variable {i}"
            )

        if ref.shape[1:] != da.shape[1:]:
            raise ValueError(
                f"Spatial shape mismatch between variable 0 {ref.shape} and variable {i} {da.shape}"
            )
        
        if not np.array_equal(ref.time.values, da.time.values):
            raise ValueError("Time mismatch")

    print("All variables share the exact same grid.")


def prepare_anomalies(da):
    mean = da.mean("time")
    anom = da - mean
    std = anom.std("time")
    norm = anom / std
    return norm, std



def compute_preEOF(da, D_eof=50):
    solver = eof(da)
    eofs = solver.eofs(neofs=D_eof)
    pcs = solver.pcs(npcs=D_eof, pcscaling=0)
    varfrac = solver.varianceFraction(neigs=D_eof)
    expvar = varfrac * 100
    return pcs, eofs, expvar



def embedding(data, M):
    """
    Construct the trajectory (Hankel) matrix for MSSA.

    Parameters
    ----------
    data : np.ndarray or xarray.DataArray
        Shape (N, D): time × variables (PCs after pre-EOF)
    M : int
        Embedding window length

    Returns
    -------
    Xemb : np.ndarray
        Trajectory matrix of shape (N-M+1, M*D)
    """

    # Convert xarray → numpy if needed
    if hasattr(data, "values"):
        data = data.values

    N, D = data.shape

    # Hankel index matrix
    c = np.arange(0, N - M + 1)
    r = np.arange(N - M, N)
    idx = hankel(c, r)

    Xemb = np.zeros((N - M + 1, M * D))

    for d in range(D):
        x = data[:, d]
        Xemb[:, d*M:(d+1)*M] = x[idx]

    return Xemb