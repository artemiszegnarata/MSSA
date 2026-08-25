# ======================================================================
# This code contains the functions for the different plots for the MSSA
# author: artemis zr
# started: 17/07/2026
# ======================================================================


import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
sns.set_theme(style="whitegrid")
import cmocean

import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

from matplotlib.colors import CenteredNorm
from matplotlib.cm import ScalarMappable


import sys
sys.path.append('/Users/azr/Desktop/MSSA/scripts/')
from mssa_PC_RC import reconstruct_RCs_group, compute_mssa_phase_from_RC_field, phase_indices_from_phin


def plotting_eigenvalues(expvar,title,varset_name,varname,saving_path=None,freq=None):
    '''Produce the diverse plots for eigenvalues (preMSSA and postMSSA)
       Inputs
       expvar:
       title:
       saving_path: default is None and if not None, the fig is saved
       freq: default = None (index), if not None -> eigenvalues wrt freq
    '''
    fig, ax = plt.subplots(figsize=(12, 4))
    if freq is not None: 
        ax.stem(freq, expvar)
        ax.set_xlabel("Frequency (cycles/year)")
        # Annotate each point with its mode index
        for i, (x, y) in enumerate(zip(freq, expvar)):
            ax.annotate(
                str(i + 1),                  # label = mode index
                (x, y),                      # position
                textcoords="offset points",
                xytext=(3, 3),               # small offset for readability
                fontsize=8,
                color='darkblue'
            )
    else:
        ax.stem(expvar)
        ax.set_xlabel("Index")
    ax.set_title(f"Eigenvalues {title}")
    ax.set_ylabel("% of variance explained")
    plt.tight_layout()
    if saving_path is not None:
        if freq is not None: 
            plt.savefig(f"{saving_path}/eigval_{title}_freq_{varset_name}_{varname}.png")
        else:
            plt.savefig(f"{saving_path}/eigval_{title}_index_{varset_name}_{varname}.png")
    plt.show()
    plt.close()


def plotting_EOFmaps(eofs,expvar,cfg,title,varset_name,varname,saving_path=None):
    fig, ax = plt.subplots(2, 2, figsize=(14, 11))
    ax = ax.flatten()

    for m in range(4):
        eofs.sel(mode=m).plot(ax=ax[m], cbar_kwargs={'label': cfg.get("unit")})
        ax[m].set_title(f"EOF {m+1}: {int(expvar[m])}% - {title}")

    plt.tight_layout()
    if saving_path is not None:
        plt.savefig(f"{saving_path}/EOFmaps_{title}_{varset_name}_{varname}.png")
    plt.show()
    plt.close()


def plotting_PCs(PCs,freq_dom,expvar,M,varset_name,saving_path=None):
    plt.figure(8, figsize=(10, 20))
    plt.suptitle('Principal Components (PCs)')

    for m in range(16):
        ax = plt.subplot(16, 1, m + 1)
        #ax.plot(PCs[:, m], 'k-')
        sns.lineplot(PCs[:, m],ax=ax,color="#0F90B5",linewidth=1.5)
        ax.set_ylabel(f'PC {m + 1}')
        plt.xlim((0,PCs.shape[0]-1))

        # --- Add text at top-right corner (axes coordinates)
        ax.text(
            0.98, 0.90, f"T = {np.round(1/freq_dom[m],1)} yrs, expvar = {np.round(expvar[m],1)}%", size=10,
            ha="right", va="top",
            transform=ax.transAxes,  # important: use axes coordinates
            bbox=dict(
                boxstyle="square",
                ec=(1., 0.5, 0.5),
                fc=(1., 0.8, 0.8, 0.5)
            )
        )

    plt.tight_layout(rect=[0, 0, 1, 0.98])  # leave space for title
    if saving_path is not None:
        plt.savefig(f"{saving_path}/M_{M}_PCs_{varset_name}.png")
    plt.show()
    plt.close()




# --- RECONSTRUCTED COMPONENTS 

# --- Plot the timeseries of the original data, along with the RC1 and the sum of the N first RCs (with N=nb_RCs)

def plot_timeseries_comparison(timeseries_list, labels, time_axis,
                                colors=None, linestyles=None, cmap=None,
                                title=None, ylabel=None,
                                legend=True):
    """
    Plot plusieurs timeseries sur le même axe pour les comparer.

    Parameters
    ----------
    timeseries_list : list of 1D array-like, of length N (same as time_axis)
    labels : list of str, legend for each timeseries 
    time_axis : array-like (datetime)
    colors : list, optional, if None there are generated automatically from cmap
    linestyles : list of str, optional, default: '-' for the first and '--' for the other ones
    title, ylabel : str, optional
    legend : bool, default True

    Returns
    -------
    ax : matplotlib.axes.Axes
    """
    n = len(timeseries_list)

    _, ax = plt.subplots(figsize=(14, 6))

    if colors is None:
        if cmap is not None:
            cmap_obj = plt.get_cmap(cmap)
            colors = [cmap_obj(i / max(n - 1, 1)) for i in range(n)]
        else:
            colors = [None] * n  # laisse matplotlib gérer le cycle de couleurs

    if linestyles is None:
        linestyles = ['-'] + ['--'] * (n - 1)

    for ts, lab, col, ls in zip(timeseries_list, labels, colors, linestyles):
        ax.plot(time_axis, ts, color=col, linestyle=ls, label=lab)

    ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y'))
    ax.xaxis.set_major_locator(plt.matplotlib.dates.YearLocator(5))
    ax.set_xlim(time_axis[0], time_axis[-1])

    if title:
        ax.set_title(title, fontsize=20)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=14)
    if legend:
        ax.legend()

    return ax


def plot_RCs_by_variable(RCs_by_var, eofs_maps, var_std, var_names, units_list,
                          time_axis, groups, saving_path=None, title=None, colors=None):
    """
    For each variable, calculate different groups of RCs (ex: RC1, RC1+2, all RCs)
    and compare them in one plot using plot_timeseries_comparison.

    Parameters
    ----------
    groups : list of (label, rc_indices or None)
        rc_indices=None -> all RCs (equivalent to the original timeserie)
        Example:
            [("Original (all RCs)", None),
             ("RC1", [1]),
             ("First 3 RCs", [1, 2, 3])]
    """
    nvar = len(RCs_by_var)
    default_colors = ['k', '#C5230C', '#0F90B5', '#65B50F', '#9B59B6']
    colors = colors or default_colors

    for i in range(nvar):
        n_modes_total = RCs_by_var[i].shape[-1]
        timeseries_list, labels = [], []

        for label, rc_idx in groups:
            if rc_idx is None:
                rc_idx = list(range(1, n_modes_total + 1))
            _, ts, _ = reconstruct_RCs_group(RCs_by_var[i], eofs_maps[i], var_std[i], rc_idx)
            timeseries_list.append(ts)
            labels.append(label)

        plot_timeseries_comparison(
            timeseries_list, labels, time_axis,
            colors=colors[:len(labels)],
            title=f'Reconstructed components RCs - {var_names[i]}',
            ylabel=units_list[i]
        )

    plt.tight_layout()
    if saving_path:
        plt.savefig(f"{saving_path}/timeseries_comp_{title}.png")
    plt.show()
    plt.close()



# --- Plot the timeserie for the selected RCs, along with their spatial decomposition in different phases

def plot_RCs_phases(
    VARs,                  # list of reconstructed fields: [VAR1, VAR2?]
    var_names,             # list of variable names: ['SST', 'SLP']
    expvars,               # list of explained variance
    units,                 # list of units for each variable
    lats,
    lons,
    time_axis,
    M,
    varset_name,
    RCs_group=None,        # temporal RCs for phase decomposition, shape: (D_group, time)
    title='default_RCs',           # plot title prefix
    period_name='default_period',           # plot period in title
    saving_path=None
):
    """
    Plot reconstructed RCs with automatic 4-phase decomposition based on oscillation.

    Parameters
    ----------
    VARs : list of np.ndarray
        Each element: (lat, lon, time)
    var_names : list of str
    expvars : list of float
    units : list of str
    RCs_group : np.ndarray or None
        Temporal RCs to compute synthetic oscillation, shape (D_group, time)
        If None, phases will be just 4 quarters of time
    """

    n_var = len(VARs)
    #lat, lon, N = VARs[0].shape
    N = VARs[0].shape[2]

    # ---------- COLORS ----------
    cmap = plt.get_cmap("rainbow")
    rainbow = cmap(np.linspace(0, 1, 4))[:, :3]
    mycolor = cmocean.cm.balance
    #mycolor = cm.vik
    #mycolor = custom_cmap
    proj = ccrs.PlateCarree()
    colors_var=["#C5230C","#0F90B5","#36B50F","#B5800F","#000000","#6B22D1"]

    # ---------- PHASES ----------
    if RCs_group is not None:
        # Convert (lat, lon, time) -> (time, lat, lon)
        VAR_T = np.transpose(VARs[0], (2, 0, 1))
        phin = compute_mssa_phase_from_RC_field(VAR_T)
        phases = phase_indices_from_phin(phin, n_phases=4)
        
    else:
        # no RCs_group -> just 4 time quarters
        phases = [np.arange(i*N//4,(i+1)*N//4) for i in range(4)]

    # ---------- TIME SERIES ----------
    #fig_ts, ax_ts = plt.subplots(figsize=(14,4))
    #axes_ts = [ax_ts]
    #if n_var > 1:
    #    ax2 = ax_ts.twinx()
    #    axes_ts.append(ax2)
    fig_ts, axes_ts = plt.subplots(n_var, 1, figsize=(14, 3*n_var), sharex=True)
    axes_ts = np.atleast_1d(axes_ts)
    for i, VAR in enumerate(VARs):
        ts = np.nanmean(VAR, axis=(0,1))
        #axes_ts[i].plot(time_axis, ts, label=var_names[i],color=colors_var[i])
        sns.lineplot(x=time_axis,y=ts,ax=axes_ts[i],label=var_names[i],color=colors_var[i],legend=False)
        axes_ts[i].set_ylabel(f' {var_names[i]} ({units[i]})',color=colors_var[i])
        axes_ts[i].set_xlabel('Years')
        axes_ts[i].set_xlim(time_axis[0],time_axis[-1])

    for k,ii in enumerate(phases): 
        # USE LINEPLOT TO HAVE MARKER ABOVE THE PLOT AND SCATTER TO HAVE THEM BEHIND
        # marker can also be 'o' to have small circles 
        
        sns.lineplot(x=time_axis[ii],y=np.nanmean(VARs[0], axis=(0,1))[ii],ax=axes_ts[0],marker='D',markeredgecolor=rainbow[k],markerfacecolor=rainbow[k],linestyle='',markersize=4,color=rainbow[k],legend=False)
        #axes_ts[0].scatter(time_axis[ii],np.nanmean(VARs[0], axis=(0,1))[ii],color=rainbow[k],s=15,marker='D')
        
    #axes_ts[0].set_title(f"{title}. {', '.join([f'{v:.1f}%' for v in expvars])} of explained variance", fontsize=20)
    axes_ts[0].set_title(f"{title} - {int(np.round(expvars))}% of explained variance - {period_name}", fontsize=15, fontweight='bold')


    fig_ts.tight_layout()
    if saving_path is not None:
        fig_ts.savefig(f"{saving_path}/timeseries_{title}_{varset_name}.png")
    

    # ---------- SPATIAL COMPOSITES ----------
    fig_map, axes_map = plt.subplots(n_var,4, figsize=(16,9), subplot_kw={'projection': proj}, squeeze=False, constrained_layout=True)
    #axes_map = axes_map.reshape(n_var,4)

    for i, VAR in enumerate(VARs):
        # Compute fields for this variable
        fields = [np.nanmean(VAR[:,:,idx], axis=2) for idx in phases]

        # Variable-specific normalization 
        global_max = max(np.nanmax(np.abs(f)) for f in fields)
        #global_max = 1.5e+19
        print("global max:",global_max)
        norm = CenteredNorm(vcenter=0, halfrange=global_max)

        # Plot phases
        for j in range(4):
            ax = axes_map[i,j]
            cf = ax.contourf(lons[i], lats[i], fields[j], levels=60, cmap=mycolor, norm=norm, transform=proj)
            #cf = ax.pcolormesh(lon, lat, fields[j], cmap=mycolor, norm=norm, transform=proj, shading='auto')

            #print("np.nanmin, np.nanmax",np.nanmin(fields[j]), np.nanmax(fields[j]))
            #print("norm.vmin, norm.vmax",norm.vmin, norm.vmax)

            ax.coastlines(resolution='110m', linewidth=0.6)
            ax.add_feature(cfeature.BORDERS, linestyle=':')
            ax.add_feature(cfeature.LAND, facecolor='lightgray', alpha=0.4)
            gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
            gl.top_labels = False
            gl.right_labels = False
            gl.xlabel_style={'size':8}
            gl.ylabel_style={'size':8}
            gl.xformatter = LONGITUDE_FORMATTER
            gl.yformatter = LATITUDE_FORMATTER
            gl.xlocator = plt.MultipleLocator(30)
            gl.ylocator = plt.MultipleLocator(15)
            ax.set_title(f"{var_names[i]} Phase {j+1}", color=rainbow[j],fontsize=18)

        # Create one colobar per row
        sm = ScalarMappable(norm=norm, cmap=mycolor)
        sm.set_array([])

        cbar = fig_map.colorbar(sm, ax=axes_map[i,:], orientation='vertical', fraction=0.02, pad=0.02, aspect=8)
        cbar.set_label(units[i],fontsize=10)
        cbar.ax.tick_params(labelsize=8)

    if saving_path is not None:
        fig_map.savefig(f"{saving_path}/phases_{title}_{varset_name}.png")
    plt.show()




