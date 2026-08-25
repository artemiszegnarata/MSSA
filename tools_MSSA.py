# ===========================================================================
# This code contains some tools useful for the MSSA computation and data use
# author: artemis zr
# started: 18/07/2026
# ===========================================================================

import pandas as pd
import numpy as np

def save_timeseries_as_csv(ts,time_axis,M,title,varset_name):
    ts_to_save = pd.DataFrame({'date': time_axis, 'value': ts})
    ts_to_save.set_index('date',inplace=True)
    ts_to_save.to_csv(f"/Users/azr/Desktop/MSSA/data/timeseries/M_{M}/phases_{title}_{varset_name}.csv")

# How to use:
#ts_OHC_IAP_700_2000_RC123 = pd.read_csv("/Users/azr/Desktop/MSSA/data/timeseries/M_{M}/phases_RC123_OHC_IAP_700_2000.csv",index_col="date",parse_dates=True) 


def detrend_from_MSSA(RCs_by_var, pcs_list, n_vars, trend_modes=[0]): 
    ''' This function removes the trend (such as determined with the MSSA) from the original timeserie on which the MSSA was applied.
        Inputs:
            RCs_by_var: RCs resulting from the function separate_RCs_by_var
            pcs_list: data on which the MSSA is applied (PCs resulting from the pre-EOF step)
            n_vars: number of variables studied in the MSSA
            trend_modes: modes of the trend to be removed, e.g. [0], or [0,1], default = [0]
        Output: 
            pcs_detrended_list: same format as pcs_list, but with the trend modes removed. The original data can then be reconstructed from this. --> how? 
    '''

    pcs_detrended_list = []

    for ivar in range(n_vars):
        RCs_var = RCs_by_var[ivar]  # (N, D_eof, K)

        # Sum all "trend" RCs
        trend_pcs = np.sum(RCs_var[:, :, trend_modes], axis=2)  # (N, D_eof)

        # Original PCs (after pre-EOF step)
        original_pcs = pcs_list[ivar].values  # (N, D_eof)

        # Substraction of the trend modes from the original timeseries
        pcs_detrended = original_pcs - trend_pcs  # (N, D_eof)
        pcs_detrended_list.append(pcs_detrended)

    return pcs_detrended_list
