# Small function to smooth the dynamic vector computed when parsing an MXML file
# Horizon is used after a spanning dynamic to see if a static dynamic is defined.
# It is given in number of quarter notes

import numpy as np


def smooth_dyn(dynamic, flag, quantization, horizon):
    T = dynamic.shape[0]
    for t in range(0, T):
        if t in flag.keys():
            if (flag[t] == 'Cresc_start') or (flag[t] == 'Dim_start'):
                start_dyn = dynamic[t]
                start_time = t
            elif flag[t] == 'Cresc_stop':
                for h in range(0, T):
                    if flag[t + h] == 'N':
                        stop_dyn = dynamic[t + h]
                        # Just a little check
                        if stop_dyn < start_dyn:
                            stop_dyn = min(start_dyn + 0.5, 1)
                    else:
                        stop_dyn = min(start_dyn + 0.5, 1)
                dynamic[start_time:t] = np.linspace(start_dyn, stop_dyn, t - start_time)
            elif flag[t] == 'Dim_stop':
                for h in range(0, T):
                    if flag[t + h] == 'N':
                        stop_dyn = dynamic[t + h]
                        # Just a little check
                        if stop_dyn > start_dyn:
                            stop_dyn = max(start_dyn - 0.5, 0)
                    else:
                        stop_dyn = max(start_dyn - 0.5, 0)
                dynamic[start_time:t] = np.linspace(start_dyn, stop_dyn, t - start_time)

    return dynamic
