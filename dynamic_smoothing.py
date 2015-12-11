#!/usr/bin/env python
# -*- coding: utf8 -*-

import numpy as np


def dynamics_smoothing(dynamics, spanning_flag, quantization):
    start_span = 0
    stop_span = 0
    check_span = quantization * 1   # One quarter note
    for time in dynamics.shape[0]:
        if spanning_flag[time] == 'start':
            start_span = time
        if spanning_flag[time] == 'stop':
            stop_span = time
            for time_forward in np.arange(time + 1, time + check_span):
                if dynamics[time_forward] != dynamics[time_forward - 1]:
                    # It's a jump motherfucker = linearly interpolate
                    end_dyn = dynamics[time_forward]
                    start_dyn = dynamics[start_span]
                    dynamics[start_span:stop_span] = np.linspace(start_dyn, end_dyn, start_span - stop_span)
                    break
                if spanning_flag[time]:
                    # Beginning of another spanning dynamic. Just get out of this loop
                    break

# if __name__ == '__main__':
