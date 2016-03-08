#!/usr/bin/env python
# -*- coding: utf8 -*-

import numpy as np


def get_event_ind(pr):
    # Return the list of new events
    THRESHOLD = 0.01
    pr_diff = np.sum(np.absolute(pr[0:-1, :] - pr[1:, :]), axis=1)
    pr_event = (pr_diff > THRESHOLD).nonzero()
    return (pr_event[0] + np.ones(pr_event[0].shape[0], dtype=np.int))
