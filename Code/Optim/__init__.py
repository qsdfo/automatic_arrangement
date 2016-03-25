#!/usr/bin/env python
# -*- coding: utf8 -*-

import theano.tensor as T
from Models.Utils.init import shared_zeros
from collections import OrderedDict


class adam_L2(object):
    def __init__(self, b1=0.9, b2=0.999, alpha=0.001, epsilon=1e-8, **kwargs):
        self.b1 = b1
        self.b2 = b2
        self.alpha = alpha
        self.epsilon = epsilon

    def get_updates(self, param, grads):
        """
        From cle (Junyoung Chung toolbox)
        """
        updates = OrderedDict()
        i = shared_zeros(0., 'counter')
        i_t = i + 1.
        b1_t = self.b1 ** i_t
        b2_t = self.b2 ** i_t

        for p, g in zip(param, grads):
            m = shared_zeros(p.get_value() * 0.)
            # WOW PUtain
            # p.get_value ensure that at each iteration we keep the same node in the the graph, and not intialize a new one
            v = shared_zeros(p.get_value() * 0.)
            m_t = self.b1 * m + (1 - self.b1) * g
            v_t = self.b2 * v + (1 - self.b2) * g ** 2
            m_t_hat = m_t / (1. - b1_t)
            v_t_hat = v_t / (1. - b2_t)
            g_t = m_t_hat / (T.sqrt(v_t_hat) + self.epsilon)
            p_t = p - self.alpha * g_t
            updates[m] = m_t
            updates[v] = v_t
            updates[p] = p_t

        updates[i] = i_t

        return updates
