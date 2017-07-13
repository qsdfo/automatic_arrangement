#!/usr/bin/env python
# -*- coding: utf8 -*-


class Categorical_lop_model(object):
    """
    Template class for the lop models.
    Contains plot methods
    """

    def __init__(self, model_param, dimensions, checksum_database):
        # Add number of categories
        self.N_Cat = model_param['number_category']
        return
