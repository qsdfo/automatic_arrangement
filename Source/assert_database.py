#!/usr/bin/env python
# -*- coding: utf8 -*-


def assert_database(metadata, script_param):
    assert metadata["temporal_granularity"] == script_param['temporal_granularity']
    assert metadata["quantization"] == script_param['quantization']
    assert metadata["max_translation"] == script_param['max_translation']
    assert metadata["unit_type"] == script_param['unit_type']
