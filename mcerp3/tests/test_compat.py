#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  7 11:43:32 2026

@author: maximilian
"""

import pytest
import mcerp3
import scipy.stats as ss
import numpy as np

def test_binom_test_compat():
    """Test binom_test wrapper returns float p-values."""
    p_old_style = mcerp3.stats.binom_test(5, 10, 0.5)
    assert isinstance(p_old_style, (float, np.ndarray))
    assert 0 <= p_old_style <= 1

def test_version():
    """Test __version__ exists."""
    assert hasattr(mcerp3, '__version__')
    assert isinstance(mcerp3.__version__, str)


def test_lhd_basic():
    """Test lhd with docstring example."""
    d0 = ss.uniform(loc=-1, scale=2)
    pts = mcerp3.lhd(dist=d0, size=5)
    assert pts.shape == (5, 1)
    assert np.all(pts >= -1) and np.all(pts <= 1)
    assert np.all(np.isfinite(pts))

def test_lhd_multiple_dims():
    """Test single dist → multiple vars."""
    d1 = ss.norm(loc=0, scale=1)
    pts = mcerp3.lhd(dist=d1, size=7, dims=3)
    assert pts.shape == (7, 3)
    assert np.all(np.isfinite(pts))