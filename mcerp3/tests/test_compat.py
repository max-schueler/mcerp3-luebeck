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
