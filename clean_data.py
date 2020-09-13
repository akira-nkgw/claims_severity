#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 13 16:02:59 2020

@author: akiranakagawa
"""
import pandas as pd


data = pd.read_csv('train.csv', error_bad_lines=False)

data.describe()