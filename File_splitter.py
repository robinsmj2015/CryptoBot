#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 20 15:47:54 2022

@author: robinson
"""

import File_Utils


file_nums = []

func = lambda l: 'altCB' + str(l) + '.pkl'
file_names = list(map(func, file_nums))
File_Utils.big_file_splitter(file_names)