#!/usr/bin/env python3
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


import tensorflow as tf
import numpy as np 
import matplotlib.pyplot as plt 
import xlrd
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

DATA_FILE = 'data/firetheft.xls'

#Step 1. Read in the data
book = xlrd.open_workbook(DATA_FILE, encoding_override = "utf-8")
sheet = book.sheet_by_index(0)

print(sheet)