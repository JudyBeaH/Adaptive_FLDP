#Step1
from Global_Parameter import *
import tensorflow as tf
import numpy as np
import csv
import time
import struct
import pickle
import copy
import os


class EPS_round:
    def __init__(self,vali_x,vali_y):
        self.vali_x = vali_x
        self.vali_y = vali_y
    
    def RoundlyAccount(old_global_model,eps_global,t): #计算得该轮的隐私预算
        eps_round = 0
        #
        #
        #
        return  eps_round
    
    