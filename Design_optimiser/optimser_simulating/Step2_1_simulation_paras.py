# @Author : JM_Huang
# @Time   : 28/09/21
import os
import sys
import torch
import numpy as np
from my_utils import init_trained_rnn, prepare_input_matx, convert_ratio_to_ms
from argparse import ArgumentParser


def get_default_para():
    args = ArgumentParser().parse_args()
    args.saved_ms_fpa = '/home/tech/Workspace/Projects/Facelift/optimiser/optimser_simulating/ws_dstore/target_model_whole_ms.npy'
    args.saved_rnn_fpa = '/home/tech/Workspace/Projects/Facelift/optimiser/optimser_simulating/ws_dstore/rnn_model_epoch350'
    return args

def get_astra_settings():
    args = get_default_para()

    args.maker = 'Vauxhall'
    args.genmodel= 'Astra'
    args.base_year = 2001
    args.input_pre_extend = 6
    args.simulation_start_year = 2011
    args.simulation_end_year = 2015
    args.ori_facelift_year = 2012
    args.base_aes = 1.280
    args.ori_facelift_aes = 1.2076844
    args.given_aes_rises_l = [1.268-args.base_aes, 1.307-args.base_aes]

    return args


def get_audi_settings():
    args = get_default_para()
    args.maker = 'Audi'
    args.genmodel= 'A4'
    args.base_year = 2001
    args.input_pre_extend = 2
    args.simulation_start_year = 2009
    args.simulation_end_year = 2015
    args.ori_facelift_year = 2012
    args.base_aes = 1.394
    args.ori_facelift_aes = 1.4375798
    args.given_aes_rises_l = [1.382-args.base_aes, 1.439-args.base_aes]

    return args


def get_ford_settings():
    args = get_default_para()
    args.maker = 'Ford'
    args.genmodel= 'EcoSport'
    args.base_year = 2001
    args.input_pre_extend = 2
    args.simulation_start_year = 2016
    args.simulation_end_year = 2018
    args.ori_facelift_year = 2017
    args.base_aes = 1.102
    args.ori_facelift_aes = 1.3295
    args.given_aes_rises_l = [1.072-args.base_aes, 1.138-args.base_aes]

    return args


def get_skoda_settings():
    args = get_default_para()
    args.maker = 'SKODA'
    args.genmodel= 'Yeti'
    args.base_year = 2001
    args.input_pre_extend = 1
    args.simulation_start_year = 2012
    args.simulation_end_year = 2017
    args.ori_facelift_year = 2013 #2012
    args.base_aes = 0
    args.ori_facelift_aes = 0
    args.given_aes_rises_l = [0.1]
    return args


def get_jetta_settings():
    args = get_default_para()
    args.maker = 'Volkswagen'
    args.genmodel= 'Jetta'
    args.base_year = 2001
    args.input_pre_extend = 2
    args.simulation_start_year = 2013
    args.simulation_end_year = 2018
    args.ori_facelift_year = 2014
    args.base_aes = 1
    args.ori_facelift_aes = 1.1
    args.given_aes_rises_l = [0.1]

    return args


def get_rav4_settings():
    args = get_default_para()
    args.maker = 'Toyota'
    args.genmodel= 'RAV4'
    args.base_year = 2001
    args.input_pre_extend = 2
    args.simulation_start_year = 2013
    args.simulation_end_year = 2018
    args.ori_facelift_year = 2015
    args.base_aes = 1
    args.ori_facelift_aes = 1.1
    args.given_aes_rises_l = [0.1]

    return args


def get_jazz_settings():
    args = get_default_para()
    args.maker = 'Honda'
    args.genmodel= 'Jazz'
    args.base_year = 2001
    args.input_pre_extend = 1
    args.simulation_start_year = 2014
    args.simulation_end_year = 2018
    args.ori_facelift_year = 2017
    args.base_aes = 1
    args.ori_facelift_aes = 1.1
    args.given_aes_rises_l = [0.1]

    return args
#=========================
def get_i10_settings():
    args = get_default_para()
    args.maker = 'Hyundai'
    args.genmodel= 'i10'
    args.base_year = 2001
    args.input_pre_extend = 1
    args.simulation_start_year = 2011
    args.simulation_end_year = 2016
    args.ori_facelift_year = 2010
    args.base_aes = 1
    args.ori_facelift_aes = 1.1
    args.given_aes_rises_l = [0.1]

    return args

def get_508_settings():
    args = get_default_para()
    args.maker = 'Peugeot'
    args.genmodel = '508'
    args.base_year = 2001
    args.input_pre_extend = 2
    args.simulation_start_year = 2013
    args.simulation_end_year = 2018
    args.ori_facelift_year = 2014
    args.base_aes = 1
    args.ori_facelift_aes = 1.1
    args.given_aes_rises_l = [0.1]

    return args

def get_1Series_settings():
    args = get_default_para()
    args.maker = 'BMW'
    args.genmodel = '1 Series'
    args.base_year = 2001
    args.input_pre_extend = 1
    args.simulation_start_year = 2012
    args.simulation_end_year = 2019
    args.ori_facelift_year = 2015
    args.base_aes = 1
    args.ori_facelift_aes = 1.1
    args.given_aes_rises_l = [0.1]

    return args

def get_2008_settings():
    args = get_default_para()
    args.maker = 'Peugeot'
    args.genmodel = '2008'
    args.base_year = 2001
    args.input_pre_extend = 3
    args.simulation_start_year = 2015
    args.simulation_end_year = 2019
    args.ori_facelift_year = 2017
    args.base_aes = 1
    args.ori_facelift_aes = 1.1
    args.given_aes_rises_l = [0.1]

    return args

def get_phaeton_settings():
    args = get_default_para()
    args.maker = 'Volkswagen'
    args.genmodel = 'Phaeton'
    args.base_year = 2001
    args.input_pre_extend = 1
    args.simulation_start_year = 2005
    args.simulation_end_year = 2015
    args.ori_facelift_year = 2010
    args.base_aes = 1
    args.ori_facelift_aes = 1.1
    args.given_aes_rises_l = [0.1]

    return args
