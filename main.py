import numpy as np
import pandas as pd 
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data

import pre_train_ar
import run_ar

import train_ar

import train_point_forecast_vanilla
import train_point_forecast_reinforced

import train_reinforced
import train_vanilla
import train_reinforced_W
import train_vanilla_W
import run_trained_pf_vanilla
import run_trained_pf_reinforced
import run_trained_reinforced
import intro_plot


# Define folders and fname
# folders end with "/"
# fname not

# mode: choose train GLOW or train AR:
# ----------------------- mode == 1: pre-train AR for point estimation and train GLOW
# ----------------------- mode == 2: train AR
for mode in [11]:
    # Write a loop to change data_folder and data_fname
    #for group_no in range(1, 11, 1):
    for group_no in range(1, 2, 1):
        # pretrain_data_folder +  pretrain_data_fname
        # --------------------- download dataset for pre-training AR
        pretrain_data_folder = 'datasets/Aggregation-Level-10/G{}/'.format(group_no)
        pretrain_data_fname = 'Aggregated-2013-2014.csv'


        # ARModel_save_folder + ARModel_save_fname
        # --------------------- save pre-trained AR model
        ARModel_save_folder = 'datasets/Aggregation-Level-10/G{}/'.format(group_no)
        #ARModel_save_fname = 'best_ARModel.pt'

        # GLOW_data_folder + GLOW_data_fname
        # --------------------- download dataset for training GLOW
        GLOW_data_folder = 'datasets/Aggregation-Level-10/G{}/'.format(group_no) 
        GLOW_data_fname = 'Aggregated-2015-2017.csv'

        # GLOW_data_folder + GLOW_data_fname
        # --------------------- download dataset for training GLOW
        AR_data_folder = 'datasets/Aggregation-Level-10/G{}/'.format(group_no) 
        AR_data_fname = 'Aggregated-2013-2017.csv'

        # GLOW_data_folder + AR_est_folder
        # --------------------- save point estimates from AR
        # --------------------- download point estimates for training GLOW
        # --------------------- point_est_on_train.npy
        # --------------------- point_est_on_valid.npy
        # --------------------- point_est_on_test.npy
        AR_est_folder = GLOW_data_folder

        # GLOW_save_folder
        # --------------------- save results from GLOW
        GLOW_save_folder = "GLOW_results/Aggregation-Level-10/G{}/".format(group_no)

        # Myflows_save_folder
        # --------------------- save results from GLOW
        Myflows_save_folder = "Myflows/Aggregation-Level-10/G{}/".format(group_no)

        # synth_save_folder
        # --------------------- save results from direct conditional generation using Myflows
        direct_synth_save_folder = "Direct_Synth/Aggregation-Level-10/G{}/".format(group_no)

        # synth_save_folder
        # --------------------- save results from direct conditional generation using Myflows
        direct_synth_save_folder2 = "Direct_Synth_flows/Aggregation-Level-10/G{}/".format(group_no)

        # synth_save_folder
        # --------------------- save results from direct conditional generation using Myflows
        regularized_direct_synth_save_folder = "Regularized_Direct_Synth/Aggregation-Level-10/G{}/".format(group_no)
        regularized_direct_synth_save_folder2 = "Regularized_Direct_Synth_flows/Aggregation-Level-10/G{}/".format(group_no)



        # AR_save_folder
        # --------------------- save results from GLOW
        AR_save_folder = "AR_results/Aggregation-Level-10/G{}/".format(group_no)


        if mode == 1:
            # Train AR on G{}
            # --------------- pre-train AR 
            print('*****************')
            print('Processing G{}...'.format(group_no))
            print('*****************')
            pre_train_ar.train_AR(pretrain_data_folder, pretrain_data_fname, 
                                    GLOW_data_folder, GLOW_data_fname, ARModel_save_folder)

            # Train GLOW using generated point estimates
            train_pf_vanilla.train_flow(GLOW_save_folder, GLOW_data_folder, GLOW_data_fname)

        elif mode == 2:
            # Train AR
            print('*****************')
            print('Processing G{}...'.format(group_no))
            print('*****************')
            train_ar.train_AR(AR_data_folder, AR_data_fname, AR_save_folder)

        elif mode == 3:
            # Train GLOW using Myflows
            train_pf_reinforced.train_flow(Myflows_save_folder, GLOW_data_folder, GLOW_data_fname)

        elif mode == 4:
            # Direct synthesize using Myflows
            GLOW_data_fname = 'Aggregated-2013-2017.csv'
            train_reinforced.train_flow(direct_synth_save_folder, GLOW_data_folder, GLOW_data_fname)

        elif mode == 5:
            # Direct synthesize using flows
            # revise Myflows to flows
            GLOW_data_fname = 'Aggregated-2013-2017.csv'
            train_vanilla.train_flow(direct_synth_save_folder2, GLOW_data_folder, GLOW_data_fname)

        elif mode == 6:
            run_trained_pf_vanilla.run_trained_flow(GLOW_save_folder, GLOW_data_folder, GLOW_data_fname)
        
        elif mode == 7:
            run_trained_pf_reinforced.run_trained_flow(Myflows_save_folder, GLOW_data_folder, GLOW_data_fname)
        
        elif mode == 8:
            run_reinforced.run_trained_flow(direct_synth_save_folder, GLOW_data_folder, GLOW_data_fname)

        elif mode == 9:
            train_reinforced_W.train_W_flow(regularized_direct_synth_save_folder, GLOW_data_folder, GLOW_data_fname)

        elif mode == 10:
            train_vanilla_W.train_W_flow(regularized_direct_synth_save_folder2, GLOW_data_folder, GLOW_data_fname)

        elif mode == 11:
            intro_plot.run_trained_flow(direct_synth_save_folder2, GLOW_data_folder, GLOW_data_fname)

        else:
            print('Please choose the correct mode.')
