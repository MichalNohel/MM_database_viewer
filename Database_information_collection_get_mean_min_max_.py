# -*- coding: utf-8 -*-
"""
Created on Tue Jun 17 12:28:18 2025

@author: nohel
"""

import pandas as pd
import numpy as np

if __name__ == "__main__":
    
    metadata_table = pd.read_excel('metadata_export_only_conv_sorted.xlsx')
    
    #%% get Values of Patient Age  
    patient_age = metadata_table['PatientAge'].tolist()
    min_patient_age = int(np.min(patient_age))
    max_patient_age = int(np.max(patient_age))
    mean_patient_age = float(np.mean(patient_age))
    
    
    #%% get Values of Patient Sex 
    patient_sex = metadata_table['PatientSex'].tolist()
    num_of_men = patient_sex.count('M')
    num_of_women = patient_sex.count('F')
    
    #%% get Values of Patient Spacing 
    pixel_spacing = metadata_table['PixelSpacing'].tolist()
    min_pixel_spacing = float(np.min(pixel_spacing))
    max_pixel_spacing = float(np.max(pixel_spacing))
