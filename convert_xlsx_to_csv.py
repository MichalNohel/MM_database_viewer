# -*- coding: utf-8 -*-
"""
Created on Tue Jun 24 10:15:44 2025

@author: nohel
"""

import pandas as pd
import os
join=os.path.join

if __name__ == "__main__":
    
    base = 'F:\Spinal-Multiple-Myeloma-SEG-Example-Data-Nohel\Spreadsheets' 
    
    Table_clinical_data = pd.read_excel(join(base,'Table_clinical_data.xlsx'))
    Table_clinical_data.to_csv(join(base,'Table_clinical_data.csv'), index=False, encoding='cp1250')
    
    Table_of_acquisition_parameters = pd.read_excel(join(base,'Table_of_acquisition_parameters.xlsx'))
    Table_of_acquisition_parameters.to_csv(join(base,'Table_of_acquisition_parameters.csv'), index=False, encoding='cp1250')
    
    
    Table_of_image_data_description = pd.read_excel(join(base,'Table_of_image_data_description.xlsx'))
    Table_of_image_data_description.to_csv(join(base,'Table_of_image_data_description.csv'), index=False, encoding='cp1250')
    
    