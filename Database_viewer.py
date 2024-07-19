# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 11:01:21 2024

@author: nohel
"""

import napari
import SimpleITK as sitk
import pydicom
import numpy as np
import os
join=os.path.join



if __name__ == "__main__":
    
    base='../MM_Dataset/'
    ID_patient="S80060"
    patient_main_file=join(base,ID_patient)
    
    DICOM_files_all = []

    # Procházení souborů v adresáři
    for filename in os.listdir(patient_main_file):
        if filename.startswith('S20'):
            DICOM_files_all.append(filename)

    print(DICOM_files_all)
    
    
    dicom_dir = join(patient_main_file,DICOM_files_all[0])
    
    # Načtení všech DICOM souborů z adresáře, vynechání souboru DIRFILE
    dicom_files = [os.path.join(dicom_dir, f) for f in os.listdir(dicom_dir) if f != 'DIRFILE']
    
    # Načtení metadat a pixelových dat
    slices = [pydicom.dcmread(dicom_file) for dicom_file in dicom_files]
    
    # Seřazení slices podle SliceLocation nebo InstanceNumber
    slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))
    
    # Extrahování pixelových dat
    image_data = np.stack([s.pixel_array for s in slices])
    
    viewer = napari.view_image(image_data)
    napari.run()
    
    
    
    
    
    
    
