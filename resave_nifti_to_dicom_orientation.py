# -*- coding: utf-8 -*-
"""
Created on Fri Mar 28 10:56:33 2025

@author: nohel
"""

import pydicom
import nibabel as nib
from nibabel.orientations import axcodes2ornt, ornt_transform, apply_orientation
import numpy as np
import os
join=os.path.join
import re

def get_valid_dicom_files(dicom_folder):
    """Najde všechny validní DICOM soubory ve složce a vrátí jejich seznam."""
    dicom_dir = dicom_folder
    dicom_files = [os.path.join(dicom_dir, f) for f in os.listdir(dicom_dir) if f != 'DIRFILE']
    dicom_files.sort(key=lambda x: int(re.search(r'\d+', os.path.basename(x)).group()))
    return dicom_files

def get_dicom_geometry(dicom_folder):
    """Načte spacing a origin z DICOM souborů."""
    dicom_files = get_valid_dicom_files(dicom_folder)
    if len(dicom_files) < 2:
        raise ValueError("Není dostatek validních DICOM souborů pro výpočet geometrie.")

    dcm1 = pydicom.dcmread(dicom_files[0])
    dcm2 = pydicom.dcmread(dicom_files[1])

    # Spacing (velikost voxelů)
    pixel_spacing = np.array(dcm1.PixelSpacing, dtype=np.float32)  # (X, Y)
    slice_thickness = abs(float(dcm2.ImagePositionPatient[2]) - float(dcm1.ImagePositionPatient[2]))  # Z-spacing
    spacing = np.array([pixel_spacing[0], pixel_spacing[1], slice_thickness])

    # Origin (počátek souřadnic)
    origin = np.array(dcm1.ImagePositionPatient, dtype=np.float32)  # (X, Y, Z)

    return spacing, origin

def adjust_nifti_geometry(nifti_path, spacing, origin, output_path):
    """Upraví NIfTI tak, aby měl stejné spacing a origin jako DICOM."""
    nifti_img = nib.load(nifti_path)
    nifti_data = nifti_img.get_fdata()
    affine = nifti_img.affine.copy()

    # Změna škálování v matici (Spacing)
    affine[:3, :3] *= np.diag(spacing)

    # Změna počátku souřadnic (Origin)
    affine[:3, 3] = origin

    # Vytvoření a uložení nového NIfTI
    new_nifti = nib.Nifti1Image(nifti_data, affine, header=nifti_img.header)
    nib.save(new_nifti, output_path)
    print(f"Uložen nový NIfTI s DICOM geometrií: {output_path}")
    
#%%
if __name__ == "__main__":
    base='../TEST/'
    ID_patient="S80060"
    print(ID_patient)
    patient_main_file=join(base,ID_patient)
    # #Load Masks
    path_spine_mask= [os.path.join(patient_main_file, f) for f in os.listdir(patient_main_file) if f.endswith('nnUNet_cor.nii.gz')]
    
    
    # nib.save(reoriented_nifti_img, output_name) 
    # SegmMaskSpine=reoriented_nifti_img.get_fdata()
    # SegmMaskSpine_zxy = SegmMaskSpine.transpose(2,0,1)
    
    dicom_folder = join(patient_main_file,'S207300')
    nifti_path = path_spine_mask[0]
    output_path = path_spine_mask[0][:-7] + '_DICOM_Orientation.nii.gz'
    
    # Spuštění procesu
    spacing, origin = get_dicom_geometry(dicom_folder)
    adjust_nifti_geometry(nifti_path, spacing, origin, output_path)
    
    
    