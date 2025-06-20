# -*- coding: utf-8 -*-
"""
Created on Fri Mar 28 10:56:33 2025

@author: nohel
"""

import SimpleITK as sitk
import numpy as np
import os
join=os.path.join
import nibabel as nib
from batchgenerators.utilities.file_and_folder_operations import *

def load_dicom_series(dicom_folder):
    """Načte celou DICOM sérii jako 3D objem a vrátí afinní transformační matici."""
    
    # Inicializace čtečky DICOM souborů
    reader = sitk.ImageSeriesReader()
    
    # Získání seznamu souborů v DICOM sérii
    dicom_series = reader.GetGDCMSeriesFileNames(dicom_folder)
    reader.SetFileNames(dicom_series)
    
    # Načtení objemu jako SimpleITK image
    image = reader.Execute()
    
    # Získání afinní transformační matice
    direction = np.array(image.GetDirection()).reshape(3,3)  # 3x3 rotační matice
    spacing = np.array(image.GetSpacing())  # Velikost voxelů
    origin = np.array(image.GetOrigin())  # Počátek v DICOM světě
    
    # Sestavení 4x4 afinní matice
    affine_matrix = np.eye(4)
    affine_matrix[:3, :3] = direction * spacing  # Maticová násobení -> správné měřítko
    affine_matrix[:3, 3] = origin  # Posun
    
    return image, affine_matrix

def reorient_to_DICOM_orientation(dicom_folder,nifti_path,output_path):
    # Načtení objemu a afinní matice
    image, affine_matrix = load_dicom_series(dicom_folder)

    # Načtení segmentační masky jako NIfTI souboru
    mask_img = nib.load(nifti_path)
    
    # Získání dat segmentační masky a její původní afinní matice
    mask_data = mask_img.get_fdata()
    mask_data = mask_data.transpose(1,0,2)
    mask_data = np.rot90(mask_data, k=-1, axes=(0, 1))
    
    
    original_affine = mask_img.affine    
    # Změna orientace: Vynásobíme první a druhý řádek afinní matice -1
    affine_matrix[0, :] *= -1  # Vynásobení celého 1. řádku -1
    affine_matrix[1, :] *= -1  # Vynásobení celého 2. řádku -1
    
    # Uložení segmentační masky s novou afinní maticí
    new_mask_img = nib.Nifti1Image(mask_data, affine_matrix)
    
    # Uložení do nového souboru
    nib.save(new_mask_img, output_path)
    
#%%
if __name__ == "__main__":
    base = 'E:/Znaceni_dat/Data/'
    
    for t in subdirs(base, join=False, prefix="Myel_"): 
        
        dicom_folder = join(base, t, 'ConvCT_data_dicom')
           
        path_spine_mask = subfiles(join(base, t, 'Spine_labels/NN_Unet'), join=True, suffix="spine_seg_nnUNet_cor.nii.gz")[0]   
        nifti_path = path_spine_mask
        # output_path = path_lesion_mask[:-7] + '_DICOM_Orientation_final_3.nii.gz'
        output_path = join(base, t, 'Spine_labels/NN_Unet', t + '_spine_segmentation.nii.gz')
        reorient_to_DICOM_orientation(dicom_folder,nifti_path,output_path)
        
        

        
        # #Load Masks
        path_lesion_mask = subfiles(join(base, t, 'Lesion_labels'), join=True, suffix="validation_VV_final_semantic.nii.gz")[0]  
        nifti_path = path_lesion_mask
        # output_path = path_lesion_mask[:-7] + '_DICOM_Orientation_final_3.nii.gz'
        output_path = join(base, t, 'Lesion_labels', t + '_lesions_segmentation.nii.gz')
        reorient_to_DICOM_orientation(dicom_folder,nifti_path,output_path)
        
        
        
        
        
        
        
        
        
        
        
        
        
    
    
    
    
    
    