# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 11:01:21 2024

@author: nohel
"""

import napari
import SimpleITK as sitk
import pydicom
import nibabel as nib
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
    image_data_txz = np.stack([s.pixel_array for s in slices])
    
    
    # viewer = napari.view_image(image_data)
    # napari.run()
    
    
    #Load Masks
    path_spine_mask= [os.path.join(patient_main_file, f) for f in os.listdir(patient_main_file) if f.endswith('nnUNet_cor.nii.gz')]
    
    path_lesion_mask= [os.path.join(patient_main_file, f) for f in os.listdir(patient_main_file) if f.endswith('final.nii.gz')]
    
    
    #sitk.ProcessObject_SetGlobalWarningDisplay(False)
    
    #SegmMaskSpine = sitk.GetArrayFromImage(sitk.ReadImage(path_spine_mask)).astype(np.float32)
    SegmMaskSpine = nib.load(path_spine_mask[0]).get_fdata()
    SegmMaskSpine_txy=SegmMaskSpine.transpose(2,0,1)
    SegmMaskSpine_txy = np.rot90(SegmMaskSpine_txy, k=1, axes=(1, 2))

    
    #SegmMaskLesions = sitk.GetArrayFromImage(sitk.ReadImage(path_lesion_mask)).astype(np.float32)
    SegmMaskLesions = nib.load(path_lesion_mask[0]).get_fdata()
    SegmMaskLesions_txy=SegmMaskLesions.transpose(2,0,1)
    SegmMaskLesions_txy = np.rot90(SegmMaskLesions_txy, k=1, axes=(1, 2))
    #%%
    v = napari.Viewer()
    datalayer = v.add_image(image_data_txz, name='data')    
    datalayer.colormap = 'gray'
    datalayer.blending = 'additive'
    
    
    SpineMaskLayer = v.add_image(SegmMaskSpine_txy, name='SpineMask')
    SpineMaskLayer.colormap = 'gray'
    SpineMaskLayer.blending = 'additive'
    
    SpineMaskLayer = v.add_image(SegmMaskLesions_txy, name='LessionMask')
    SpineMaskLayer.colormap = 'red'
    SpineMaskLayer.blending = 'additive'
    
    
    napari.run()
    
    
    
