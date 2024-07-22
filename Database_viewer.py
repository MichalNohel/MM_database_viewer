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
from nibabel.orientations import axcodes2ornt, ornt_transform, apply_orientation

join=os.path.join

def get_dicom_orientation(dicom_file_path):
    # Načtení DICOM souboru
    dicom_data = pydicom.dcmread(dicom_file_path)
    
    # Získání ImageOrientationPatient a ImagePositionPatient
    image_orientation = dicom_data.ImageOrientationPatient
    
    # Výpis hodnot
    print(f"ImageOrientationPatient: {image_orientation}")
    
    # Převod na numpy pole pro snadnější manipulaci
    image_orientation = np.array(image_orientation).reshape(2, 3)
    
    # Definice os
    row_cosines = image_orientation[0, :]
    col_cosines = image_orientation[1, :]
    
    # Výpočet cross product pro získání normálového vektoru k obrazové rovině
    normal_cosines = np.cross(row_cosines, col_cosines)
    
    # Sloučení do jedné matice (řádek, sloupec, normála)
    orientation_matrix = np.vstack((row_cosines, col_cosines, normal_cosines))
    
    # Definice os podle orientace
    axis_labels = ['R', 'A', 'S']
    
    orientation = []
    for axis in orientation_matrix:
        max_index = np.argmax(np.abs(axis))
        sign = np.sign(axis[max_index])
        orientation.append(axis_labels[max_index] if sign > 0 else axis_labels[max_index].swapcase())
    
    return ''.join(orientation)

def reorient_nifti_to_ras(nifti_file_path):
    # Načtení NIFTI souboru
    nifti_img = nib.load(nifti_file_path)
    nifti_data = nifti_img.get_fdata()
    nifti_affine = nifti_img.affine

    # Zjištění aktuální orientace
    current_orientation = nib.aff2axcodes(nifti_affine)
    print(f"Current orientation: {current_orientation}")

    # Cílová orientace (RAS)
    target_orientation = ('R', 'A', 'S')
    
    # Transformace orientace
    ornt_trans = ornt_transform(axcodes2ornt(current_orientation), axcodes2ornt(target_orientation))
    reoriented_data = apply_orientation(nifti_data, ornt_trans)

    # Vytvoření nové afinní matice po reorientaci
    reoriented_affine = nifti_affine @ nib.orientations.inv_ornt_aff(ornt_trans, nifti_data.shape)
    
    # Vytvoření nového NIFTI obrazu s reorientovanými daty a aktualizovanou afinní maticí
    reoriented_nifti_img = nib.Nifti1Image(reoriented_data, reoriented_affine)

    return reoriented_nifti_img

#%%
if __name__ == "__main__":
    
    #%%
    dicom_file_path = 'F:/MM_Dataset/S80060/S207300/I10'
    orientation = get_dicom_orientation(dicom_file_path)
    print(f"DICOM Orientation: {orientation}")
    
    
    #%%
    
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
    
    series_description = pydicom.dcmread(dicom_files[0]).get('SeriesDescription')
    
    # Načtení metadat a pixelových dat
    slices = [pydicom.dcmread(dicom_file) for dicom_file in dicom_files]
    
    # Seřazení slices podle SliceLocation nebo InstanceNumber
    slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))
    
    
    
    #%% 
    # Extrahování pixelových dat
    image_data_zxy = np.stack([s.pixel_array for s in slices])
    
    image_data_tzxy=np.zeros([2,1095,512,512])
    image_data_tzxy[0,:,:,:]=image_data_zxy
    image_data_tzxy[1,:,:,:]=image_data_zxy
    # viewer = napari.view_image(image_data)
    # napari.run()
    
    # %%
    #Load Masks
    path_spine_mask= [os.path.join(patient_main_file, f) for f in os.listdir(patient_main_file) if f.endswith('nnUNet_cor.nii.gz')]
    
    path_lesion_mask= [os.path.join(patient_main_file, f) for f in os.listdir(patient_main_file) if f.endswith('final.nii.gz')]
    
    #%%
    #sitk.ProcessObject_SetGlobalWarningDisplay(False)
    
    # SegmMaskSpine_tzxy = sitk.GetArrayFromImage(sitk.ReadImage(path_spine_mask)).astype(np.float32)
    # SegmMaskSpine_zxy=SegmMaskSpine_tzxy[0,:,:,:]
    
    # SegmMaskLesions_tzxy = sitk.GetArrayFromImage(sitk.ReadImage(path_lesion_mask)).astype(np.float32)
    # SegmMaskLesions_zxy=SegmMaskLesions_tzxy[0,:,:,:]
    
    
    #%%
    SegmMaskSpine = nib.load(path_spine_mask[0]).get_fdata()
    SegmMaskSpine_zxy = SegmMaskSpine.transpose(2,0,1)
    SegmMaskSpine_zxy = np.rot90(SegmMaskSpine_zxy, k=1, axes=(1, 2))
    

    SegmMaskLesions = nib.load(path_lesion_mask[0]).get_fdata()
    SegmMaskLesions_zxy = SegmMaskLesions.transpose(2,0,1)
    SegmMaskLesions_zxy = np.rot90(SegmMaskLesions_zxy, k=1, axes=(1, 2))
    #%%
    v = napari.Viewer()
    #datalayer = v.add_image(image_data_zxy, name='data')    
    datalayer = v.add_image(image_data_tzxy, name='data')   
    
    datalayer.colormap = 'gray'
    datalayer.blending = 'additive'
        
    SpineMaskLayer = v.add_image(SegmMaskSpine_zxy, name='SpineMask')
    SpineMaskLayer.colormap = 'blue'
    SpineMaskLayer.blending = 'additive'
    SpineMaskLayer.opacity = 0.5
    
    LesionMaskLayer = v.add_image(SegmMaskLesions_zxy, name='LessionMask')
    LesionMaskLayer.colormap = 'red'
    LesionMaskLayer.blending = 'additive'
    LesionMaskLayer.opacity = 1
    
    
    napari.run()
    
    
    
