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
    
    return join(orientation)

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

def load_DICOM_data(DICOM_files):
    
    # Načtení metadat a pixelových dat
    slices = [pydicom.dcmread(dicom_file) for dicom_file in DICOM_files]
    # Seřazení slices podle SliceLocation nebo InstanceNumber
    slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))
    # Extrahování pixelových dat
    img_data_zxy = np.stack([s.pixel_array for s in slices])
    return img_data_zxy
    
#%%
if __name__ == "__main__":
    
    #%%
    # dicom_file_path = 'F:/MM_Dataset/S80060/S207300/I10'
    # orientation = get_dicom_orientation(dicom_file_path)
    # print(f"DICOM Orientation: {orientation}")
    
    
    #%%
    
    base='../MM_Dataset/'
    ID_patient="S14180"
    print(ID_patient)
    patient_main_file=join(base,ID_patient)
    
    DICOM_folders_all = []

    # Procházení souborů v adresáři
    for filename in os.listdir(patient_main_file):
        if filename.startswith('S20'):
            DICOM_folders_all.append(filename)
    print(DICOM_folders_all)
    
    
    
    
    #%%
    
    # dicom_dir = join(patient_main_file,DICOM_folders_all[0])
    
    # # Načtení všech DICOM souborů z adresáře, vynechání souboru DIRFILE
    # dicom_files = [os.path.join(dicom_dir, f) for f in os.listdir(dicom_dir) if f != 'DIRFILE']
    
    # series_description = pydicom.dcmread(dicom_files[0]).get('SeriesDescription')
    
    # # Načtení metadat a pixelových dat
    # slices = [pydicom.dcmread(dicom_file) for dicom_file in dicom_files]
    
    # # Seřazení slices podle SliceLocation nebo InstanceNumber
    # slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))
    
    # # Extrahování pixelových dat
    # image_data_zxy = np.stack([s.pixel_array for s in slices])
    
    
    
    # # %%
    # #Load Masks
    # path_spine_mask= [os.path.join(patient_main_file, f) for f in os.listdir(patient_main_file) if f.endswith('nnUNet_cor.nii.gz')]
    
    # path_lesion_mask= [os.path.join(patient_main_file, f) for f in os.listdir(patient_main_file) if f.endswith('final.nii.gz')]
        
    
    # # %%
    # reoriented_nifti_img=reorient_nifti_to_ras(path_spine_mask[0])
    # SegmMaskSpine=reoriented_nifti_img.get_fdata()
    # SegmMaskSpine_zxy = SegmMaskSpine.transpose(2,0,1)
    
    # #%%
    # #sitk.ProcessObject_SetGlobalWarningDisplay(False)
    
    # SegmMaskSpine_tzxy = sitk.GetArrayFromImage(sitk.ReadImage(path_spine_mask)).astype(np.float32)
    # SegmMaskSpine_zxy=SegmMaskSpine_tzxy[0,:,:,:]
    
    # SegmMaskLesions_tzxy = sitk.GetArrayFromImage(sitk.ReadImage(path_lesion_mask)).astype(np.float32)
    # SegmMaskLesions_zxy=SegmMaskLesions_tzxy[0,:,:,:]    
    
    #%%
    # SegmMaskSpine = nib.load(path_spine_mask[0]).get_fdata()
    # SegmMaskSpine_zxy = SegmMaskSpine.transpose(2,0,1)
    # #SegmMaskSpine_zxy = np.rot90(SegmMaskSpine_zxy, k=1, axes=(1, 2))
    

    # SegmMaskLesions = nib.load(path_lesion_mask[0]).get_fdata()
    # SegmMaskLesions_zxy = SegmMaskLesions.transpose(2,0,1)
    # #SegmMaskLesions_zxy = np.rot90(SegmMaskLesions_zxy, k=1, axes=(1, 2))
    #%%
    # v = napari.Viewer()
    # #datalayer = v.add_image(image_data_zxy, name='data')    
    # datalayer = v.add_image(image_data_zxy, name='data')   
    
    # datalayer.colormap = 'gray'
    # datalayer.blending = 'additive'
        
    # SpineMaskLayer = v.add_image(SegmMaskSpine_zxy, name='SpineMask')
    # SpineMaskLayer.colormap = 'blue'
    # SpineMaskLayer.blending = 'additive'
    # SpineMaskLayer.opacity = 0.5
    
    # LesionMaskLayer = v.add_image(SegmMaskLesions_zxy, name='LessionMask')
    # LesionMaskLayer.colormap = 'red'
    # LesionMaskLayer.blending = 'additive'
    # LesionMaskLayer.opacity = 1
    # napari.run()
    
    
    
    
    
    
    
    
    
    
    # %%
    for DICOM_folder in DICOM_folders_all:
        DICOM_folder_path=join(patient_main_file,DICOM_folder)
        # Načtení všech DICOM souborů z adresáře, vynechání souboru DIRFILE
        DICOM_files = [os.path.join(DICOM_folder_path, f) for f in os.listdir(DICOM_folder_path) if f != 'DIRFILE']
        series_description = pydicom.dcmread(DICOM_files[0]).get('SeriesDescription')
        print(series_description)
        if series_description=='Calcium Suppression 25 Index[HU*]':
            CaSupp25_zxy=load_DICOM_data(DICOM_files)
        elif series_description=='Calcium Suppression 50 Index[HU*]':
            CaSupp50_zxy=load_DICOM_data(DICOM_files)
        elif series_description=='Calcium Suppression 75 Index[HU*]':
            CaSupp75_zxy=load_DICOM_data(DICOM_files)
        elif series_description=='Calcium Suppression 100 Index[HU*]':
            CaSupp100_zxy=load_DICOM_data(DICOM_files)
        elif series_description=='MonoE 40keV[HU]':
            VMI40_zxy=load_DICOM_data(DICOM_files)
        elif series_description=='MonoE 80keV[HU]':
            VMI80_zxy=load_DICOM_data(DICOM_files)
        elif series_description=='MonoE 120keV[HU]':
            VMI120_zxy=load_DICOM_data(DICOM_files)
        else:
            ConvCT_zxy=load_DICOM_data(DICOM_files)
    # %%
    #Load Masks
    path_spine_mask= [os.path.join(patient_main_file, f) for f in os.listdir(patient_main_file) if f.endswith('nnUNet_cor.nii.gz')]
    
    path_lesion_mask= [os.path.join(patient_main_file, f) for f in os.listdir(patient_main_file) if f.endswith('final.nii.gz')]
    
    #%%
    SegmMaskSpine = nib.load(path_spine_mask[0]).get_fdata()
    SegmMaskSpine_zxy = SegmMaskSpine.transpose(2,0,1)
    SegmMaskSpine_zxy = np.rot90(SegmMaskSpine_zxy, k=1, axes=(1, 2))
    

    SegmMaskLesions = nib.load(path_lesion_mask[0]).get_fdata()
    SegmMaskLesions_zxy = SegmMaskLesions.transpose(2,0,1)
    SegmMaskLesions_zxy = np.rot90(SegmMaskLesions_zxy, k=1, axes=(1, 2))
    #%%
    v = napari.Viewer()  
    ConvCT_layer = v.add_image(ConvCT_zxy, name='ConvCT')       
    ConvCT_layer.colormap = 'gray'
    ConvCT_layer.blending = 'additive'
    
    VMI40_layer = v.add_image(VMI40_zxy, name='VMI40')       
    VMI40_layer.colormap = 'gray'
    VMI40_layer.blending = 'additive'
    VMI40_layer.visible = False
    
    VMI80_layer = v.add_image(VMI80_zxy, name='VMI80')       
    VMI80_layer.colormap = 'gray'
    VMI80_layer.blending = 'additive'
    VMI80_layer.visible = False
    
    VMI120_layer = v.add_image(VMI120_zxy, name='VMI120')       
    VMI120_layer.colormap = 'gray'
    VMI120_layer.blending = 'additive'
    VMI120_layer.visible = False
        
    CaSupp25_layer = v.add_image(CaSupp25_zxy, name='CaSupp25')       
    CaSupp25_layer.colormap = 'gray'
    CaSupp25_layer.blending = 'additive'
    CaSupp25_layer.visible = False
    
    CaSupp50_layer = v.add_image(CaSupp50_zxy, name='CaSupp50')       
    CaSupp50_layer.colormap = 'gray'
    CaSupp50_layer.blending = 'additive'
    CaSupp50_layer.visible = False
    
    CaSupp75_layer = v.add_image(CaSupp75_zxy, name='CaSupp75')       
    CaSupp75_layer.colormap = 'gray'
    CaSupp75_layer.blending = 'additive'
    CaSupp75_layer.visible = False
    
    CaSupp100_layer = v.add_image(CaSupp100_zxy, name='CaSupp100')       
    CaSupp100_layer.colormap = 'gray'
    CaSupp100_layer.blending = 'additive'
    CaSupp100_layer.visible = False
    
    SpineMaskLayer = v.add_image(SegmMaskSpine_zxy, name='SpineMask')
    SpineMaskLayer.colormap = 'blue'
    SpineMaskLayer.blending = 'additive'
    SpineMaskLayer.opacity = 0.5
    
    LesionMaskLayer = v.add_image(SegmMaskLesions_zxy, name='LessionMask')
    LesionMaskLayer.colormap = 'red'
    LesionMaskLayer.blending = 'additive'
    LesionMaskLayer.opacity = 1
    napari.run()
    
