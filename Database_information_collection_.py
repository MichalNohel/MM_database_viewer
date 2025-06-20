# -*- coding: utf-8 -*-
"""
Created on Wed May 14 12:33:23 2025

@author: nohel
"""
import os
join=os.path.join
from pydicom.fileset import FileSet
import pandas as pd
import datetime
from matplotlib import pyplot as plt
import seaborn as sns
import pydicom
import numpy as np
from statistics import mode

def create_file_set_from_path(dicoms_root_dir):
    fs = FileSet()
    file_list = list(os.walk(dicoms_root_dir))
    for root, _, files in file_list:
        for file in files:
            if 'I10' == file:
                file_path = os.path.join(root, file)
                fs.add(file_path)
            else:
                continue
    return fs

def get_metadata_SBI(dicoms_dir_path):
    fs = create_file_set_from_path(dicoms_dir_path)
    results_table = pd.DataFrame()
    for scan_idx, study in enumerate(fs):
        ds = study.load()        
        if ds.PatientID.startswith('sCT_MyelCon'):
            continue        
        # if hasattr(ds, 'SeriesDescription') and hasattr(ds, 'ConvolutionKernel'):
        #     if hasattr(ds, 'ContrastBolusAgent'):
        #         ContrastBolusAgent = ds.ContrastBolusAgent
        #     else:
        #         ContrastBolusAgent = 'None'
                
        if hasattr(ds, 'SeriesDescription') and hasattr(ds, 'ConvolutionKernel'):
            if hasattr(ds, 'SpacingBetweenSlices'):
                SpacingBetweenSlices = ds.SpacingBetweenSlices
            else:
                SpacingBetweenSlices = 'None'
                
                
            results_table = pd.concat([results_table, pd.DataFrame({'StudyID': 'S' + str(ds.StudyID) + '0',
                                                                    'PatientID': ds.PatientID,
                                                                    'SeriesNumber': 'S' + str(ds.SeriesNumber) + '0',
                                                                    'SeriesDescription': ds.SeriesDescription,
                                                                    'PatientAge': ds.PatientAge[:-1],
                                                                    'PatientSex': ds.PatientSex,
                                                                    'Rows': ds.Rows,
                                                                    'columns': ds.Columns,
                                                                    'PixelSpacing': float(ds.PixelSpacing[0]),
                                                                    'SliceThickness': float(ds.SliceThickness),
                                                                    'SpacingBetweenSlices': SpacingBetweenSlices,
                                                                    #'ContrastBolusAgent': ContrastBolusAgent,
                                                                    'FilterType': ds.FilterType,
                                                                    'ConvolutionKernel': ds.ConvolutionKernel,
                                                                    'DataCollectionDiameter': float(ds.DataCollectionDiameter),
                                                                    'ReconstructionDiameter': float(ds.ReconstructionDiameter),
                                                                    'ScanOptions': ds.ScanOptions,
                                                                    'KVP': int(ds.KVP),
                                                                    'XRayTubeCurrent': int(ds.XRayTubeCurrent),
                                                                    'ExposureTime': ds.ExposureTime,
                                                                    'Exposure': ds.Exposure,
                                                                    'CTDIvol': ds.CTDIvol,
                                                                    'ManufacturerModelName': ds.ManufacturerModelName
                                                                    }, index=[scan_idx])])
        else:
            continue
    return results_table

def get_metadata(dicoms_dir_path,only_conv=True):
    fs = create_file_set_from_path(dicoms_dir_path)
    results_table = pd.DataFrame()
    for scan_idx, study in enumerate(fs):
        ds = study.load()    
        
        if only_conv:            
            if not ds.SeriesDescription.endswith('_konv'):
                continue   

        
        # if hasattr(ds, 'SeriesDescription') and hasattr(ds, 'ConvolutionKernel'):
        #     if hasattr(ds, 'ContrastBolusAgent'):
        #         ContrastBolusAgent = ds.ContrastBolusAgent
        #     else:
        #         ContrastBolusAgent = 'None'
                
        if hasattr(ds, 'SeriesDescription') and hasattr(ds, 'ConvolutionKernel'):
            if hasattr(ds, 'SpacingBetweenSlices'):
                SpacingBetweenSlices = ds.SpacingBetweenSlices
            else:
                SpacingBetweenSlices = 'None'
                
            results_table = pd.concat([results_table, pd.DataFrame({'StudyID': 'S' + str(ds.StudyID) + '0',
                                                                    'PatientID': ds.PatientID,
                                                                    'SeriesNumber': 'S' + str(ds.SeriesNumber) + '0',
                                                                    'SeriesDescription': ds.SeriesDescription,
                                                                    'PatientAge': ds.PatientAge[:-1],
                                                                    'PatientSex': ds.PatientSex,                                                                    
                                                                    'Rows': ds.Rows,
                                                                    'columns': ds.Columns,
                                                                    'PixelSpacing': float(ds.PixelSpacing[0]),
                                                                    #'PixelSpacing2': float(ds.PixelSpacing[1]),
                                                                    'SliceThickness': float(ds.SliceThickness),
                                                                    'SpacingBetweenSlices': SpacingBetweenSlices,
                                                                    #'ContrastBolusAgent': ContrastBolusAgent,
                                                                    'FilterType': ds.FilterType,
                                                                    'ConvolutionKernel': ds.ConvolutionKernel,
                                                                    'DataCollectionDiameter': float(ds.DataCollectionDiameter),
                                                                    'ReconstructionDiameter': float(ds.ReconstructionDiameter),
                                                                    'ScanOptions': ds.ScanOptions,
                                                                    'KVP': int(ds.KVP),
                                                                    'XRayTubeCurrent': int(ds.XRayTubeCurrent),
                                                                    'ExposureTime': ds.ExposureTime,
                                                                    'Exposure': ds.Exposure,
                                                                    #'CTDIvol': ds.CTDIvol,
                                                                    'ManufacturerModelName': ds.ManufacturerModelName
                                                                    #'AcquisitionDate': datetime.datetime.strptime(ds.AcquisitionDate, '%Y%m%d')
                                                                    }, index=[scan_idx])])
        else:
            continue
    return results_table


def get_spacing_between_slices_from_DICOM_data(DICOM_files):
    
    # Load metadata from dicom 
    slices = [pydicom.dcmread(dicom_file) for dicom_file in DICOM_files]
    # Sorting of slices based on ImagePosition
    slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))
    
    spacing_between_slices_all = []
    for idx in range(0,len(slices)-1):
        spacing_between_slices_all.append(round(float(np.abs(slices[idx].SliceLocation - slices[idx+1].SliceLocation)),2))
          
    unique_spacing_between_slices = np.unique(spacing_between_slices_all)
    spacing_between_slices = mode(spacing_between_slices_all)
    
    

    return spacing_between_slices,unique_spacing_between_slices

if __name__ == "__main__":
    
    
    metadata_table = get_metadata('E:/ISP_Myelomy_export', only_conv=False)
    '''
    
    base='E:/ISP_Myelomy_export'
    
    all_ID_patient = [name for name in os.listdir(base) if os.path.isdir(os.path.join(base, name)) and name.startswith('S')]
    
    results_table_SpacingBetweenSlices = pd.DataFrame()
    
    for patient_idx, ID_patient in enumerate(all_ID_patient):
        patient_main_file=join(base,ID_patient)    
        DICOM_folders_all = []
        # Procházení souborů v adresáři
        for filename in os.listdir(patient_main_file):
            if filename.startswith('S20'):
                DICOM_folders_all.append(filename)
        print(DICOM_folders_all)
    
    
        # %%
        for DICOM_folder in DICOM_folders_all:
            DICOM_folder_path=join(patient_main_file,DICOM_folder)
            # Načtení všech DICOM souborů z adresáře, vynechání souboru DIRFILE
            DICOM_files = [os.path.join(DICOM_folder_path, f) for f in os.listdir(DICOM_folder_path) if f != 'DIRFILE']
            series_description = pydicom.dcmread(DICOM_files[0]).get('SeriesDescription')
            #print(series_description)
            
            if series_description.endswith('_konv'):
                SpacingBetweenSlices,unique_spacing_between_slices = get_spacing_between_slices_from_DICOM_data(DICOM_files)
                print (series_description)
                print (SpacingBetweenSlices)
                print (unique_spacing_between_slices)
                
                results_table_SpacingBetweenSlices = pd.concat([results_table_SpacingBetweenSlices, pd.DataFrame({'SeriesDescription': series_description,
                                                                          'SpacingBetweenSlices': SpacingBetweenSlices,
                                                                          'unique_spacing_between_slices_min': unique_spacing_between_slices[0],
                                                                          'unique_spacing_between_slices_max': unique_spacing_between_slices[1]
                                                                        }, index=[patient_idx])])
                break
            else:
                continue
    
    results_table_SpacingBetweenSlices.to_excel('metadata_SpacingBetweenSlices.xlsx')
    
    '''
    
    
    
    
    # %% Get Metadata
    # metadata_table_SBI = get_metadata_SBI('F:/Raw_data_SBI')
    # #metadata_table_SBI.to_excel('metadata_SBI.xlsx')
    # metadata_table_SBI_sorted = metadata_table_SBI.sort_values(by='PatientID')
    # metadata_table_SBI_sorted.to_excel('metadata_SBI_sorted.xlsx')
    
    '''
    metadata_table = get_metadata('E:/ISP_Myelomy_export', only_conv=False)    
    #metadata_table.to_excel('metadata_export_all.xlsx')
    metadata_table_sorted = metadata_table.sort_values(by='PatientID')
    metadata_table_sorted.to_excel('metadata_export_all_sorted.xlsx')
    
    metadata_table = get_metadata('E:/ISP_Myelomy_export', only_conv=True)  
    #metadata_table.to_excel('metadata_export_only_conv.xlsx')
    metadata_table_sorted = metadata_table.sort_values(by='SeriesDescription')
    metadata_table_sorted.to_excel('metadata_export_only_conv_sorted.xlsx')
    
    
    
    # %%
    metadata_table_SBI = pd.read_excel('metadata_SBI_sorted.xlsx')
    plt.figure()
    ax = sns.histplot(metadata_table_SBI[['CTDIvol']])
    ax.set(xlabel='CTDIvol [mGy]',
        ylabel='Count (-)',
        title='CTDIvol')
    plt.show()
    plt.savefig('CTDIvol.png', bbox_inches='tight') 
    
        
    metadata_table = pd.read_excel('metadata_export_only_conv_sorted.xlsx')    
    
    plt.figure()
    ax = sns.histplot(metadata_table[['PatientAge']])
    ax.set(xlabel='Age (years)',
        ylabel='Count (-)',
        title='Age distribution')
    # plt.show()
    plt.savefig('age_distribution.png', bbox_inches='tight')
    
    #plt.figure()
    ax = sns.catplot(data=metadata_table, x='PatientSex', kind='count')
    ax.set(xlabel='Sex (-)',
        ylabel='Count (-)',
        title='Sex distribution')
    # plt.show()
    plt.savefig('sex_distribution.png', bbox_inches='tight')

    plt.figure()
    ax = sns.histplot(metadata_table[['PixelSpacing', 'SliceThickness']])
    ax.set(xlabel='Size (mm)',
        ylabel='Count (-)',
        title='Voxel size distribution')
    # plt.show()
    plt.savefig('voxel_size_distribution.png', bbox_inches='tight')
    
    #plt.figure()
    ax = sns.catplot(data=metadata_table, x='FilterType', kind='count')
    ax.set(xlabel='Filter Type (-)',
        ylabel='Count (-)',
        title='Filter type distribution')
    # plt.show()
    plt.savefig('filter_type_distribution.png', bbox_inches='tight')
    '''
    
    
    
    
    
    
    
    