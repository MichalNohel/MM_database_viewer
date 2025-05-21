import os
import datetime

import numpy as np
import seaborn as sns
import pandas as pd
import pingouin as pg

from pydicom.fileset import FileSet
from matplotlib import pyplot as plt
from scipy.stats import norm

from sklearn.linear_model import RANSACRegressor, LinearRegression


def get_image_rotation_matrix(ds):
    # # Load DICOM file
    # ds = pydicom.dcmread(dicom_file)

    # Extract direction cosines
    direction_cosines = ds.ImageOrientationPatient

    # Parse direction cosines
    row_cosines = np.array(direction_cosines[:3])
    column_cosines = np.array(direction_cosines[3:])

    # Calculate normal vector as cross product of row and column cosines
    normal_vector = np.cross(row_cosines, column_cosines)

    # Create rotation matrix
    rotation_matrix = np.column_stack((row_cosines, column_cosines, normal_vector))

    return rotation_matrix


def rotation_matrix_to_euler_angles_3d(rotation_matrix):
    '''
    Converts a 3D rotation matrix to Euler angles (roll, pitch, yaw) in radians.
    '''
    # Extract rotation matrix elements for convenience
    r11, r12, r13 = rotation_matrix[0, :]
    r21, r22, r23 = rotation_matrix[1, :]
    r31, r32, r33 = rotation_matrix[2, :]

    # Calculate yaw (rotation about z-axis)
    yaw = np.arctan2(r21, r11)

    # Calculate roll (rotation about x-axis)
    roll = np.arctan2(-r31, np.sqrt(r32**2 + r33**2))

    # Calculate pitch (rotation about y-axis)
    pitch = np.arctan2(r32, r33)

    return tuple(np.rad2deg((roll, pitch, yaw)))


def create_file_set_from_path(dicoms_root_dir):
    fs = FileSet()
    file_list = list(os.walk(dicoms_root_dir))
    for root, _, files in file_list:
        for file in files:
            if 'I10' == file:
                file_path = os.path.join(root, file)
                fs.add(file_path)
            elif 'CT000000.dcm' == file:
                file_path = os.path.join(root, file)
                fs.add(file_path)
            else:
                continue
    return fs


def compute_angle_errors_sb_expert(path, raters):
    for rater_ID, rater in enumerate(raters):
        dicoms_dir_path = path + rater
        fs = create_file_set_from_path(dicoms_dir_path)
        patient_angle_errors = {}
        angle_errors = []
        patient_series_list = []
        error_patient_series_list = []
        for study in fs:
            ds = study.load()
            if hasattr(ds, 'SeriesDescription'):
                if not ('SB' in ds.SeriesDescription):
                    study_series_id = 'S' + str(ds.StudyID) + '0/' + str(ds.SeriesNumber) + '0'
                    if study_series_id in patient_angle_errors.keys():
                        continue
                    else:
                        if rater_ID > 0:
                            if not study_series_id in results_table['Study/Series'].values:
                                continue
                            else:
                                idx_patient = np.where(results_table['Study/Series'].values == study_series_id)[0][0]
                        try:
                            rotation_matrix_corr = get_image_rotation_matrix(ds)
                            angle_errors.append(rotation_matrix_to_euler_angles_3d(rotation_matrix_corr))
                            patient_angle_errors[study_series_id] = angle_errors[-1]
                            patient_series_list.append(study_series_id)
                        except:
                            error_patient_series_list.append(study_series_id)
                            patient_series_list.append(study_series_id)
                            angle_errors.append((float('NaN'), float('NaN'), float('NaN')))
                        if rater_ID > 0:
                            results_table.at[idx_patient, 'Coronal2'] = angle_errors[-1][0]
                            results_table.at[idx_patient, 'Sagittal2'] = angle_errors[-1][1]
                            results_table.at[idx_patient, 'Axial2'] = angle_errors[-1][2]
                else:
                    continue
            else:
                print('Skipping: S' + str(ds.StudyID) + '0/' + str(ds.SeriesNumber) + '0')
                continue
        angle_errors = np.array(angle_errors)
        if rater_ID == 0:
            results_table = pd.DataFrame(data=angle_errors, columns=['Coronal1', 'Sagittal1', 'Axial1'])
            results_table.insert(0, 'Study/Series', patient_series_list, True)
            results_table['Coronal2'] = np.nan
            results_table['Sagittal2'] = np.nan
            results_table['Axial2'] = np.nan
    results_table = results_table.dropna()
    results_table.loc[results_table['Sagittal2'] >= 90, 'Sagittal2'] = np.abs(results_table.loc[results_table['Sagittal2'] >= 90, 'Sagittal2'] - 90)
    results_table.loc[results_table['Sagittal2'] <= -90, 'Sagittal2'] = np.abs(results_table.loc[results_table['Sagittal2'] <= -90, 'Sagittal2'] + 90)
    return results_table


def compute_angle_errors_sb_radiologist(dicoms_dir_path_sb, dicoms_dir_path_rad):
    fs_sb = create_file_set_from_path(dicoms_dir_path_sb)
    fs_rad = create_file_set_from_path(dicoms_dir_path_rad)
    patient_angle_errors = {}
    angle_sb = []
    angle_rad = []
    patient_series_list = []
    error_patient_series_list = []
    for study_sb in fs_sb:
        ds_sb = study_sb.load()
        if hasattr(ds_sb, 'SeriesDescription'):
            if 'SB' in ds_sb.SeriesDescription:
                study_series_id = 'S' + str(ds_sb.StudyID) + '0/' + str(int(ds_sb.SeriesNumber) + 2) + '0'
                if study_series_id in patient_angle_errors.keys():
                    continue
                else:
                    try:
                        rotation_matrix_sb = get_image_rotation_matrix(ds_sb)
                        studies_rad = fs_rad.find(StudyID=ds_sb.StudyID)
                        if len(studies_rad) == 0:
                            error_patient_series_list.append(study_series_id)
                            angle_rad.append((float('NaN'), float('NaN'), float('NaN')))
                            angle_sb.append((float('NaN'), float('NaN'), float('NaN')))
                        else:
                            ds_rad = ''
                            if hasattr(ds_sb, 'ContrastBolusAgent'):
                                for study_rad in studies_rad:
                                    ds_rad = study_rad.load()
                                    if 'KL' in ds_rad.ImageComments and ds_sb.FilterType in ds_rad.FilterType:
                                        break
                                    else:
                                        ds_rad = ''
                            else:
                                for study_rad in studies_rad:
                                    ds_rad = study_rad.load()
                                    if not ('KL' in ds_rad.ImageComments) and ds_sb.FilterType in ds_rad.FilterType:
                                        break
                                    else:
                                        ds_rad = ''
                            if ds_rad == '':
                                error_patient_series_list.append(study_series_id)
                                angle_rad.append((float('NaN'), float('NaN'), float('NaN')))
                                angle_sb.append((float('NaN'), float('NaN'), float('NaN')))
                            else:
                                rotation_matrix_rad = get_image_rotation_matrix(ds_rad)
                                angle_rad.append(rotation_matrix_to_euler_angles_3d(rotation_matrix_rad))
                                sb_angles = rotation_matrix_to_euler_angles_3d(rotation_matrix_sb)
                                angle_sb.append([-sb_angles[0], -sb_angles[1], -sb_angles[2]])
                                patient_angle_errors[study_series_id] = angle_rad[-1]
                        patient_series_list.append(study_series_id)
                    except:
                        error_patient_series_list.append(study_series_id)
                        patient_series_list.append(study_series_id)
                        angle_rad.append((float('NaN'), float('NaN'), float('NaN')))
                        angle_sb.append((float('NaN'), float('NaN'), float('NaN')))
            else:
                continue
        else:
            print('Skipping: S' + str(ds_sb.StudyID) + '0/' + str(ds_sb.SeriesNumber) + '0')
            continue
    angle_rad = np.array(angle_rad)
    angle_sb = np.array(angle_sb)
    results_table = pd.DataFrame(data=np.concatenate((angle_sb, angle_rad), axis=1), columns=['CoronalSB', 'SagittalSB', 'AxialSB', 'CoronalR', 'SagittalR', 'AxialR'])
    results_table.insert(0, 'Study/Series', patient_series_list, True)
    results_table = results_table.dropna()
    return results_table


def compute_angle_errors_intrarater(dicoms_dir_path_rad1, dicoms_dir_path_rad2, dicoms_dir_path_sb, result_table):
    fs1 = create_file_set_from_path(dicoms_dir_path_rad1)
    fs2 = create_file_set_from_path(dicoms_dir_path_rad2)
    fs_sb = create_file_set_from_path(dicoms_dir_path_sb)
    patient_angle_errors = {}
    angle_1 = []
    angle_2 = []
    angle_sb = []
    patient_series_list = []
    error_patient_series_list = []
    for study in result_table['Study/Series'].values:
        ds_sb_list = fs_sb.find(StudyID=study.split('/')[0][1:-1])
        for ds_sb in ds_sb_list:
            ds_sb_loaded = ds_sb.load()
            if ds_sb_loaded.SeriesNumber==str(int(study.split('/')[1][0:-1]) - 2):
                break
        rotation_matrix_sb = get_image_rotation_matrix(ds_sb_loaded)
        sb_angles = rotation_matrix_to_euler_angles_3d(rotation_matrix_sb)
        angle_sb.append([-sb_angles[0], -sb_angles[1], -sb_angles[2]])
        
        ds1_list = fs1.find(StudyID=study.split('/')[0][1:-1])
        for ds1 in ds1_list:
            ds1_loaded = ds1.load()
            if ds1_loaded.SeriesNumber==study.split('/')[1][0:-1]:
                break
        rotation_matrix_1 = get_image_rotation_matrix(ds1_loaded)
        
        ds2_list = fs2.find(StudyID=study.split('/')[0][1:-1])
        if len(ds2_list) == 0:
            error_patient_series_list.append(study)
            angle_1.append((float('NaN'), float('NaN'), float('NaN')))
            angle_2.append((float('NaN'), float('NaN'), float('NaN')))
            print('Skipping: S' + str(ds1_loaded.StudyID) + '0/' + str(ds1_loaded.SeriesNumber) + '0')
            continue
        ds2_matched = ''
        for ds2 in ds2_list:
            ds2_loaded = ds2.load()
            if ('SB' in ds2_loaded.SeriesDescription) or ('MPR' in ds2_loaded.SeriesDescription) or (str(ds2_loaded.SeriesNumber)[0]=='8'):
                print('Skipping2: S' + str(ds2_loaded.StudyID) + '0/' + str(ds2_loaded.SeriesNumber) + '0')
                ds2_matched = ''
                continue
            else:
                if hasattr(ds1_loaded, 'ContrastBolusAgent'):
                    if 'KL' in ds2_loaded.ImageComments and ds1_loaded.FilterType in ds2_loaded.FilterType:
                        ds2_matched = ds2_loaded
                        break
                    else:
                        ds2_matched = ''
                else:
                    if not ('KL' in ds2_loaded.ImageComments) and ds1_loaded.FilterType in ds2_loaded.FilterType:
                        ds2_matched = ds2_loaded
                        break
                    else:
                        ds2_matched = ''
                        
        if ds2_matched == '':
            error_patient_series_list.append(study)
            angle_1.append((float('NaN'), float('NaN'), float('NaN')))
            angle_2.append((float('NaN'), float('NaN'), float('NaN')))
        else:
            rotation_matrix_2 = get_image_rotation_matrix(ds2_loaded)
            angle_1.append(rotation_matrix_to_euler_angles_3d(rotation_matrix_1))
            angle_2.append(rotation_matrix_to_euler_angles_3d(rotation_matrix_2))
            patient_angle_errors[study] = angle_2[-1]
            
        patient_series_list.append(study)
    angle_2 = pd.DataFrame(data=np.asarray(angle_2), columns=['Coronal1B', 'Sagittal1B', 'Axial1B'])
    angle_sb = pd.DataFrame(data=np.asarray(angle_sb), columns=['CoronalSB', 'SagittalSB', 'AxialSB'])
    result_table = pd.concat([result_table, angle_2], axis=1)
    result_table = pd.concat([result_table, angle_sb], axis=1)
    return(result_table)


def compute_angle_errors_cq_dataset(path_sb, path_corr):
    dicoms_dir_path_sb = path_sb
    dicoms_dir_path_corr = path_corr
    fs_sb = create_file_set_from_path(dicoms_dir_path_sb)
    fs_corr = create_file_set_from_path(dicoms_dir_path_corr)
    patient_angle_errors = {}
    angles_sb = []
    angles_corr = []
    patient_series_list = []
    error_patient_series_list = []
    for study in fs_corr:
        ds_corr = study.load()
        if hasattr(ds_corr, 'SeriesDescription'):
            if not ('SB' in ds_corr.SeriesDescription):
                study_series_corr_id = ds_corr.PatientID + '/S' + str(ds_corr.StudyID) + '0/' + str(ds_corr.SeriesNumber) + '0'
                if study_series_corr_id in patient_angle_errors.keys():
                    continue
                else:
                    studies_sb = fs_sb.find(PatientID=ds_corr.PatientID)
                    study_found = False
                    for study_sb in studies_sb:
                        ds_sb = study_sb.load()
                        if ds_sb.SeriesNumber == ds_corr.SeriesDescription.split(' ')[1]:
                            study_found = True
                            break
                    if not study_found:
                        error_patient_series_list.append(study_series_corr_id)
                        angles_corr.append((float('NaN'), float('NaN'), float('NaN')))
                        angles_sb.append((float('NaN'), float('NaN'), float('NaN')))
                        patient_series_list.append(study_series_corr_id)
                        print('Not paired: S' + str(ds_corr.StudyID) + '0/' + str(ds_corr.SeriesNumber) + '0')
                        continue
                    try:
                        rotation_matrix_corr = get_image_rotation_matrix(ds_corr)
                        angles_corr.append(rotation_matrix_to_euler_angles_3d(rotation_matrix_corr))
                        rotation_matrix_sb = get_image_rotation_matrix(ds_sb)
                        angles_sb.append(rotation_matrix_to_euler_angles_3d(rotation_matrix_sb))
                        patient_angle_errors[study_series_corr_id] = angles_corr[-1]
                        patient_series_list.append(study_series_corr_id)
                    except:
                        error_patient_series_list.append(study_series_corr_id)
                        patient_series_list.append(study_series_corr_id)
                        angles_corr.append((float('NaN'), float('NaN'), float('NaN')))
                        angles_sb.append((float('NaN'), float('NaN'), float('NaN')))
            else:
                continue
        else:
            print('Skipping: S' + str(ds_corr.StudyID) + '0/' + str(ds_corr.SeriesNumber) + '0')
            continue
    angles_corr = np.array(angles_corr)
    results_table = pd.DataFrame(data=angles_corr, columns=['Coronal', 'Sagittal', 'Axial'])
    angles_sb = pd.DataFrame(data=np.array(angles_sb), columns=['CoronalSB', 'SagittalSB', 'AxialSB'])
    results_table = pd.concat([results_table, angles_sb], axis=1)
    results_table.insert(0, 'Study/Series', patient_series_list, True)
    results_table = results_table.dropna()
    return results_table


def get_metadata(dicoms_dir_path):
    fs = create_file_set_from_path(dicoms_dir_path)
    results_table = pd.DataFrame()
    for scan_idx, study in enumerate(fs):
        ds = study.load()
        if hasattr(ds, 'SeriesDescription') and hasattr(ds, 'ConvolutionKernel'):
            if hasattr(ds, 'ContrastBolusAgent'):
                ContrastBolusAgent = ds.ContrastBolusAgent
            else:
                ContrastBolusAgent = 'None'
            results_table = pd.concat([results_table, pd.DataFrame({'StudyID': 'S' + str(ds.StudyID) + '0',
                                                                    'SeriesNumber': 'S' + str(ds.SeriesNumber) + '0',
                                                                    'SeriesDescription': ds.SeriesDescription,
                                                                    'PatientAge': ds.PatientAge[:-1],
                                                                    'PatientSex': ds.PatientSex,
                                                                    'PixelSpacing': float(ds.PixelSpacing[0]),
                                                                    'SliceThickness': float(ds.SliceThickness),
                                                                    'ContrastBolusAgent': ContrastBolusAgent,
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
                                                                    'ManufacturerModelName': ds.ManufacturerModelName,
                                                                    'AcquisitionDate': datetime.datetime.strptime(ds.AcquisitionDate, '%Y%m%d')}, index=[scan_idx])])
        else:
            continue
    return results_table


def bland_altman_plot(data: pd.DataFrame, col1: str, col2: str, *,
                      title: str = 'Bland-Altman Plot', ci: float = 0.95, sd_limit: float = 0.95,
                      annotate: bool = True, units: str = 'deg', gt_data: bool = False,
                      plot_list: list = ['regression', 'BA', 'BA_percent'],
                      outlier_method: str = 'none', outlier_threshold: float = 1.5,  # Used for SD/IQR only, 'none', 'sd', 'iqr', or 'ransac'
                      save_path: str = None):
    '''
    Generate a Bland-Altman plot with 95% confidence intervals for the mean difference and LOA.

    Parameters:
    - data: pd.DataFrame containing the two measurement columns.
    - col1: str, name of the first measurement column.
    - col2: str, name of the second measurement column.
    - title: str, optional title for the plot.
    '''
    # Prepare data
    diff = data[col1] - data[col2]
    
    if gt_data:
        mean = data[col1]
    else:
        mean = data[[col1, col2]].mean(axis=1)
        
    diff_percent = diff / mean * 100
    
    
    mask = np.ones(len(data), dtype=bool)
    if outlier_method != 'none':
        if outlier_method == 'sd':
            mu, sigma = diff.mean(), diff.std()
            mask = ((diff >= mu - outlier_threshold * sigma) & (diff <= mu + outlier_threshold * sigma)) & mask
            mask = mask.to_numpy()
        elif outlier_method == 'iqr':
            q1, q3 = diff.quantile(0.25), diff.quantile(0.75)
            iqr = q3 - q1
            lower, upper = q1 - outlier_threshold * iqr, q3 + outlier_threshold * iqr
            mask = ((diff >= lower) & (diff <= upper)) & mask
            mask = mask.to_numpy()
        elif outlier_method == 'ransac':
            X = mean.values.reshape(-1, 1)
            y = diff.values
            model = RANSACRegressor(estimator=LinearRegression(), random_state=0)
            model.fit(X, y)
            mask = model.inlier_mask_
            mask = mask.to_numpy()
        else:
            raise ValueError('outlier_method must be "none", "sd", "iqr", or "ransac"')
        
    if outlier_method != 'none':
        if outlier_method == 'sd':
            mu, sigma = diff_percent.mean(), diff_percent.std()
            mask = ((diff_percent >= mu - outlier_threshold * sigma) & (diff_percent <= mu + outlier_threshold * sigma)) & mask
            mask = mask.to_numpy()
        elif outlier_method == 'iqr':
            q1, q3 = diff_percent.quantile(0.25), diff_percent.quantile(0.75)
            iqr = q3 - q1
            lower, upper = q1 - outlier_threshold * iqr, q3 + outlier_threshold * iqr
            mask = ((diff_percent >= lower) & (diff_percent <= upper)) & mask
            mask = mask.to_numpy()
        elif outlier_method == 'ransac':
            pass
        else:
            raise ValueError('outlier_method must be "none", "sd", "iqr", or "ransac"')

    num_outliers = len(diff_percent) - np.sum(mask)
    
    plot_df = data.copy()
    plot_df['mean'] = mean
    plot_df['diff'] = diff
    plot_df['diff_percent'] = diff_percent
    plot_df['status'] = np.where(mask, 'Inlier', 'Outlier')
    
    n = len(diff)
    
    inliers = plot_df[plot_df['status'] == 'Inlier']
    mean_diff = inliers['diff'].mean()
    std_diff = inliers['diff'].std(ddof=1)
    mean_diff_percent = inliers['diff_percent'].mean()
    std_diff_percent = inliers['diff_percent'].std(ddof=1)
    
    # Z-score for confidence level
    z_ci = norm.ppf(1 - (1 - ci) / 2)
    z_sd = norm.ppf(1 - (1 - sd_limit) / 2)

    # Limits of agreement
    loa_upper = mean_diff + z_sd * std_diff
    loa_lower = mean_diff - z_sd * std_diff
    loa_upper_percent = mean_diff_percent + z_sd * std_diff_percent
    loa_lower_percent = mean_diff_percent - z_sd * std_diff_percent

    # Standard error calculations
    se_diff = std_diff / np.sqrt(n)
    ci_mean = z_ci * se_diff
    se_diff_percent = std_diff_percent / np.sqrt(n)
    ci_mean_percent = z_ci * (se_diff_percent / np.sqrt(n))

    # CI for limits of agreement (approximation by Bland & Altman)
    se_loa = std_diff * np.sqrt(1/n + (z_sd**2) / (2 * (n - 1)))
    ci_loa = z_ci * se_loa
    se_loa_percent = std_diff_percent * np.sqrt(1/n + (z_sd**2) / (2 * (n - 1)))
    ci_loa_percent = z_ci * se_loa_percent

    # Plot
    num_subplots = len(plot_list)
    fig = plt.figure(figsize=(20, 8))
    list_of_subplots = []
    list_of_axes = []
    for idx, plot in enumerate(plot_list):
        plot_number = int('1' + str(num_subplots) + str(idx + 1))
        list_of_subplots.append(plt.subplot(plot_number))
        if plot == 'BA':
            list_of_axes.append(sns.scatterplot(data=plot_df, x='mean', y='diff', hue='status', palette={'Inlier': 'tab:blue', 'Outlier': 'tab:red'}, ax=list_of_subplots[idx]))
            # Get x axis limits
            x_min = mean.min() - (mean.max() - mean.min()) * 0.05
            x_max = mean.max() + (mean.max() - mean.min()) * 0.05
            list_of_axes[idx].set_xlim((x_min, x_max))
            x_fill = np.linspace(x_min, x_max, 100)
            # Main lines
            plt.axhline(0, color='tab:gray', linestyle='-', alpha=0.4, label='Zero difference')
            plt.axhline(mean_diff, color='tab:blue', linestyle='--', alpha=0.6, label='Mean difference')
            plt.axhline(loa_upper, color='tab:red', linestyle='--', alpha=0.6, label='Upper LOA')
            plt.axhline(loa_lower, color='tab:red', linestyle='--', alpha=0.6, label='Lower LOA')
            # Confidence intervals
            plt.fill_between(x_fill, mean_diff - ci_mean, mean_diff + ci_mean, color='tab:blue', alpha=0.15, label=f'{ci*100:.0f}% CI (mean)')
            plt.fill_between(x_fill, loa_upper - ci_loa, loa_upper + ci_loa, color='tab:red', alpha=0.15, label=f'{ci*100:.0f}% CI (upper LOA)')
            plt.fill_between(x_fill, loa_lower - ci_loa, loa_lower + ci_loa, color='tab:red', alpha=0.15, label=f'{ci*100:.0f}% CI (lower LOA)')
            # Text annotations
            if annotate:
                plt.text(x_max * 0.98, mean_diff, f'Mean\n{mean_diff:.2f} {units}', color='tab:blue', va='center', ha='right')
                plt.text(x_max * 0.98, loa_upper, f'Upper LOA (+{z_sd:.2f} SD)\n{loa_upper:.2f} {units}', color='tab:red', va='center', ha='right')
                plt.text(x_max * 0.98, loa_lower, f'Lower LOA (-{z_sd:.2f} SD)\n{loa_lower:.2f} {units}', color='tab:red', va='center', ha='right')
            plt.title(title)
            if gt_data:
                plt.xlabel(f'Ground Truth Measurement ({units})')
            else:
                plt.xlabel(f'Mean of Measurements ({units})')
            plt.ylabel(f'Difference of Measurements ({units})')
            # plt.legend()
            plt.grid(False)
            plt.tight_layout()
        elif plot == 'BA_percent':
            list_of_axes.append(sns.scatterplot(data=plot_df, x='mean', y='diff_percent', hue='status', palette={'Inlier': 'tab:blue', 'Outlier': 'tab:red'}, ax=list_of_subplots[idx]))
            # Get x axis limits
            x_min = mean.min() - (mean.max() - mean.min()) * 0.05
            x_max = mean.max() + (mean.max() - mean.min()) * 0.05
            list_of_axes[idx].set_xlim((x_min, x_max))
            x_fill = np.linspace(x_min, x_max, 100)
            # Main lines
            plt.axhline(0, color='tab:gray', linestyle='-', alpha=0.4, label='Zero difference')
            plt.axhline(mean_diff_percent, color='tab:blue', linestyle='--', alpha=0.6, label='Mean difference')
            plt.axhline(loa_upper_percent, color='tab:red', linestyle='--', alpha=0.6, label='Upper LOA')
            plt.axhline(loa_lower_percent, color='tab:red', linestyle='--', alpha=0.6, label='Lower LOA')
            # Confidence intervals
            plt.fill_between(x_fill, mean_diff_percent - ci_mean_percent, mean_diff_percent + ci_mean_percent, color='tab:blue', alpha=0.15, label=f'{ci*100:.0f}% CI (mean)')
            plt.fill_between(x_fill, loa_upper_percent - ci_loa_percent, loa_upper_percent + ci_loa_percent, color='tab:red', alpha=0.15, label=f'{ci*100:.0f}% CI (upper LOA)')
            plt.fill_between(x_fill, loa_lower_percent - ci_loa_percent, loa_lower_percent + ci_loa_percent, color='tab:red', alpha=0.15, label=f'{ci*100:.0f}% CI (lower LOA)')
            # Text annotations
            if annotate:
                plt.text(x_max * 0.98, mean_diff_percent, f'Mean\n{mean_diff_percent:.2f} %', color='tab:blue', va='center', ha='right')
                plt.text(x_max * 0.98, loa_upper_percent, f'Upper LOA (+{z_sd:.2f} SD)\n{loa_upper_percent:.2f} %', color='tab:red', va='center', ha='right')
                plt.text(x_max * 0.98, loa_lower_percent, f'Lower LOA (-{z_sd:.2f} SD)\n{loa_lower_percent:.2f} %', color='tab:red', va='center', ha='right')
            plt.title(title)
            if gt_data:
                plt.xlabel(f'Ground Truth Measurement ({units})')
            else:
                plt.xlabel(f'Mean of Measurements ({units})')
            plt.ylabel(f'Percentage Difference of Measurements (%)')
            # plt.legend()
            plt.grid(False)
            plt.tight_layout()
        elif plot == 'regression':
            list_of_axes.append(sns.scatterplot(data=plot_df, x=col1, y=col2, hue='status', palette={'Inlier': 'tab:blue', 'Outlier': 'tab:red'}, ax=list_of_subplots[idx]))
            list_of_axes[idx].set(xlabel=f'{col1} ({units})', ylabel=f'{col2} ({units})')
            corr_res = pg.corr(data[col1][mask], data[col2][mask], method='spearman')
            r2 = corr_res['r']['spearman']**2
            pval = corr_res['p-val']['spearman']
            pwr = corr_res['power']['spearman']
            list_of_axes[idx].annotate(f'n = {n:.0f}    # outliers = {num_outliers:.0f}', xy=(0.05, 0.95), xycoords='axes fraction', fontsize=12, ha='left', va='top', color='black')
            list_of_axes[idx].annotate(f'R2 = {r2:.2f}    p = {pval:.2e}    power = {pwr:.2e}', xy=(0.05, 0.93), xycoords='axes fraction', fontsize=12, ha='left', va='top', color='black')
            slope, intercept = np.polyfit(data[col1][mask], data[col2][mask], 1)
            list_of_axes[idx].annotate(f'y = {slope:.2f}x + {intercept:.2f}', xy=(0.05, 0.91), xycoords='axes fraction', fontsize=12, ha='left', va='top', color='black')
            plt.title(f'Regression between {col1} and {col2}')
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    else:
        plt.show()


if __name__ == '__main__':
    main_path = r'D:/SmartBrain/V1_5md_compare'
    orig_path = r'D:/SmartBrain/pro smartbrain'
    raters_list = ['PO', 'MM']
    rater_scratch_path = r'D:/SmartBrain/PO Manual MPR'
    sb_path = r'D:/SmartBrain/SmartBrain_results_v1_5/result_batch/dicoms'
    cq_corr_data_path = r'D:/CQ500_corrected'
    cq_sb_data_path = r'D:/CQ500SB'
    
    # metadata_table = get_metadata(orig_path)
    # metadata_table.to_excel('metadata_fnusa.xlsx')
    metadata_table = pd.read_excel('metadata_fnusa.xlsx')
    
    # results_table_cq_dataset = compute_angle_errors_cq_dataset(cq_sb_data_path, cq_corr_data_path)
    # results_table_cq_dataset.to_excel('results_table_CQ500_new.xlsx')
    results_table_experts_cq = pd.read_excel('results_table_CQ500_new.xlsx',
                                          usecols=['Study/Series', 'Coronal', 'Sagittal', 'Axial', 'CoronalSB', 'SagittalSB', 'AxialSB'])
    
    # results_table_experts = compute_angle_errors_sb_expert(main_path, raters_list)
    # results_table_experts.to_excel('results_table_SB_vs_experts.xlsx')
    results_table_experts = pd.read_excel('results_table_SB_vs_experts.xlsx',
                                          usecols=['Study/Series', 'Coronal1', 'Sagittal1', 'Axial1', 'Coronal2', 'Sagittal2', 'Axial2'])
    
    # results_table_radiologist = compute_angle_errors_sb_radiologist(sb_path, main_path + 'MPR')
    # results_table_radiologist.to_excel('results_table_SB_vs_radiol.xlsx')
    results_table_radiologist = pd.read_excel('results_table_SB_vs_radiol.xlsx',
                                              usecols=['Study/Series', 'CoronalSB', 'SagittalSB', 'AxialSB', 'CoronalR', 'SagittalR', 'AxialR'])
    
    # results_table_intrarater = compute_angle_errors_intrarater(main_path + raters_list[0], rater_scratch_path, sb_path, results_table_experts)
    # results_table_intrarater.to_excel('results_table_SB_vs_experts_and_intra.xlsx')
    results_table_experts = pd.read_excel('results_table_SB_vs_experts_and_intra.xlsx',
                                          usecols=['Study/Series', 'Coronal1', 'Sagittal1', 'Axial1', 'Coronal2', 'Sagittal2', 'Axial2',
                                                   'Coronal1B', 'Sagittal1B', 'Axial1B'])
    
    results_table_combined = results_table_experts.merge(results_table_radiologist, how='left', on='Study/Series')
    
    results_table_combined['CoronalDiffSBR'] =  results_table_combined['CoronalR'].to_numpy() - results_table_combined['CoronalSB'].to_numpy()
    results_table_combined['SagittalDiffSBR'] =   results_table_combined['SagittalR'].to_numpy() - results_table_combined['SagittalSB'].to_numpy()
    results_table_combined['AxialDiffSBR'] =  results_table_combined['AxialR'].to_numpy() - results_table_combined['AxialSB'].to_numpy()
    results_table_combined['CoronalDiff12'] =  results_table_combined['Coronal1'].to_numpy() - results_table_combined['Coronal2'].to_numpy()
    results_table_combined['SagittalDiff12'] =  results_table_combined['Sagittal1'].to_numpy() - results_table_combined['Sagittal2'].to_numpy()
    results_table_combined['AxialDiff12'] =  results_table_combined['Axial1'].to_numpy() - results_table_combined['Axial2'].to_numpy()
    results_table_combined['CoronalDiff1R'] = results_table_combined['CoronalR'].to_numpy() - (results_table_combined['Coronal1'].to_numpy() + results_table_combined['CoronalSB'].to_numpy())
    results_table_combined['SagittalDiff1R'] =  results_table_combined['SagittalR'].to_numpy() - (results_table_combined['Sagittal1'].to_numpy() + results_table_combined['SagittalSB'].to_numpy())
    results_table_combined['AxialDiff1R'] =  results_table_combined['AxialR'].to_numpy() - (results_table_combined['Axial1'].to_numpy() + results_table_combined['AxialSB'].to_numpy())
    results_table_combined['CoronalDiff2R'] = results_table_combined['CoronalR'].to_numpy() - (results_table_combined['Coronal2'].to_numpy() + results_table_combined['CoronalSB'].to_numpy())
    results_table_combined['SagittalDiff2R'] =  results_table_combined['SagittalR'].to_numpy() - (results_table_combined['Sagittal2'].to_numpy() + results_table_combined['SagittalSB'].to_numpy())
    results_table_combined['AxialDiff2R'] =  results_table_combined['AxialR'].to_numpy() - (results_table_combined['Axial2'].to_numpy() + results_table_combined['AxialSB'].to_numpy())
    results_table_combined['CoronalDiffI1'] =  results_table_combined['CoronalSB'].to_numpy() + results_table_combined['Coronal1'].to_numpy() - results_table_combined['Coronal1B'].to_numpy()
    results_table_combined['SagittalDiffI1'] =  results_table_combined['SagittalSB'].to_numpy() + results_table_combined['Sagittal1'].to_numpy() - results_table_combined['Sagittal1B'].to_numpy()
    results_table_combined['AxialDiffI1'] = results_table_combined['AxialSB'].to_numpy() + results_table_combined['Axial1'].to_numpy() - results_table_combined['Axial1B'].to_numpy()
    results_table_combined.rename(columns={'Coronal1': 'CoronalDiffSB1', 'Sagittal1': 'SagittalDiffSB1', 'Axial1': 'AxialDiffSB1',
                                         'Coronal2': 'CoronalDiffSB2', 'Sagittal2': 'SagittalDiffSB2', 'Axial2': 'AxialDiffSB2'}, inplace=True)
    results_table_combined['Coronal1'] = results_table_combined['CoronalDiffSB1'].to_numpy() + results_table_combined['CoronalSB'].to_numpy()
    results_table_combined['Coronal2'] = results_table_combined['CoronalDiffSB2'].to_numpy() + results_table_combined['CoronalSB'].to_numpy()
    results_table_combined['Sagittal1'] = results_table_combined['SagittalDiffSB1'].to_numpy() + results_table_combined['SagittalSB'].to_numpy()
    results_table_combined['Sagittal2'] = results_table_combined['SagittalDiffSB2'].to_numpy() + results_table_combined['SagittalSB'].to_numpy()
    results_table_combined['Axial1'] = results_table_combined['AxialDiffSB1'].to_numpy() + results_table_combined['AxialSB'].to_numpy()
    results_table_combined['Axial2'] = results_table_combined['AxialDiffSB2'].to_numpy() + results_table_combined['AxialSB'].to_numpy()
    results_table_combined['Coronal12mean'] = (results_table_combined['Coronal1'].to_numpy() + results_table_combined['Coronal2'].to_numpy()) / 2
    results_table_combined['Sagittal12mean'] = (results_table_combined['Sagittal1'].to_numpy() + results_table_combined['Sagittal2'].to_numpy()) / 2
    results_table_combined['Axial12mean'] = (results_table_combined['Axial1'].to_numpy() + results_table_combined['Axial2'].to_numpy()) / 2
    # print(results_table_combined)
    
    results_table_experts_cq['CoronalE'] =  results_table_experts_cq['CoronalSB'].to_numpy() + results_table_experts_cq['Coronal'].to_numpy()
    results_table_experts_cq['AxialE'] =  results_table_experts_cq['AxialSB'].to_numpy() + results_table_experts_cq['Axial'].to_numpy()
    results_table_experts_cq['SagittalE'] =  results_table_experts_cq['SagittalSB'].to_numpy() + results_table_experts_cq['Sagittal'].to_numpy()
    
    ylim = (-40, 40)
    xticklist = ['roll (y-axis - coronal view)', 'pitch (x-axis - sagittal view)', 'yaw (z-axis - axial view)']
    
    plt.figure()
    ax = sns.histplot(metadata_table[['PatientAge']])
    ax.set(xlabel='Age (years)',
        ylabel='Count (-)',
        title='Age distribution')
    # plt.show()
    plt.savefig('age_distribution.png', bbox_inches='tight')
    
    plt.figure()
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
    
    plt.figure()
    ax = sns.catplot(data=metadata_table, x='FilterType', kind='count')
    ax.set(xlabel='Filter Type (-)',
        ylabel='Count (-)',
        title='Filter type distribution')
    # plt.show()
    plt.savefig('filter_type_distribution.png', bbox_inches='tight')
    
    plt.figure()
    ax = sns.catplot(data=metadata_table, x='AcquisitionDate', kind='count')
    ax.set(xlabel='Date (-)',
        ylabel='Count (-)',
        title='Acquisition date distribution')
    ax.set_xticklabels(rotation=90)
    # plt.show()
    plt.savefig('acquisition_date_distribution.png', bbox_inches='tight')
    
    plt.close('all')
    
    results_to_plot = results_table_combined[['Coronal1', 'Sagittal1', 'Axial1', 'Coronal2', 'Sagittal2', 'Axial2']].copy()
    results_to_plot = results_to_plot.dropna()
    
    bland_altman_plot(results_to_plot, 'Coronal1', 'Coronal2', title='Bland-Altman plot Coronal (roll) E1 vs E2', outlier_method='iqr', save_path='BA_1_2_cor.png')
    bland_altman_plot(results_to_plot, 'Sagittal1', 'Sagittal2', title='Bland-Altman plot Sagittal (pitch) E1 vs E2', outlier_method='iqr', save_path='BA_1_2_sag.png')
    bland_altman_plot(results_to_plot, 'Axial1', 'Axial2', title='Bland-Altman plot Axial (yaw) E1 vs E2', outlier_method='iqr', save_path='BA_1_2_axi.png')
    
    results_to_plot = results_table_combined[['CoronalR', 'SagittalR', 'AxialR', 'CoronalSB', 'SagittalSB', 'AxialSB']].copy()
    results_to_plot = results_to_plot.dropna()
    
    bland_altman_plot(results_to_plot, 'CoronalR', 'CoronalSB', title='Bland-Altman plot Coronal (roll) R vs SB', outlier_method='iqr', gt_data=True, save_path='BA_R_SB_cor.png')
    bland_altman_plot(results_to_plot, 'SagittalR', 'SagittalSB', title='Bland-Altman plot Sagittal (pitch) R vs SB', outlier_method='iqr', gt_data=True, save_path='BA_R_SB_sag.png')
    bland_altman_plot(results_to_plot, 'AxialR', 'AxialSB', title='Bland-Altman plot Axial (yaw) R vs SB', outlier_method='iqr', gt_data=True, save_path='BA_R_SB_axi.png')
    
    results_to_plot = results_table_combined[['Coronal12mean', 'Sagittal12mean', 'Axial12mean', 'CoronalSB', 'SagittalSB', 'AxialSB']].copy()
    results_to_plot = results_to_plot.dropna()
    
    bland_altman_plot(results_to_plot, 'Coronal12mean', 'CoronalSB', title='Bland-Altman plot Coronal (roll) E vs SB', outlier_method='iqr', gt_data=True, save_path='BA_E_SB_cor.png')
    bland_altman_plot(results_to_plot, 'Sagittal12mean', 'SagittalSB', title='Bland-Altman plot Sagittal (pitch) E vs SB', outlier_method='iqr', gt_data=True, save_path='BA_E_SB_sag.png')
    bland_altman_plot(results_to_plot, 'Axial12mean', 'AxialSB', title='Bland-Altman plot Axial (yaw) E vs SB', outlier_method='iqr', gt_data=True, save_path='BA_E_SB_axi.png')
    
    results_to_plot = results_table_combined[['Coronal12mean', 'Sagittal12mean', 'Axial12mean', 'CoronalR', 'SagittalR', 'AxialR']].copy()
    results_to_plot = results_to_plot.dropna()
    
    bland_altman_plot(results_to_plot, 'Coronal12mean', 'CoronalR', title='Bland-Altman plot Coronal (roll) E vs R', outlier_method='iqr', gt_data=True, save_path='BA_E_R_cor.png')
    bland_altman_plot(results_to_plot, 'Sagittal12mean', 'SagittalR', title='Bland-Altman plot Sagittal (pitch) E vs R', outlier_method='iqr', gt_data=True, save_path='BA_E_R_sag.png')
    bland_altman_plot(results_to_plot, 'Axial12mean', 'AxialR', title='Bland-Altman plot Axial (yaw) E vs R', outlier_method='iqr', gt_data=True, save_path='BA_E_R_axi.png')
    
    results_to_plot = results_table_combined[['Coronal1', 'Sagittal1', 'Axial1', 'Coronal1B', 'Sagittal1B', 'Axial1B']].copy()
    results_to_plot = results_to_plot.dropna()
    
    bland_altman_plot(results_to_plot, 'Coronal1', 'Coronal1B', title='Bland-Altman plot Coronal (roll) E1a vs E1b', outlier_method='iqr', save_path='BA_1A_1B_cor.png')
    bland_altman_plot(results_to_plot, 'Sagittal1', 'Sagittal1B', title='Bland-Altman plot Sagittal (pitch) E1a vs E1b', outlier_method='iqr', save_path='BA_1A_1B_sag.png')
    bland_altman_plot(results_to_plot, 'Axial1', 'Axial1B', title='Bland-Altman plot Axial (yaw) E1a vs E1b', outlier_method='iqr', save_path='BA_1A_1B_axi.png')
    
    bland_altman_plot(results_table_experts_cq, 'CoronalE', 'CoronalSB', title='Bland-Altman plot Coronal (roll) CQ500', gt_data=True, outlier_method='none', save_path='BA_CQ500_cor.png')
    bland_altman_plot(results_table_experts_cq, 'SagittalE', 'SagittalSB', title='Bland-Altman plot Sagittal (pitch) CQ500', gt_data=True, outlier_method='none', save_path='BA_CQ500_sag.png')
    bland_altman_plot(results_table_experts_cq, 'AxialE', 'AxialSB', title='Bland-Altman plot Axial (yaw) CQ500', gt_data=True, outlier_method='none', save_path='BA_CQ500_axi.png')
    