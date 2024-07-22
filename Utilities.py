# -*- coding: utf-8 -*-
"""
Created on Thu Aug 18 14:14:02 2022

@author: jakubicek
"""

import os
import numpy as np
import SimpleITK as sitk   
#import matplotlib.pyplot as plt
#from skimage import measure
# import torch
import time
import nibabel as nib
#from skimage.transform import rescale
#from skimage.filters import gaussian


def resample_image(itk_image, out_spacing=[1.0, 1.0, 1.0], is_label=False):
    original_spacing = itk_image.GetSpacing()
    original_size = itk_image.GetSize()

    out_size = [
        int(np.round(original_size[0] * (original_spacing[0] / out_spacing[0]))),
        int(np.round(original_size[1] * (original_spacing[1] / out_spacing[1]))),
        int(np.round(original_size[2] * (original_spacing[2] / out_spacing[2])))
    ]

    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(out_spacing)
    resample.SetSize(out_size)
    resample.SetOutputDirection(itk_image.GetDirection())
    resample.SetOutputOrigin(itk_image.GetOrigin())
    resample.SetTransform(sitk.Transform())
    resample.SetDefaultPixelValue(itk_image.GetPixelIDValue())

    if is_label:
        resample.SetInterpolator(sitk.sitkNearestNeighbor)
    else:
        resample.SetInterpolator(sitk.sitkBSpline)

    return resample.Execute(itk_image) 


def resize_pyramid(imgs, masks, scale, sigma, np_dtype):
    
    imgs_res = []
    masks_res = []
    for img, mask in zip(imgs,masks):
        if sigma > 0:
            img = gaussian(img,sigma)
            
        imgs_res.append(rescale(img, 1/scale, preserve_range=True).astype(np_dtype))
        masks_res.append(rescale(mask, 1/scale, preserve_range=True).astype(np_dtype))
        
    return np.stack(imgs_res,0), np.stack(masks_res,0) 


def resave_dicom(data_directory, out_dir, name, ser, info, bias=True):
    
    ## for load dicom
    series_IDs = sitk.ImageSeriesReader.GetGDCMSeriesIDs(data_directory+'\\')
    series_file_names = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(data_directory, series_IDs[0])
    
    series_file_names = series_file_names[info['vol']*ser :  info['vol']*(ser+1)]
    
    if info['Z-direction']:
        series_file_names = series_file_names[::-1]
        
    series_reader = sitk.ImageSeriesReader()
    series_reader.SetFileNames(series_file_names)
    series_reader.LoadPrivateTagsOn()
    image3D=series_reader.Execute()
    sizeImg = image3D.GetSize()
    # spacing = image3D.GetSpacing()
    
    # orig = series_reader.Execute()
    # size = orig.GetSize()

    # resamplig and rotating
    image_out = []
    image_out = sitk.Image( sizeImg , sitk.sitkInt16)
    image_out = sitk.DICOMOrient(series_reader.Execute(), 'LPS')
    image_out = resample_image(image_out, out_spacing=[1.0, 1.0, 1.0], is_label=False)
    
    if bias:
        corrector = sitk.N4BiasFieldCorrectionImageFilter()
        # corrector.SetMaximumNumberOfIterations((100,100,100,100))
        corrector.SetMaximumNumberOfIterations((100,50))
        corrector.SetNumberOfHistogramBins(10)
        image_out = sitk.Cast(image_out, sitk.sitkFloat32)
        image_out = corrector.Execute(image_out)
        
    image_out = sitk.Cast(image_out, sitk.sitkInt16)
    
    image_out.SetOrigin((0, 0, 0))
    image_out.SetSpacing((1,1,1))
    # image_out.SetDirection(tuple((0.0, 0.0, 1.0, -1.0, 0.0, 0.0, 0.0, 1.0, 0.0)))
    image_out.SetDirection(tuple((1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)))
    
    # sitk.Show(orig, debugOn=True)
    # sitk.Show(image_out, debugOn=True)
    
    ser_num = f"{ser:04d}"

    # A.SetOrigin((0, 0, 0))
    # A.SetSpacing((1,1,1))
    
    path_save_temp = out_dir + '\\' + name + '_' + ser_num + '.nii.gz'
    writer = sitk.ImageFileWriter()    
    writer.SetFileName(path_save_temp)
    writer.Execute(image_out)
    # sitk.WriteImage(image_out, path_save_temp)
    
    # return path_save_temp, info
    return path_save_temp

def check_orientation(ct_image, ct_arr):
    """
    Check the NIfTI orientation, and flip to  'RPS' if needed.
    :param ct_image: NIfTI file
    :param ct_arr: array file
    :return: array after flipping
    """
    x, y, z = nib.aff2axcodes(ct_image.affine)
    if x != 'R':
        ct_arr = nib.orientations.flip_axis(ct_arr, axis=0)
    if y != 'P':
        ct_arr = nib.orientations.flip_axis(ct_arr, axis=1)
    if z != 'S':
        ct_arr = nib.orientations.flip_axis(ct_arr, axis=2)
    return ct_arr


def read_nii(file_name, current_index):   
    
    file_reader = sitk.ImageFileReader()
    file_reader.SetImageIO("NiftiImageIO")
    file_reader.SetFileName(file_name)
    file_reader.ReadImageInformation()
    sizeA=file_reader.GetSize()
    
            
    if current_index==-1:

        img = sitk.GetArrayFromImage(file_reader.Execute())
        # img = np.transpose(img, (1,2,0))
        
    else:
        if (current_index<1 and current_index>0):
            current_index = int( np.round( current_index*sizeA[2] ))
        current_index =  (0, 0, current_index, 0)
        file_reader.ReadImageInformation()
        file_reader.SetExtractIndex(current_index)
        extract_size = (sizeA[0], sizeA[1], 1, 1)
        file_reader.SetExtractSize(extract_size)

        img = sitk.GetArrayFromImage(file_reader.Execute())
        img = np.squeeze(img)
    
    
    # img = np.pad(img,((addX[2],addX[3]),(addX[0],addX[1]),(0,0)),'constant',constant_values=(-1024, -1024))


    return img


def read_nii_info(file_name):   
    
    info = dict()
    file_reader = sitk.ImageFileReader()
    file_reader.SetFileName(file_name)
    file_reader.ReadImageInformation()
    sizeA = file_reader.GetSize()
    
    info['size'] = (sizeA[0], sizeA[1])
    info['slices'] = (sizeA[2])
    info['Z-direction'] = False
    
    return info


def write_nii(A, file_name):  
    # A = np.transpose(A, (2,0,1))
    A = sitk.GetImageFromArray(A)
    sitk.Cast( A , sitk.sitkInt16)
    # A.SetOrigin((0, 0, 0))
    # A.SetSpacing((1,1,1))
    writer = sitk.ImageFileWriter()    
    writer.SetFileName(file_name)
    writer.Execute(A)



def read_dicom_info(data_directory):   
    
    info = dict()
    series_IDs = sitk.ImageSeriesReader.GetGDCMSeriesIDs(data_directory+'\\')
    series_file_names = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(data_directory, series_IDs[0])
    
    slices_loc = []
    for i,item in enumerate(series_file_names):
        file_reader = sitk.ImageFileReader()
        file_reader.SetFileName(item)    
        file_reader.ReadImageInformation()
        slices_loc.append(file_reader.GetMetaData('0020|1041'))
    
    u_loc = unique(slices_loc)
    info['vol'] = len(u_loc)
    info['series'] =  len(slices_loc) / len(u_loc)
    info['Z-direction'] = (float(slices_loc[0])-float(slices_loc[1]))>0
    
    return info



def unique(list1):
  
    # initialize a null list
    unique_list = []
  
    # traverse for all elements
    for x in list1:
        # check if exists in unique_list or not
        if x not in unique_list:
            unique_list.append(x)

    return unique_list


def resize_with_padding(img, expected_size):
    delta_width = expected_size[0] - img.shape[0]
    delta_height = expected_size[1] - img.shape[1]
    pad_width = delta_width // 2
    pad_height = delta_height // 2
    padding = np.array([pad_width, pad_height, delta_width - pad_width, delta_height - pad_height])
    padding[padding<0]=0
    img = np.pad(img, [(padding[0], padding[2]), (padding[1], padding[3])], mode='constant')
    img = crop_center_2D(img, new_width=expected_size[0], new_height=expected_size[1])
    return img


def crop_center_2D(img, new_width=None, new_height=None):        
    width = img.shape[1]
    height = img.shape[0]
    if new_width is None:
        new_width = min(width, height)
    if new_height is None:
        new_height = min(width, height)  
        
    left = int(np.ceil((width - new_width) / 2))
    right = width - int(np.floor((width - new_width) / 2))
    top = int(np.ceil((height - new_height) / 2))
    bottom = height - int(np.floor((height - new_height) / 2))

    if len(img.shape) == 2:
        center_cropped_img = img[top:bottom, left:right]
        z = 1;
    else:
        center_cropped_img = img[top:bottom, left:right, ...]
        z = img.shape[2]   
        
    return center_cropped_img


def crop_center_3D(img,cropx,cropy,cropz):
    y,x,z = img.shape[0:3]
    startx = x//2-(cropx//2)
    starty = y//2-(cropy//2)
    startz = z//2-(cropz//2)
    return img[starty:starty+cropy,startx:startx+cropx,startz:startz+cropz]


def crop_center_3D_batch(img,cropx,cropy,cropz):
    img_cropped = np.zeros((cropx,cropy,cropz,img.shape[3]))
    for i in range(img.shape[3]):
        img_cropped[:,:,:,i] = crop_center_3D(img[:,:,:,i],cropx,cropy,cropz)
    return img_cropped


def display_reg(path_ref,path_mov,path_save_reg, sl):
    
    ##----- display results of reg ----- 
    # sl = 0.6
    img_ref = read_nii(path_ref, sl)
    img_mov = read_nii(path_mov, sl)
    img_reg = read_nii(path_save_reg, sl)
    
    img_ref = (( img_ref - img_ref.min() ) / (img_ref.max() - img_ref.min()) )
    img_mov = (( img_mov - img_mov.min() ) / (img_mov.max() - img_mov.min()) )
    img_reg = (( img_reg - img_reg.min() ) / (img_reg.max() - img_reg.min()) )
    
    
    img_mov = resize_with_padding(img_mov, img_ref.shape )
    
    # fig, axs = plt.subplots(2,2)
    # axs[0,0].imshow(img_ref, cmap='gray')
    # axs[0,1].imshow(img_mov, cmap='gray')
    # axs[1,0].imshow(img_ref, cmap='gray')
    # axs[1,1].imshow(img_reg, cmap='gray')
    
    img = np.zeros( (img_ref.shape[0],img_ref.shape[1],3) )
    img[:,:,0] = img_mov
    img[:,:,1] = img_ref
    img[:,:,2] = img_mov
    
    # plt.figure()
    # plt.imshow(img)
    
    fig, axs = plt.subplots(1,2)        
    axs[0].imshow(img)
    
    img = np.zeros( (img_ref.shape[0],img_ref.shape[1],3) )
    img[:,:,0] = img_reg
    img[:,:,1] = img_ref
    img[:,:,2] = img_reg
    
   
    # plt.figure()
    # plt.imshow(img)

    axs[1].imshow(img)
    plt.show()

   
    
def bound_3D(img, add):    
    # mask.ndim
    # g = np.where(mask.sum(axis=0)>0)
    loc = np.where(img>0)

    borders=[]
    for i in range(0,len(loc)):
        g = [int(loc[i].min())-add, int(loc[i].max())+add]
        
        if g[0]<0: g[0]=0
        if g[1]<0: g[1]=0
        if g[0]>np.size(img,i): g[0]=np.size(img,i)
        if g[1]>np.size(img,i): g[1]=np.size(img,i)
        
        borders.append(g[:])
    
    return borders


def bwareafilt(mask, n=1, area_range=(0, np.inf)):
    """Extract objects from binary image by size """
    # For openCV > 3.0 this can be changed to: areas_num, labels = cv2.connectedComponents(mask.astype(np.uint8))
    labels = measure.label(mask.astype('uint8'), background=0)
    area_idx = np.arange(1, np.max(labels) + 1)
    areas = np.array([np.sum(labels == i) for i in area_idx])
    inside_range_idx = np.logical_and(areas >= area_range[0], areas <= area_range[1])
    area_idx = area_idx[inside_range_idx]
    areas = areas[inside_range_idx]
    keep_idx = area_idx[np.argsort(areas)[::-1][0:n]]
    kept_areas = areas[np.argsort(areas)[::-1][0:n]]
    if np.size(kept_areas) == 0:
        kept_areas = np.array([0])
    if n == 1:
        kept_areas = kept_areas[0]
    kept_mask = np.isin(labels, keep_idx)
    return kept_mask, kept_areas



# def construct_transf_matrix(params, resize_factor, inds, device):
    
#     angle_rad, tx, ty = params
    
#     theta = torch.zeros((len(inds)), 3, 3).to(device)
#     theta[:,2,2] = 1

#     theta[:,0,0] = resize_factor * torch.cos(angle_rad[inds])
#     theta[:,1,1] = resize_factor * torch.cos(angle_rad[inds])
#     theta[:,0,1] = torch.sin(angle_rad[inds])
#     theta[:,1,0] = -torch.sin(angle_rad[inds])
    
#     theta[:,0,2] = tx[inds]
#     theta[:,1,2] = ty[inds]
    
#     return theta


# def construct_transf_matrix_3D(params, resize_factor, inds, device):
    
#     thx, thy, thz, tx, ty, tz = params
    
    
#     theta = torch.zeros((len(inds)), 4, 4).to(device)
#     theta[:,3,3] = 1
    
#     R_z = torch.zeros((len(inds)), 3, 3).to(device)
#     R_z[:,2,2] = 1
#     R_z[:,0,0] = torch.cos(thz[inds])
#     R_z[:,0,1] = -torch.sin(thz[inds])
#     R_z[:,1,0] = torch.sin(thz[inds])
#     R_z[:,1,1] =  torch.cos(thz[inds])
    
#     R_y = torch.zeros((len(inds)), 3, 3).to(device)
#     R_y[:,1,1] = 1
#     R_y[:,0,0] = torch.cos(thy[inds])
#     R_y[:,2,0] = -torch.sin(thy[inds])
#     R_y[:,0,2] = torch.sin(thy[inds])
#     R_y[:,2,2] =  torch.cos(thy[inds])
    
#     R_x = torch.zeros((len(inds)), 3, 3).to(device)
#     R_x[:,0,0] = 1
#     R_x[:,1,1] = torch.cos(thx[inds])
#     R_x[:,2,1] = -torch.sin(thx[inds])
#     R_x[:,1,2] = torch.sin(thx[inds])
#     R_x[:,2,2] =  torch.cos(thx[inds])
    
#     R = torch.bmm( torch.bmm(R_z, R_y), R_x )
#     # R = torch.bmm(R_z, R_y)
    
#     T = torch.zeros((len(inds)), 3, 3).to(device)
#     T[:,0,0] = resize_factor
#     T[:,1,1] = resize_factor
#     T[:,2,2] = resize_factor
 
#     theta[:,0:3,0:3] = torch.bmm( T , R )
    
#     theta[:,0,3] = tx[inds]
#     theta[:,1,3] = ty[inds]
#     theta[:,2,3] = tz[inds]

#     # theta[:,0,0] = resize_factor * torch.cos(angle_rad[inds])
#     # theta[:,1,1] = resize_factor * torch.cos(angle_rad[inds])
#     # theta[:,0,1] = torch.sin(angle_rad[inds])
#     # theta[:,1,0] = -torch.sin(angle_rad[inds])
    
#     # theta[:,0,2] = tx[inds]
#     # theta[:,1,2] = ty[inds]
    
#     return theta


def show_3D(imgs):
    num = imgs.shape[2]
    for i in range(num):
        plt.subplot(3, int(np.floor(num/3)), i + 1)
        plt.imshow(imgs[:,:,i])
        plt.gcf().set_size_inches(10, 10)
    plt.show()




class Randomizer():
    
    def __init__(self, num_batches, num_imgs):
        self.num_batches = num_batches
        self.num_imgs = num_imgs
        self.rand_inds = np.random.permutation(num_imgs)
        self.batch_num = 0
        
    
    def get_batch_inds(self, batch_num):
        ind_start = (self.num_imgs //  self.num_batches) * batch_num
        ind_stop = (self.num_imgs //  self.num_batches) * (batch_num+1)
        if batch_num == (self.num_batches - 1):
            ind_stop = self.num_imgs 
        inds = self.rand_inds[ind_start:ind_stop]
        
        return inds
        
    def __iter__(self):
        return self
        
    def __next__(self):
        self.batch_num  += 1
        
        if self.batch_num  == (self.num_batches + 1):
            raise StopIteration
            
        inds =  self.get_batch_inds(self.batch_num - 1)
        
        return inds
        
    def order_corection(self,output):
    
        order_corection = np.argsort(self.rand_inds)
        output = output[order_corection,:,:]
        
        return output
    
class Timer(object):
    def __init__(self, name=None):
        self.name = name

    def __enter__(self):
        self.tstart = time.time()

    def __exit__(self, type, value, traceback):
        if self.name:
            print('[%s]' % self.name,)
        print('Elapsed: %s' % (time.time() - self.tstart))
        
        
        
def merge_Nifti(files, path_dir, path_save):
    file_reader = sitk.ImageFileReader()
    file_reader.SetImageIO("NiftiImageIO")
    file_reader.SetFileName(path_dir + '\\' + files[0])
    imgP = file_reader.Execute()
    vel = imgP.GetSize()
    
    img = np.zeros((vel[0], vel[1], vel[2],len(files) ), dtype=(np.int16))
    # img = img.squeeze()
    # sImg = sitk.Image(list((vel[0],vel[1], vel[2],len(files)+1)), sitk.sitkInt16, 1 )
    
    for i,file in enumerate(files):
        file_reader = sitk.ImageFileReader()
        file_reader.SetImageIO("NiftiImageIO")
        file_reader.SetFileName(path_dir + '\\' + file)
        img1 = file_reader.Execute()
        img1 = sitk.GetArrayFromImage(img1).astype(np.int16)
        img1 = np.transpose(img1,(2,1,0))
        # img[:,:,:,i+1] = img1[BB[0][0]:BB[0][1], BB[1][0]:BB[1][1], BB[2][0]:BB[2][1]]
        img[:,:,:,i] = img1
        
        os.remove(path_dir + '\\' + file)
    
    OM = np.eye(4)*np.diag([-1,-1,1,1])
    OM[1,3] = 239
    data = nib.Nifti1Image(img, OM )  # Save axis for data (just identity)
    
    data.header.set_xyzt_units('mm')
    # data.header.set
    # data.header.SetDirection(imgP.GetDirection())
    data.to_filename(path_save)  # Save as NiBabel file