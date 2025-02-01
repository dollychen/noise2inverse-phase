import os
import PIL.Image
import io
import numpy as np
import cv2
import glob
import glymur
import pdb
import argparse
from skimage.transform import downscale_local_mean


def get_args():
    parser = argparse.ArgumentParser(description="Binning script")
    parser.add_argument("-input_path",type=str, help="Path to the folder containing the images to bin", 
                        default = 'D:/Dolly\A.Study\predictions/noise2inverse\kidney_scan_all_20230723-010507\epoch_10/test_output_20240624-171144\cropped')
    parser.add_argument("-output_folder", type=str, help="Path to the folder where the binned images should be saved",
                         default = 'D:/Dolly\A.Study\predictions/noise2inverse\kidney_scan_all_20230723-010507\epoch_10/test_output_20240624-171144\cropped_binned')
    parser.add_argument("-bin_factor", type=int, help="Bin factor",
                        default = 2)
    parser.add_argument("-file_type", type=str, choices=["tif", "jp2"], help="File type (tif or jp2)",
                        default = "tif")
    parser.add_argument("-prefix", type=str, help="Prefix for the output file names",
                        default = "HA-900_25.14um_LADAF-2021-17_kidney_left_0_pag-0.01_0.92_split_2_0_denoised_cropped_bin_2")
    return parser.parse_args()


def binning(input_path, output_folder, bin_factor, file_type, prefix):
    # Find all images in the specified folder with the specified file type
    im_list = glob.glob(f"{input_path}/*.{file_type}")
    print(str(len(im_list)) + ' images were found.')
    
    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Read the first image in the list to get the height and width of the images
    im_tmp = cv2.imread(im_list[0], -1)
    h, w = im_tmp.shape
    del im_tmp
    
    # Loop through the images in groups of bin_factor


    im_list_blocks = [im_list[i:i+bin_factor] for i in range(0, len(im_list), bin_factor)] 
    for idx, im_block in enumerate(im_list_blocks):
        file_name = f"{output_folder}/{prefix}_{idx:05}.{file_type}"
        
        # # Skip if the binned image already exists
        # if os.path.isfile(file_name):
        #     print('Slice already present')
        #     continue
        
        # Create a numpy array of zeros with the shape of the binned image
        array = np.zeros((len(im_block), h, w), dtype="uint16")
        
        # Loop through the images in the group and add them to the numpy array
        for c, img_path in enumerate(im_block):
            #print(f"Reading: slice {im_list[idx+c]}")
            im = cv2.imread(img_path,-1)
            #print(im_block)
            array[c, :, :] = im
        
        # Take the mean of the numpy array to get the binned image
        array = array.mean(axis=0)
        
        # Resize the binned image using PIL
        # im = PIL.Image.fromarray(array)
        # im = im.resize(
        #     (int(array.shape[1] / bin_factor),
        #      int(array.shape[0] / bin_factor)),
        #     PIL.Image.BICUBIC,
        # )

        #resize uisng skimage.transform
        im = downscale_local_mean(array, (bin_factor, bin_factor))
        
        # Save the binned image in the output folder
        print("Writing: " + file_name)
        im = np.array(im).astype("uint16")
        if file_type == "tif":
            cv2.imwrite(file_name, im)
        elif file_type == "jp2":
            ratio_compression = 10
            glymur.Jp2k(file_name, data=np.array(im), cratios=[ratio_compression])


def main():
    # Get the input and output folder paths from the command line arguments
    args = get_args()
    input_path = args.input_path
    output_folder = args.output_folder
    
    # Get the bin factor and file type from the command line arguments
    bin_factor = args.bin_factor
    file_type = args.file_type
    prefix = args.prefix
    # Run the binning function
    binning(input_path, output_folder, bin_factor, file_type, prefix)
    
    print('Finished')


if __name__ == "__main__":
    main()