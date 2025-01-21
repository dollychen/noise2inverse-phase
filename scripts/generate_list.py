"""
This script aims to generate a list of files used for training.
this should only generate image names, paths are added in the training script
"""
import argparse
import glob
import numpy as np
import os

def main(args):

    train_data_path = args.train_data_path
    file_ext = args.file_ext
    output_file = args.output_file
    include_all = args.include_all
    imgs_to_include = args.imgs_to_include
    scan_no = args.scan_no

    #check if the path exists
    assert os.path.exists(train_data_path), f'{train_data_path} Path does not exist'
    assert os.path.isdir(os.path.dirname(output_file)), f'{os.path.dirname(output_file)} is not a directory'

    if include_all:
        #get all images in the path
        images = glob.glob(train_data_path + '/*{}'.format(file_ext))
        #write to file
        with open(output_file, 'w+') as f:
            for image in images:
                f.write(os.path.basename(image) + '\n')

        return
    elif imgs_to_include != -1:
        #randomly select images from the path
        images = glob.glob(train_data_path + '/*{}'.format(file_ext))
        num_imgs = len(images)
        img_index = np.random.randint(0, num_imgs, size=imgs_to_include)

        #write to file
        with open(output_file, 'w+') as f:
            for i in img_index:
                f.write(os.path.basename(images[i]) + '\n')
        return
    
    elif scan_no != -1:
        #TODO: to be completed later, make note that scan number is 3 digits eg: _012__
        #get all images within a specific scan
        #images = glob.glob(train_data_path + '/*_scan_{}_*.{}'.format(scan_no, file_ext))
        images = glob.glob(train_data_path + '/*_0{}__*{}'.format(scan_no, file_ext))
        #write to file
        with open(output_file, 'w+') as f:
            for image in images:
                f.write(os.path.basename(image) + '\n')
        return


if __name__ == '__main__':
    # parse arguments
    parser = argparse.ArgumentParser(description='Generate train list')
    parser.add_argument('--train_data_path', type=str, help='path to training images', \
                        default='D:/Dolly\A.Study\data\kidney_left/2021_17/25.14um/recon\orig_recon_joseph/all\split_2/1')
    parser.add_argument('--file_ext', type=str, help='file extension', default='.tif')
    parser.add_argument('--output_file', type=str, help='path and name to output file', default='./configs/kidney/kidney_inf_data_list_all.txt')
    parser.add_argument('--include_all', type = bool, help = 'if all images are needed', default = True)
    parser.add_argument('--imgs_to_include', type = str, help = 'if a small list is needed set the number of image to randomly select,\
                         set to -1 if not needed', default = -1)
    parser.add_argument('--scan_no', type = int, help = 'if a specific scan is required, set to -1 if not needed', default = -1)
    args = parser.parse_args()

    main(args)