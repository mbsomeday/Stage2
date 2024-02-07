# Deal with .seq format for video sequence
# Author: Kaij
# The .seq file is combined with images,
# so I split the file into several images with the image prefix
# "\xFF\xD8\xFF\xE0\x00\x10\x4A\x46\x49\x46".

import os.path
import fnmatch
import shutil


def open_save(file, savepath):
    # read .seq file, and save the images into the savepath

    f = open(file, 'rb+')
    string = f.read().decode('latin-1')
    # PNG
    # splitstring = "\x89\x50\x4e\x47\x0d\x0a\x1a\x0a"
    splitstring = "\xFF\xD8\xFF\xE0\x00\x10\x4A\x46\x49\x46"
    # split .seq file into segment with the image prefix
    strlist = string.split(splitstring)
    f.close()
    count = 0
    # delete the image folder path if it exists
    if os.path.exists(savepath):
        shutil.rmtree(savepath)
    # create the image folder path
    if not os.path.exists(savepath):
        os.mkdir(savepath)
    # deal with file segment, every segment is an image except the first one
    for img in strlist:
        filename = str(count) + '.jpg'
        filenamewithpath = os.path.join(savepath, filename)
        # abandon the first one, which is filled with .seq header
        if count > 0:
            i = open(filenamewithpath, 'wb+')
            i.write(splitstring.encode('latin-1'))
            i.write(img.encode('latin-1'))
            i.close()
        count += 1


if __name__ == "__main__":
    file = r'D:\chrom_download\CUHK\set00-occ.seq'
    save_path = r'D:\chrom_download\CUHK\images'

    open_save(file, save_path)

    seq_dir = r'D:\chrom_download\CUHK\seq'
    image_dir = r'D:\chrom_download\CUHK\images'

    seq_list = os.listdir(seq_dir)

    for seq in seq_list:
        dir_name = seq.split('.')[0]

        dir_path = os.path.join(image_dir, dir_name)
        if not os.path.exists(dir_path):
            os.mkdir(dir_path)

        open_save(os.path.join(seq_dir, seq), dir_path)















