import os
import random
import argparse
from PIL import Image

# rename file
def file_rename(path, name, num, file_type):
    count = 0
    print('Generating new filename such as:' + name + num + file_type)
    filenames = os.listdir(path)
    for files in filenames:
        old_name_path = os.path.join(path, files)
        if os.path.isdir(old_name_path):
            name = files
            old_name = os.listdir(old_name_path)
            for sub_path in old_name:
                full_img_path = os.path.join(old_name_path, sub_path)
                if not os.path.isdir(full_img_path):
                    new_name = os.path.join(old_name_path, name + str(count + int(num)) + file_type)
                    if os.path.isdir(new_name):
                        # count += 1
                        new_name = os.path.join(new_name, name + str(count + int(num)) + file_type)
                    print(new_name)
                    # if old_name.endswith('.png') or old_name.endswith('.bmp') :
                    im1 = Image.open(full_img_path).convert('RGB')
                    im1.save(new_name)
                    os.remove(full_img_path)
                    # else:
                    #     os.rename(old_name, new_name)
                    count += 1
    print(str(count) + 'files renamed')


# make txt file
def write_txt(file_path, txt_path, test_percent):
    totals = os.listdir(file_path)
    with open(txt_path, 'w') as f:
        for sub_path in totals:
            file_path_jpg = os.path.join(file_path,sub_path)
            if os.path.isdir(file_path_jpg):
                sub_total = os.listdir(file_path_jpg)
                num = len(sub_total)
                list = range(num)
                test = int(test_percent * num)
                testlist = random.sample(list, test)

                for i in list:
                    name = sub_total[i]
                    if i in testlist:
                        out_path = os.path.join(file_path_jpg, name)
                        # print(out_path)
                        f.write(out_path + '\n')
    f.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_path', '--f', type=str, default='', help='file path')
    parser.add_argument('--txt_name', '--t', type=str, default='', help='output file name')
    parser.add_argument('--start_name', '--n', type=str, default='', help='start name')
    parser.add_argument('--start_num', '--b', type=str, default='', help='start num')
    parser.add_argument('--file_type', type=str, default='.jpg', help='file type such as:.jpg')
    parser.add_argument('--options', '--o', type=str, default='renam', help='options rename or write')
    parser.add_argument('--percent', type=float, default=1, help='test image number percent')

    args = parser.parse_args()
    if args.options == 'rename':
        file_rename(args.file_path, args.start_name, args.start_num, args.file_type)
    else:
        write_txt(args.file_path, args.txt_name, args.percent)
