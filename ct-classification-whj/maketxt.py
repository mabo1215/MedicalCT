import os
import random
import argparse


# rename file
def file_rename(path, name, num, file_type):
    count = 0
    print('Generating new filename such as:' + name + num + file_type)
    filenames = os.listdir(path)
    for files in filenames:
        old_name = os.path.join(path, files)
        if os.path.isdir(old_name):
            continue
        new_name = os.path.join(path, name + str(count + int(num)) + file_type)
        os.rename(old_name, new_name)
        count += 1
    print(str(count) + 'files renamed')


# make txt file
def write_txt(file_path, txt, test_percent):
    total = os.listdir(file_path)
    num = len(total)
    list = range(num)
    test = int(test_percent * num)
    testlist = random.sample(list, test)
    f = open(txt, 'w')
    for i in list:
        name = total[i]
        if i in testlist:
            out_path = file_path + '/' + name
            print(out_path)
            f.write(out_path + '\n')
    f.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_path', '--f', type=str, default='', help='file path')
    parser.add_argument('--txt_name', '--t', type=str, default='', help='output file name')
    parser.add_argument('--start_name', '--n', type=str, default='', help='start name')
    parser.add_argument('--start_num', '--b', type=str, default='', help='start num')
    parser.add_argument('--file_type', type=str, default='', help='file type such as:.jpg')
    parser.add_argument('--options', '--o', type=str, default='', help='options rename or write')
    parser.add_argument('--percent', type=float, default=1, help='test image number percent')

    args = parser.parse_args()
    if args.options == 'rename':
        file_rename(args.file_path, args.start_name, args.start_num, args.file_type)
    else:
        write_txt(args.file_path, args.txt_name, args.percent)
