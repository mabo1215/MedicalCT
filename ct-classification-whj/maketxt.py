import os
import argparse


def file_rename(path, name, num, ftype):
    paths = path
    start_name = name
    start_num = num
    file_type = ftype
    count = 0
    print('Generating new filename such as:' + start_name + start_num + file_type)
    filenames = os.listdir(paths)
    for files in filenames:
        old_name = os.path.join(paths, files)
        if os.path.isdir(old_name):
            continue
        new_name = os.path.join(paths, start_name + str(count + int(start_num)) + file_type)
        os.rename(old_name, new_name)
        count += 1
    print(str(count) + 'files renamed')


# 图片文件夹路径
def write_txt(file_path, txt):
    paths = file_path
    txtnames = txt
    f = open(txtnames, 'w')
    filenames = os.listdir(paths)
    filenames.sort()
    for filename in filenames:
        # 图片绝对路径
        out_path = paths + '/' + filename
        print(out_path)
        f.write(out_path + '\n')
    f.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_path', '--f', type=str, default='D:/Download/data/COVID-19 Dataset/Non-COVID',
                        help='file path')
    parser.add_argument('--txt_name', '--t', type=str, default='', help='output file name')
    parser.add_argument('--start_name', '--n', type=str, default='non-covid-', help='start name')
    parser.add_argument('--start_num', '--b', type=str, default='001', help='start num')
    parser.add_argument('--file_type', type=str, default='.png', help='file type such as:.jpg')
    parser.add_argument('--options', '--o', type=str, default='rename', help='options rename or write txt')

    args = parser.parse_args()
    if args.options == 'rename':
        file_rename(args.file_path, args.start_name, args.start_num, args.file_type)
    else:
        write_txt(args.file_path, args.txt_name)
