import os
import argparse


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
    parser.add_argument('--file_path', '--f', type=str, default='', help='file path')
    parser.add_argument('--txt_name', '--t', type=str, default='', help='output file name')

    args = parser.parse_args()

    write_txt(args.file_path, args.txt_name)
