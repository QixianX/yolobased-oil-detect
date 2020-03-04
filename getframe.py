# -*- coding:utf8 -*-

import cv2

import os

import shutil

# 视频文件名字
def get_f(file):

    filename = file
    filelist = []
    # 保存图片的路径

    savedpath = filename.split('.')[0] + '/'

    isExists = os.path.exists(savedpath)

    if not isExists:

        os.makedirs(savedpath)

        print('path of %s is build' % (savedpath))

    else:

        shutil.rmtree(savedpath)

        os.makedirs(savedpath)

        print('path of %s already exist and rebuild' % (savedpath))

    # 视频帧率

    fps = 60

    # 保存图片的帧率间隔

    count = 80

    # 开始读视频

    videoCapture = cv2.VideoCapture(filename)

    i = 0

    j = 0

    while True:

        success, frame = videoCapture.read()

        i += 1

        if (i % count == 0):
            # 保存图片

            j += 1
            savedname = filename.split('.')[0] + '_' + str(i) + '_' + '.jpg'
            cv2.imwrite(savedpath + savedname, frame)
            print('image of %s is saved' % (savedname))
            filelist.append(savedpath+savedname)

        if not success:
            print('video is all read')

            break

    return filelist


