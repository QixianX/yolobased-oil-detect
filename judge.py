# -*- coding:utf8 -*-
import cv2
#import os
import numpy as np
def detect_f(inputimg):    

    input_img = inputimg

    # 转ycrcb提取y通道
    iycrcb = cv2.cvtColor(input_img,cv2.COLOR_BGR2YCrCb)
    iycrcbchannels = cv2.split(iycrcb)
    iy = iycrcbchannels[0]
    #cv2.imshow("y",iy)
    #cv2.waitKey()
    #cv2.imwrite('ych.jpg',iy)
    #cv2.imwrite('iycrcb.jpg',iycrcb)



    ret, im = cv2.threshold(iy, 100, 255, 0)
    #cv2.imshow("bim",im)
    #cv2.waitKey()
    params = cv2.SimpleBlobDetector_Params()

    # 膨胀
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(6, 6))
    dilated = cv2.dilate(im,kernel) 
    #cv2.imshow("dilated",dilated)
    #cv2.waitKey()

    # 调整阈值
    params.minThreshold = 100
    params.maxThreshold = 255

    # 颜色
    params.filterByColor = True
    params.blobColor = 255

    # 面积大小
    params.filterByArea = True
    params.minArea = 4

    # 形状（凸）
    params.filterByCircularity = True
    params.minCircularity = 0.25

    # 形状（凹）
    params.filterByConvexity = True
    params.minConvexity = 0.25

    # 形状（圆）
    params.filterByInertia = True
    params.minInertiaRatio = 0.25


    # 创建检测器
    detector = cv2.SimpleBlobDetector_create(params) 

    # 检测斑点
    keypoints = detector.detect(dilated)
    #print(len(keypoints))
    # 标记出斑点
    kpim = cv2.drawKeypoints(iycrcbchannels[0], keypoints, np.array([]), (0, 0, 255),cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    #cv2.imshow("Keypoints",kpim)
    #cv2.waitKey(0)
    if len(keypoints) > 0:
        k = 1
    else:
        k = 0
    return k