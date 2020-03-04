# -*- coding: utf-8 -*-
"""
Class definition of YOLO_v3 style detection model on image and video
"""

import colorsys
import os
from timeit import default_timer as timer
import cv2
import numpy as np
from keras import backend as K
from keras.models import load_model
from keras.layers import Input
from PIL import Image, ImageFont, ImageDraw

from yolo3.model import yolo_eval, yolo_body, tiny_yolo_body
from yolo3.utils import letterbox_image
import os
from keras.utils import multi_gpu_model
import getframe
import judge
class YOLO(object):
    _defaults = {
        "model_path": 'logs/000/trained_weights.h5',
        "anchors_path": 'model_data/yolo_anchors.txt',
        "classes_path": 'model_data/voc_classes.txt',
        "score" : 0.75,
        "iou" : 0.75,
        "model_image_size" : (416, 416),
        "gpu_num" : 1,
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults) # set up default values
        self.__dict__.update(kwargs) # and update with user overrides
        self.class_names = self._get_class()
        self.anchors = self._get_anchors()
        self.sess = K.get_session()
        self.boxes, self.scores, self.classes = self.generate()

    def _get_class(self):
        classes_path = os.path.expanduser(self.classes_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    def _get_anchors(self):
        anchors_path = os.path.expanduser(self.anchors_path)
        with open(anchors_path) as f:
            anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        return np.array(anchors).reshape(-1, 2)

    def generate(self):
        model_path = os.path.expanduser(self.model_path)
        assert model_path.endswith('.h5'), 'Keras model or weights must be a .h5 file.'

        # Load model, or construct model and load weights.
        num_anchors = len(self.anchors)
        num_classes = len(self.class_names)
        is_tiny_version = num_anchors==6 # default setting
        try:
            self.yolo_model = load_model(model_path, compile=False)
        except:
            self.yolo_model = tiny_yolo_body(Input(shape=(None,None,3)), num_anchors//2, num_classes) \
                if is_tiny_version else yolo_body(Input(shape=(None,None,3)), num_anchors//3, num_classes)
            self.yolo_model.load_weights(self.model_path) # make sure model, anchors and classes match
        else:
            assert self.yolo_model.layers[-1].output_shape[-1] == \
                num_anchors/len(self.yolo_model.output) * (num_classes + 5), \
                'Mismatch between model and given anchor and class sizes'

        print('{} model, anchors, and classes loaded.'.format(model_path))

        # Generate colors for drawing bounding boxes.
        hsv_tuples = [(x / len(self.class_names), 1., 1.)
                      for x in range(len(self.class_names))]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(
            map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                self.colors))
        np.random.seed(10101)  # Fixed seed for consistent colors across runs.
        np.random.shuffle(self.colors)  # Shuffle colors to decorrelate adjacent classes.
        np.random.seed(None)  # Reset seed to default.

        # Generate output tensor targets for filtered bounding boxes.
        self.input_image_shape = K.placeholder(shape=(2, ))
        if self.gpu_num>=2:
            self.yolo_model = multi_gpu_model(self.yolo_model, gpus=self.gpu_num)
        boxes, scores, classes = yolo_eval(self.yolo_model.output, self.anchors,
                len(self.class_names), self.input_image_shape,
                score_threshold=self.score, iou_threshold=self.iou)
        return boxes, scores, classes

    def detect_image(self, image):
        start = timer()

        #if self.model_image_size != (None, None):
        if self.model_image_size != (416, 416):
            assert self.model_image_size[0]%32 == 0, 'Multiples of 32 required'
            assert self.model_image_size[1]%32 == 0, 'Multiples of 32 required'
            boxed_image = letterbox_image(image, tuple(reversed(self.model_image_size)))
        else:
            new_image_size = (image.width - (image.width % 32),
                              image.height - (image.height % 32))
            boxed_image = letterbox_image(image, new_image_size)
        image_data = np.array(boxed_image, dtype='float32')

        print(image_data.shape)
        image_data /= 255.
        image_data = np.expand_dims(image_data, 0)  # Add batch dimension.

        out_boxes, out_scores, out_classes = self.sess.run(
            [self.boxes, self.scores, self.classes],
            feed_dict={
                self.yolo_model.input: image_data,
                self.input_image_shape: [image.size[1], image.size[0]],
                K.learning_phase(): 0
            })

        print('Found {} boxes for {}'.format(len(out_boxes), 'img'))

        font = ImageFont.truetype(font='font/FiraMono-Medium.otf',
                    size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
        thickness = (image.size[0] + image.size[1]) // 300

        for i, c in reversed(list(enumerate(out_classes))):
            predicted_class = self.class_names[c]
            box = out_boxes[i]
            score = out_scores[i]

            label = '{} {:.2f}'.format(predicted_class, score)
            draw = ImageDraw.Draw(image)
            label_size = draw.textsize(label, font)

            top, left, bottom, right = box
            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
            right = min(image.size[0], np.floor(right + 0.5).astype('int32'))
            print(label, (left, top), (right, bottom))

            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])

            # My kingdom for a good redistributable image drawing library.
            for i in range(thickness):
                draw.rectangle(
                    [left + i, top + i, right - i, bottom - i],
                    outline=self.colors[c])
            draw.rectangle(
                [tuple(text_origin), tuple(text_origin + label_size)],
                fill=self.colors[c])
            draw.text(text_origin, label, fill=(0, 0, 0), font=font)
            del draw

            # 河道的裁剪,保证每次只裁剪一段河道
            if len(box)  < 5:
                x1 = left
                y1 = top
                wt = right-left
                ht = bottom-top
                #image1 = cv2.imread(path)
                image1 = cv2.cvtColor(np.asarray(image),cv2.COLOR_RGB2BGR) 
                # 河道裁剪的范围[左上坐标,右下坐标]
                cropi = image1[y1+10:bottom,x1-10:right]

                img_HSV = cv2.cvtColor(cropi,cv2.COLOR_BGR2HSV)#numpy类型
                #lower_river = np.array([90,10,85]) # 设定河面颜色的阈值
                lower_river = np.array([75,10,10]) # 设定河面颜色的阈值
                upper_river = np.array([120,75,140])
                mask = cv2.inRange(img_HSV,lower_river,upper_river)  # 根据阈值构建掩模   
                res = cv2.bitwise_and(cropi,cropi,mask=mask)#对原图像和掩模进行位运算
                dst = Image.fromarray(res, mode='RGB')
                
####################把掩膜的颜色进行替换，换成和河面一样的颜色############
                for ii in range(0,dst.size[0]):#遍历所有长度的点
                   for jj in range(0,dst.size[1]):#遍历所有宽度的点
                       data = (dst.getpixel((ii,jj)))#打印该图片的所有点
                       if (data[0]<=20 and data[1]<=20 and data[2]<=20):
                           dst.putpixel((ii,jj),(80,85,90,255))
                dst = np.array(dst)

                savedname = path.split('.')[0].split('/')[1] + '.jpg'
                savedpath = 'river' + '/'
                pathExists = os.path.exists(savedpath)
                if not pathExists:
                    os.makedirs(savedpath)

                flag = judge.detect_f(dst)
                if flag == 1:
                    cv2.imshow("river",cropi)
                    cv2.waitKey(3)
                    cv2.imwrite(savedpath + savedname, image1)
                    print("成功检测到杂物并将杂物河道保存至river文件夹")
                else:
                    print("当前帧对应的河道没有杂物")
            else:
                print("目标数量过多，已跳过")
                
        end = timer()
        print(end - start)
        return image

    def close_session(self):
        self.sess.close()

if __name__ == '__main__':

    yolo = YOLO()
    # 视频路径
    videopath = 'aim.mp4'
    imagelist = getframe.get_f(videopath)

    for path in imagelist:
        try:

            input_image = Image.open(path)          
            
        except:

            print('Open Error! Try again!')

        else:
            image = input_image.resize((416,416))
            r_image = yolo.detect_image(image)
            #r_image.show()
    yolo.close_session()
    print("已检测完所有帧")