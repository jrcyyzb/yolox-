import copy
import inspect
import math

import cv2
import mmcv
import numpy as np
from numpy import random

from mmdet.core import PolygonMasks
from mmdet.core.evaluation.bbox_overlaps import bbox_overlaps
from ..builder import PIPELINES

@PIPELINES.register_module()
class RandomRotate:
    '''
    modify by randomaffine，and normal
    '''
    def __init__(self,
                 max_rotate_degree=180.0,
                 max_translate_ratio=0.3,
                 scaling_ratio_range=(0.5, 1.5),
                 max_shear_degree=2.0,
                 border=(0, 0),
                 border_val=(114, 114, 114),
                 min_bbox_size=2,
                 min_area=20,
                 min_bboxscaling=0.2
                 # max_aspect_ratio=20
                 ):
        assert 0 <= max_translate_ratio <= 1
        assert scaling_ratio_range[0] <= scaling_ratio_range[1]
        assert scaling_ratio_range[0] > 0
        self.max_rotate_degree = max_rotate_degree
        self.max_translate_ratio = max_translate_ratio
        self.scaling_ratio_range = scaling_ratio_range
        self.max_shear_degree = max_shear_degree
        self.border = border
        self.border_val = border_val
        self.min_bbox_size = min_bbox_size
        self.min_area = min_area
        self.min_bboxscaling=min_bboxscaling
        # self.max_aspect_ratio = max_aspect_ratio

    def __call__(self, results):
        img = results['img']
        height = img.shape[0] + self.border[0] * 2
        width = img.shape[1] + self.border[1] * 2

        # Center
        center_matrix = np.eye(3, dtype=np.float32) # Identity matrix
        center_matrix[0, 2] = -img.shape[1] / 2  # x translation (pixels),center as origin
        center_matrix[1, 2] = -img.shape[0] / 2  # y translation (pixels)

        # Rotation
        rotation_degree = random.uniform(-self.max_rotate_degree,
                                         self.max_rotate_degree)
        rotation_matrix = self._get_rotation_matrix(rotation_degree)

        # Scaling
        scaling_ratio = random.uniform(self.scaling_ratio_range[0],
                                       self.scaling_ratio_range[1])
        scaling_matrix = self._get_scaling_matrix(scaling_ratio)

        # Shear
        x_degree = random.uniform(-self.max_shear_degree,
                                  self.max_shear_degree)
        y_degree = random.uniform(-self.max_shear_degree,
                                  self.max_shear_degree)
        shear_matrix = self._get_shear_matrix(x_degree, y_degree)

        # Translation
        trans_x = random.uniform(0.5 - self.max_translate_ratio,
                                 0.5 + self.max_translate_ratio) * width
        trans_y = random.uniform(0.5 - self.max_translate_ratio,
                                 0.5 + self.max_translate_ratio) * height
        translate_matrix = self._get_translation_matrix(trans_x, trans_y)

        warp_matrix = (
            translate_matrix @ shear_matrix @ rotation_matrix @ scaling_matrix
            @ center_matrix) # Matrix multiplication

        img = cv2.warpPerspective(
            img,
            warp_matrix,
            dsize=(width, height),
            borderValue=self.border_val)
        results['img'] = img
        results['img_shape'] = img.shape
        # print(results['img_info'])

        for key in results.get('bbox_fields', []):
            polygons = results[key]
            num_polygons = len(polygons)
            if num_polygons:
                # homogeneous coordinates ，advanced indexing
                xs = polygons[:, [0, 2, 4, 6]].reshape(num_polygons * 4) # [x1,x2,x3,x4,...]
                ys = polygons[:, [1, 3, 5, 7]].reshape(num_polygons * 4) # [y1,y2,y3,y4,...]
                ones = np.ones_like(xs)
                points = np.vstack([xs, ys, ones])

                warp_points = warp_matrix @ points
                warp_points = warp_points[:2] / warp_points[2]
                # xs = warp_points[0].reshape(num_polygons, 4) # [[x1,x2,x3,x4],...]
                # ys = warp_points[1].reshape(num_polygons, 4) # [[],...]
                xs = warp_points[0] # [x1,x2,x3,...]
                ys = warp_points[1] # [y1,y2,y3,...]
                warp_polygons=np.zeros(num_polygons*8)
                # print(warp_polygons.size,xs.size)
                warp_polygons[0::2]=xs
                warp_polygons[1::2]=ys
                warp_polygons=warp_polygons.reshape(num_polygons,8)
                # warp_polygons=[(x/width,y/height) for x in xs for y in ys] #and normalization
                # warp_polygons=np.float32(np.array(warp_polygons)).reshape(num_polygons, 4) # [[(x1,y1),(x2,y2),(x3,y3),(x4,y4)],...]
                # poly = np.float32(np.array(poly))
                # bboxs=[]
                # for poly in warp_polygons:
                #     rect = cv2.minAreaRect(poly)  # 得到最小外接矩形的（中心(x,y), (宽,高), 旋转角度）
                #     # box = np.float32(cv2.boxPoints(rect))  # 返回rect四个点的值
                #
                #     c_x = rect[0][0]
                #     c_y = rect[0][1]
                #     w = rect[1][0]
                #     h = rect[1][1]
                #     theta = rect[-1]  # Range for angle is [-90，0)
                #
                #     trans_data = self.cvminAreaRect2longsideformat(c_x, c_y, w, h, theta)
                #     if not trans_data:
                #         if theta != 90:  # Θ=90说明wh中有为0的元素，即gt信息不完整，无需提示异常，直接删除
                #             print('opencv表示法转长边表示法出现异常,已将第%d个box排除,问题出现在该图片中:%s' % (i, img_fullname))
                #         num_gt = num_gt - 1
                #         continue
                #     else:
                #         # range:[-180，0)
                #         c_x, c_y, longside, shortside, theta_longside = trans_data
                #
                #     if (sum(bbox <= 0) + sum(bbox[:2] >= 1)) >= 1:  # 0<xy<1, 0<side<=1
                #         print('bbox[:2]中有>= 1的元素,bbox中有<= 0的元素,已将第%d个box排除,问题出现在该图片中:%s' % (i, img_fullname))
                #         print('出问题的longside形式数据:[%.16f, %.16f, %.16f, %.16f, %.1f]' % (
                #         c_x, c_y, longside, shortside, theta_longside))
                #         num_gt = num_gt - 1
                #         continue
                #     if (obj['name'] in extractclassname):
                #         id = extractclassname.index(obj['name'])  # id=类名的索引 比如'plane'对应id=0
                #     else:
                #         print('预定类别中没有类别:%s;已将该box排除,问题出现在该图片中:%s' % (obj['name'], fullname))
                #         num_gt = num_gt - 1
                #         continue
                #     theta_label = int(theta_longside + 180.5)  # range int[0,180] 四舍五入
                #     if theta_label == 180:  # range int[0,179]
                #         theta_label = 179
                #     # outline='id x y longside shortside Θ'
                #     bbox=[c_x, c_y, longside, shortside,theta_label]
                #     # bbox = np.array((c_x, c_y, longside, shortside,theta_label))
                #     # warp_bboxs = np.vstack(
                #     #     (xs.min(1), ys.min(1), xs.max(1), ys.max(1))).T #left top right bottom
                #     bboxs.append(bbox)
                #
                # warp_bboxes=np.array(bboxs)
                # # warp_bboxes[:, [0, 2]] = warp_bboxes[:, [0, 2]].clip(0, width)
                # # warp_bboxes[:, [1, 3]] = warp_bboxes[:, [1, 3]].clip(0, height)
                # warp_bboxes[:, [4]] = warp_bboxes[:, [4]].clip(0, 179)
                # for warp_poly in warp_polygons:
                #     idx1=warp_poly[0]<width&warp_poly[1]<height
                #     idx2=warp_poly[2]<width&warp_poly[3]<height
                #     idx3=warp_poly[4]<width&warp_poly[5]<height
                #     idx4=warp_poly[6]<width&warp_poly[7]<height
                warp_polygons[:,[0,2,4,6]]=warp_polygons[:,[0,2,4,6]].clip(0,width)
                warp_polygons[:,[1,3,5,7]]=warp_polygons[:,[1,3,5,7]].clip(0,height)
                # filter polygons #
                valid_index = self.filter_gt_ploygons(polygons,warp_polygons)
                # print(valid_index)
                valid_index=np.repeat(valid_index,2,axis=1)
                results[key] = warp_polygons[valid_index]
                if key in ['gt_polygons']:
                    if 'gt_labels' in results:
                        results['gt_labels'] = results['gt_labels'][
                            valid_index]
        return results

    def filter_gt_ploygons(self, origin_polygons,wrapped_polygons):
        origin_w = np.maximum(origin_polygons[:, [0, 2, 4, 6]],1) - np.minimum(origin_polygons[:, [0, 2, 4, 6]],1)
        origin_h = np.maximum(origin_polygons[:, [1, 3, 5, 7]],1) - np.minimum(origin_polygons[:, [1, 3, 5, 7]],1)
        # origin_w = origin_bboxes[:, 2] - origin_bboxes[:, 0]
        # origin_h = origin_bboxes[:, 3] - origin_bboxes[:, 1]
        # wrapped_w = wrapped_bboxes[:, 2] - wrapped_bboxes[:, 0]
        # wrapped_h = wrapped_bboxes[:, 3] - wrapped_bboxes[:, 1]
        wrapped_w=np.maximum(wrapped_polygons[:,[0,2,4,6]],1)-np.minimum(wrapped_polygons[:,[0,2,4,6]],1)
        wrapped_h=np.maximum(wrapped_polygons[:,[1,3,5,7]],1)-np.minimum(wrapped_polygons[:,[1,3,5,7]],1)
        # aspect_ratio = np.maximum(wrapped_w / (wrapped_h + 1e-16),
        #                           wrapped_h / (wrapped_w + 1e-16))
        wh_valid_idx = (wrapped_w > self.min_bbox_size) & \
                       (wrapped_h > self.min_bbox_size)
        scal_valid_idx = wrapped_w * wrapped_h / (origin_w * origin_h +
                                                  1e-16) > self.min_bboxscaling
        area_valid_idx = wrapped_w * wrapped_h>self.min_area
        # aspect_ratio_valid_idx = aspect_ratio < self.max_aspect_ratio
        return wh_valid_idx & area_valid_idx & scal_valid_idx# & aspect_ratio_valid_idx

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(max_rotate_degree={self.max_rotate_degree}, '
        repr_str += f'max_translate_ratio={self.max_translate_ratio}, '
        repr_str += f'scaling_ratio={self.scaling_ratio_range}, '
        repr_str += f'max_shear_degree={self.max_shear_degree}, '
        repr_str += f'border={self.border}, '
        repr_str += f'border_val={self.border_val}, '
        repr_str += f'min_bbox_size={self.min_bbox_size}, '
        repr_str += f'min_area_ratio={self.min_area_ratio}, '
        repr_str += f'max_aspect_ratio={self.max_aspect_ratio})'
        return repr_str

    @staticmethod
    def _get_rotation_matrix(rotate_degrees):
        radian = math.radians(rotate_degrees)
        rotation_matrix = np.array(
            [[np.cos(radian), -np.sin(radian), 0.],
             [np.sin(radian), np.cos(radian), 0.], [0., 0., 1.]],
            dtype=np.float32)
        return rotation_matrix

    @staticmethod
    def _get_scaling_matrix(scale_ratio):
        scaling_matrix = np.array(
            [[scale_ratio, 0., 0.], [0., scale_ratio, 0.], [0., 0., 1.]],
            dtype=np.float32)
        return scaling_matrix

    @staticmethod
    def _get_share_matrix(scale_ratio):
        scaling_matrix = np.array(
            [[scale_ratio, 0., 0.], [0., scale_ratio, 0.], [0., 0., 1.]],
            dtype=np.float32)
        return scaling_matrix

    @staticmethod
    def _get_shear_matrix(x_shear_degrees, y_shear_degrees):
        x_radian = math.radians(x_shear_degrees)
        y_radian = math.radians(y_shear_degrees)
        shear_matrix = np.array([[1, np.tan(x_radian), 0.],
                                 [np.tan(y_radian), 1, 0.], [0., 0., 1.]],
                                dtype=np.float32)
        return shear_matrix

    @staticmethod
    def _get_translation_matrix(x, y):
        translation_matrix = np.array([[1, 0., x], [0., 1, y], [0., 0., 1.]],
                                      dtype=np.float32)
        return translation_matrix

    # @staticmethod
    # def cvminAreaRect2longsideformat(x_c, y_c, width, height, theta):
    #     '''
    #     trans minAreaRect(x_c, y_c, width, height, θ) to longside format(x_c, y_c, longside, shortside, θ)
    #     两者区别为:
    #             当opencv表示法中width为最长边时（包括正方形的情况），则两种表示方法一致
    #             当opencv表示法中width不为最长边 ，则最长边表示法的角度要在opencv的Θ基础上-90度
    #     @param x_c: center_x
    #     @param y_c: center_y
    #     @param width: x轴逆时针旋转碰到的第一条边
    #     @param height: 与width不同的边
    #     @param theta: x轴逆时针旋转与width的夹角，由于原点位于图像的左上角，逆时针旋转角度为负 [-90, 0)
    #     @return:
    #             x_c: center_x
    #             y_c: center_y
    #             longside: 最长边
    #             shortside: 最短边
    #             theta_longside: 最长边和x轴逆时针旋转的夹角，逆时针方向角度为负 [-180, 0)
    #     '''
    #     '''
    #     意外情况:(此时要将它们恢复符合规则的opencv形式：wh交换，Θ置为-90)
    #     竖直box：box_width < box_height  θ=0
    #     水平box：box_width > box_height  θ=0
    #     '''
    #     if theta == 0:
    #         theta = -90
    #         buffer_width = width
    #         width = height
    #         height = buffer_width
    #
    #     if theta > 0:
    #         if theta != 90:  # Θ=90说明wh中有为0的元素，即gt信息不完整，无需提示异常，直接删除
    #             print('θ计算出现异常，当前数据为：%.16f, %.16f, %.16f, %.16f, %.1f;超出opencv表示法的范围：[-90,0)' % (x_c, y_c, width, height, theta))
    #         return False
    #
    #     if theta < -90:
    #         print('θ计算出现异常，当前数据为：%.16f, %.16f, %.16f, %.16f, %.1f;超出opencv表示法的范围：[-90,0)' % (x_c, y_c, width, height, theta))
    #         return False
    #
    #     if width != max(width, height):  # 若width不是最长边
    #         longside = height
    #         shortside = width
    #         theta_longside = theta - 90
    #     else:  # 若width是最长边(包括正方形的情况)
    #         longside = width
    #         shortside = height
    #         theta_longside = theta
    #
    #     if longside < shortside:
    #         print('旋转框转换表示形式后出现问题：最长边小于短边;[%.16f, %.16f, %.16f, %.16f, %.1f]' % (x_c, y_c, longside, shortside, theta_longside))
    #         return False
    #     if (theta_longside < -180 or theta_longside >= 0):
    #         print('旋转框转换表示形式时出现问题:θ超出长边表示法的范围：[-180,0);[%.16f, %.16f, %.16f, %.16f, %.1f]' % (x_c, y_c, longside, shortside, theta_longside))
    #         return False
    #
    #     return x_c, y_c, longside, shortside, theta_longside