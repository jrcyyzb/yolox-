# Copyright (c) OpenMMLab. All rights reserved.
import math
import warnings

import mmcv
import numpy as np
import torch
import torch.nn as nn

# from mmdet.core import bbox_overlaps
from mmdet.core import rotatebbox_overlaps
from ..builder import LOSSES
from .utils import weighted_loss
import cv2


@mmcv.jit(derivate=True, coderize=True)
@weighted_loss
def rotate_l1_loss(bboxes1, bboxes2):
    """L1 loss.

    Args:
        pred (torch.Tensor): The prediction.
        target (torch.Tensor): The learning target of the prediction.

    Returns:
        torch.Tensor: Calculated loss
    """
    batch_shape = bboxes1.shape[:-2]

    rows = bboxes1.size(-2)
    cols = bboxes2.size(-2)
    coord1 = bboxes1[..., [0, 1]]
    coord2 = bboxes1[..., [2, 1]]
    coord3 = bboxes1[..., [2, 3]]
    coord4 = bboxes1[..., [0, 3]]
    ons = torch.ones(rows).unsqueeze(1).to(coord1.device)
    # print(coord1.device,ons.device)
    coord1 = torch.cat((coord1, ons), 1)
    coord2 = torch.cat((coord2, ons), 1)
    coord3 = torch.cat((coord3, ons), 1)
    coord4 = torch.cat((coord4, ons), 1)
    theta = bboxes1[..., 4]
    theta = 3.1415926535898 * theta / 180
    center = (coord1 + coord3) / 2
    coord = torch.cat((coord1, coord2, coord3, coord4), 1).reshape(-1, 4, 3).permute(0, 2,
                                                                                     1)  # [bn,3,4],[[x1,y1,1],[x2,y2,1],[x3,y3,1],[x4,y4,1]]->[[x1,x2,x3,x4],[y1,y2,y3,y4],[1,1,1,1]]
    print('coord', coord.size())
    rotate_matrix = torch.FloatTensor([[1, 1, 1], [1, 1, 1], [0, 0, 1]])
    rotate_matrix = rotate_matrix.repeat(rows, 1, 1)  # [bn,3,3]
    cos_theta = torch.cos(theta)
    sin_theta = torch.sin(theta)
    x_c = center[..., 0]
    y_c = center[..., 1]
    # print('rotate',rotate_matrix.size(),rotate_matrix.type(),theta.size(),coord.size(),coord.type())
    rotate_matrix[..., 0, 0] = cos_theta
    rotate_matrix[..., 1, 1] = cos_theta
    rotate_matrix[..., 0, 1] = -sin_theta
    rotate_matrix[..., 1, 0] = sin_theta
    rotate_matrix[..., 0, 2] = x_c - x_c * cos_theta + y_c * sin_theta
    rotate_matrix[..., 1, 2] = y_c - x_c * sin_theta - y_c * cos_theta
    rotate_matrix = rotate_matrix.to(coord.device)
    rotate_coord = torch.bmm(rotate_matrix, coord)  # [bn,3,4]
    # print('rotate_coord', rotate_coord.size(), 'bboxes2', bboxes2.size())
    rotate_coord = rotate_coord[..., [0, 1], :].permute(0, 2, 1).reshape(-1, 8)
    # if target.numel() == 0:
    #     return pred.sum() * 0


    # assert pred.size() == target.size()
    loss = torch.min(
            torch.cat((torch.abs(pred[...,[2,3,4,5,6,7,0,1]] - target).unsqueeze(1),torch.abs(pred[...,[4,5,6,7,0,1,2,3]] - target).unsqueeze(1),
                        torch.abs(pred[...,[6,7,0,1,2,3,4,5]] - target).unsqueeze(1),torch.abs(pred - target).unsqueeze(1)),1),1)
    return loss


@LOSSES.register_module()
class rotate_L1Loss(nn.Module):
    """L1 loss.

    Args:
        reduction (str, optional): The method to reduce the loss.
            Options are "none", "mean" and "sum".
        loss_weight (float, optional): The weight of loss.
    """

    def __init__(self, reduction='mean', loss_weight=1.0):
        super(rotate_L1Loss, self).__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None):
        """Forward function.

        Args:
            pred (torch.Tensor): The prediction.
            target (torch.Tensor): The learning target of the prediction.
            weight (torch.Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Defaults to None.
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        loss_bbox = self.loss_weight * rotate_l1_loss(
            pred, target, weight, reduction=reduction, avg_factor=avg_factor)
        return loss_bbox

@mmcv.jit(derivate=True, coderize=True)
@weighted_loss
def iou_loss(pred, target, linear=False, mode='log', eps=1e-6):
    """IoU loss.

    Computing the IoU loss between a set of predicted bboxes and target bboxes.
    The loss is calculated as negative log of IoU.

    Args:
        pred (torch.Tensor): Predicted bboxes of format (x1, y1, x2, y2),
            shape (n, 4).
        target (torch.Tensor): Corresponding gt bboxes, shape (n, 4).
        linear (bool, optional): If True, use linear scale of loss instead of
            log scale. Default: False.
        mode (str): Loss scaling mode, including "linear", "square", and "log".
            Default: 'log'
        eps (float): Eps to avoid log(0).

    Return:
        torch.Tensor: Loss tensor.
    """
    assert mode in ['linear', 'square', 'log']
    if linear:
        mode = 'linear'
        warnings.warn('DeprecationWarning: Setting "linear=True" in '
                      'iou_loss is deprecated, please use "mode=`linear`" '
                      'instead.')
    ious = rotatebbox_overlaps(pred, target, is_aligned=True).clamp(min=eps)
    if mode == 'linear':
        loss = 1 - ious
    elif mode == 'square':
        loss = 1 - ious**2
    elif mode == 'log':
        loss = -ious.log()
    else:
        raise NotImplementedError
    return loss


# @mmcv.jit(derivate=True, coderize=True)
# @weighted_loss
# def bounded_iou_loss(pred, target, beta=0.2, eps=1e-3):
#     """BIoULoss.
#
#     This is an implementation of paper
#     `Improving Object Localization with Fitness NMS and Bounded IoU Loss.
#     <https://arxiv.org/abs/1711.00164>`_.
#
#     Args:
#         pred (torch.Tensor): Predicted bboxes.
#         target (torch.Tensor): Target bboxes.
#         beta (float): beta parameter in smoothl1.
#         eps (float): eps to avoid NaN.
#     """
#     pred_ctrx = (pred[:, 0] + pred[:, 2]) * 0.5
#     pred_ctry = (pred[:, 1] + pred[:, 3]) * 0.5
#     pred_w = pred[:, 2] - pred[:, 0]
#     pred_h = pred[:, 3] - pred[:, 1]
#     with torch.no_grad():
#         target_ctrx = (target[:, 0] + target[:, 2]) * 0.5
#         target_ctry = (target[:, 1] + target[:, 3]) * 0.5
#         target_w = target[:, 2] - target[:, 0]
#         target_h = target[:, 3] - target[:, 1]
#
#     dx = target_ctrx - pred_ctrx
#     dy = target_ctry - pred_ctry
#
#     loss_dx = 1 - torch.max(
#         (target_w - 2 * dx.abs()) /
#         (target_w + 2 * dx.abs() + eps), torch.zeros_like(dx))
#     loss_dy = 1 - torch.max(
#         (target_h - 2 * dy.abs()) /
#         (target_h + 2 * dy.abs() + eps), torch.zeros_like(dy))
#     loss_dw = 1 - torch.min(target_w / (pred_w + eps), pred_w /
#                             (target_w + eps))
#     loss_dh = 1 - torch.min(target_h / (pred_h + eps), pred_h /
#                             (target_h + eps))
#     # view(..., -1) does not work for empty tensor
#     loss_comb = torch.stack([loss_dx, loss_dy, loss_dw, loss_dh],
#                             dim=-1).flatten(1)
#
#     loss = torch.where(loss_comb < beta, 0.5 * loss_comb * loss_comb / beta,
#                        loss_comb - 0.5 * beta)
#     return loss
#
#
# @mmcv.jit(derivate=True, coderize=True)
# @weighted_loss
# def giou_loss(pred, target, eps=1e-7):
#     r"""`Generalized Intersection over Union: A Metric and A Loss for Bounding
#     Box Regression <https://arxiv.org/abs/1902.09630>`_.
#
#     Args:
#         pred (torch.Tensor): Predicted bboxes of format (x1, y1, x2, y2),
#             shape (n, 4).
#         target (torch.Tensor): Corresponding gt bboxes, shape (n, 4).
#         eps (float): Eps to avoid log(0).
#
#     Return:
#         Tensor: Loss tensor.
#     """
#     gious = bbox_overlaps(pred, target, mode='giou', is_aligned=True, eps=eps)
#     loss = 1 - gious
#     return loss
#
#
# @mmcv.jit(derivate=True, coderize=True)
# @weighted_loss
# def diou_loss(pred, target, eps=1e-7):
#     r"""`Implementation of Distance-IoU Loss: Faster and Better
#     Learning for Bounding Box Regression, https://arxiv.org/abs/1911.08287`_.
#
#     Code is modified from https://github.com/Zzh-tju/DIoU.
#
#     Args:
#         pred (Tensor): Predicted bboxes of format (x1, y1, x2, y2),
#             shape (n, 4).
#         target (Tensor): Corresponding gt bboxes, shape (n, 4).
#         eps (float): Eps to avoid log(0).
#     Return:
#         Tensor: Loss tensor.
#     """
#     # overlap
#     lt = torch.max(pred[:, :2], target[:, :2])
#     rb = torch.min(pred[:, 2:], target[:, 2:])
#     wh = (rb - lt).clamp(min=0)
#     overlap = wh[:, 0] * wh[:, 1]
#
#     # union
#     ap = (pred[:, 2] - pred[:, 0]) * (pred[:, 3] - pred[:, 1])
#     ag = (target[:, 2] - target[:, 0]) * (target[:, 3] - target[:, 1])
#     union = ap + ag - overlap + eps
#
#     # IoU
#     ious = overlap / union
#
#     # enclose area
#     enclose_x1y1 = torch.min(pred[:, :2], target[:, :2])
#     enclose_x2y2 = torch.max(pred[:, 2:], target[:, 2:])
#     enclose_wh = (enclose_x2y2 - enclose_x1y1).clamp(min=0)
#
#     cw = enclose_wh[:, 0]
#     ch = enclose_wh[:, 1]
#
#     c2 = cw**2 + ch**2 + eps
#
#     b1_x1, b1_y1 = pred[:, 0], pred[:, 1]
#     b1_x2, b1_y2 = pred[:, 2], pred[:, 3]
#     b2_x1, b2_y1 = target[:, 0], target[:, 1]
#     b2_x2, b2_y2 = target[:, 2], target[:, 3]
#
#     left = ((b2_x1 + b2_x2) - (b1_x1 + b1_x2))**2 / 4
#     right = ((b2_y1 + b2_y2) - (b1_y1 + b1_y2))**2 / 4
#     rho2 = left + right
#
#     # DIoU
#     dious = ious - rho2 / c2
#     loss = 1 - dious
#     return loss
#
#
# @mmcv.jit(derivate=True, coderize=True)
# @weighted_loss
# def ciou_loss(pred, target, eps=1e-7):
#     r"""`Implementation of paper `Enhancing Geometric Factors into
#     Model Learning and Inference for Object Detection and Instance
#     Segmentation <https://arxiv.org/abs/2005.03572>`_.
#
#     Code is modified from https://github.com/Zzh-tju/CIoU.
#
#     Args:
#         pred (Tensor): Predicted bboxes of format (x1, y1, x2, y2),
#             shape (n, 4).
#         target (Tensor): Corresponding gt bboxes, shape (n, 4).
#         eps (float): Eps to avoid log(0).
#     Return:
#         Tensor: Loss tensor.
#     """
#     # overlap
#     lt = torch.max(pred[:, :2], target[:, :2])
#     rb = torch.min(pred[:, 2:], target[:, 2:])
#     wh = (rb - lt).clamp(min=0)
#     overlap = wh[:, 0] * wh[:, 1]
#
#     # union
#     ap = (pred[:, 2] - pred[:, 0]) * (pred[:, 3] - pred[:, 1])
#     ag = (target[:, 2] - target[:, 0]) * (target[:, 3] - target[:, 1])
#     union = ap + ag - overlap + eps
#
#     # IoU
#     ious = overlap / union
#
#     # enclose area
#     enclose_x1y1 = torch.min(pred[:, :2], target[:, :2])
#     enclose_x2y2 = torch.max(pred[:, 2:], target[:, 2:])
#     enclose_wh = (enclose_x2y2 - enclose_x1y1).clamp(min=0)
#
#     cw = enclose_wh[:, 0]
#     ch = enclose_wh[:, 1]
#
#     c2 = cw**2 + ch**2 + eps
#
#     b1_x1, b1_y1 = pred[:, 0], pred[:, 1]
#     b1_x2, b1_y2 = pred[:, 2], pred[:, 3]
#     b2_x1, b2_y1 = target[:, 0], target[:, 1]
#     b2_x2, b2_y2 = target[:, 2], target[:, 3]
#
#     w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
#     w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps
#
#     left = ((b2_x1 + b2_x2) - (b1_x1 + b1_x2))**2 / 4
#     right = ((b2_y1 + b2_y2) - (b1_y1 + b1_y2))**2 / 4
#     rho2 = left + right
#
#     factor = 4 / math.pi**2
#     v = factor * torch.pow(torch.atan(w2 / h2) - torch.atan(w1 / h1), 2)
#
#     with torch.no_grad():
#         alpha = (ious > 0.5).float() * v / (1 - ious + v)
#
#     # CIoU
#     cious = ious - (rho2 / c2 + alpha * v)
#     loss = 1 - cious.clamp(min=-1.0, max=1.0)
#     return loss


@LOSSES.register_module()
class roLoss(nn.Module):
    """IoULoss.

    Computing the IoU loss between a set of predicted bboxes and target bboxes.

    Args:
        linear (bool): If True, use linear scale of loss else determined
            by mode. Default: False.
        eps (float): Eps to avoid log(0).
        reduction (str): Options are "none", "mean" and "sum".
        loss_weight (float): Weight of loss.
        mode (str): Loss scaling mode, including "linear", "square", and "log".
            Default: 'log'
    """

    def __init__(self,
                 linear=False,
                 eps=1e-6,
                 reduction='mean',
                 loss_weight=1.0,
                 mode='log'):
        super(roLoss, self).__init__()
        assert mode in ['linear', 'square', 'log']
        if linear:
            mode = 'linear'
            warnings.warn('DeprecationWarning: Setting "linear=True" in '
                          'IOULoss is deprecated, please use "mode=`linear`" '
                          'instead.')
        self.mode = mode
        self.linear = linear
        self.eps = eps
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        """Forward function.

        Args:
            pred (torch.Tensor): The prediction.
            target (torch.Tensor): The learning target of the prediction.
            weight (torch.Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Defaults to None. Options are "none", "mean" and "sum".
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        if (weight is not None) and (not torch.any(weight > 0)) and (
                reduction != 'none'):
            if pred.dim() == weight.dim() + 1:
                weight = weight.unsqueeze(1)
            return (pred * weight).sum()  # 0
        if weight is not None and weight.dim() > 1:
            # TODO: remove this in the future
            # reduce the weight of shape (n, 4) to (n,) to match the
            # iou_loss of shape (n,)
            assert weight.shape == pred.shape
            weight = weight.mean(-1)
        loss = self.loss_weight * iou_loss(
            pred,
            target,
            weight,
            mode=self.mode,
            eps=self.eps,
            reduction=reduction,
            avg_factor=avg_factor,
            **kwargs)
        return loss


@LOSSES.register_module()
class FGLoss(nn.Module):
    """FGLoss.

    Computing the Fourlier loss between a set of predicted robboxes and target robboxes.

    Args:
        linear (bool): If True, use linear scale of loss else determined
            by mode. Default: False.
        eps (float): Eps to avoid log(0).
        reduction (str): Options are "none", "mean" and "sum".
        loss_weight (float): Weight of loss.
        mode (str): Loss scaling mode, including "linear", "square", and "log".
            Default: 'log'
    """

    def __init__(self,
                 linear=False,
                 eps=1e-6,
                 reduction='mean',
                 loss_weight=1.0,
                 mode='log'):
        super(FGLoss, self).__init__()
        assert mode in ['linear', 'square', 'log']
        if linear:
            mode = 'linear'
            warnings.warn('DeprecationWarning: Setting "linear=True" in '
                          'IOULoss is deprecated, please use "mode=`linear`" '
                          'instead.')
        self.mode = mode
        self.linear = linear
        self.eps = eps
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        """Forward function.

        Args:
            pred (torch.Tensor): The prediction.
            target (torch.Tensor): The learning target of the prediction.
            weight (torch.Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Defaults to None. Options are "none", "mean" and "sum".
        """
        # assert reduction_override in (None, 'none', 'mean', 'sum')
        # reduction = (
        #     reduction_override if reduction_override else self.reduction)
        # if (weight is not None) and (not torch.any(weight > 0)) and (
        #         reduction != 'none'):
        #     if pred.dim() == weight.dim() + 1:
        #         weight = weight.unsqueeze(1)
        #     return (pred * weight).sum()  # 0
        # if weight is not None and weight.dim() > 1:
        #     # TODO: remove this in the future
        #     # reduce the weight of shape (n, 4) to (n,) to match the
        #     # iou_loss of shape (n,)
        #     assert weight.shape == pred.shape
        #     weight = weight.mean(-1)
        loss = self.loss_weight * fg_loss(
            pred,
            target,
            weight,
            mode=self.mode,
            eps=self.eps,
            # reduction=reduction,
            avg_factor=avg_factor,
            **kwargs)
        return loss

@mmcv.jit(derivate=True, coderize=True)
@weighted_loss
def fg_loss(pred, target, linear=False, mode='log', eps=1e-6):
    pi=3.1415926535898
    pred_center=(pred[...,[0,1]]+pred[...,[2,3]])/2
    # print(pred_center.size())
    pred_w=pred[...,[2]]-pred[...,[0]]
    pred_h=pred[...,[3]]-pred[...,[1]]
    pred_theta=pi*pred[...,[4]]/180
    targets=target.reshape(-1,4,2).numpy()
    rects=[]
    for polygon in targets:
        rect = cv2.minAreaRect(polygon) #((cx,cy),(w,h),angle)
        rect=[rect[0][0],rect[0][1],rect[1][0],rect[1][1],rect[2]]
        rects.append(rect)
        # rects=np.concatenate((rects,rect),0)
    print(rects)
    rects=torch.tensor(rects)
    # print(rects)
    # np.array()
    target_center=rects[...,[0,1]]
    # print(target_center)

    target_w=rects[...,[2]]
    target_h=rects[...,[3]]
    target_theta=pi*rects[...,[4]]/180
    # x_c - x_c * cos_theta + y_c * sin_theta
    # y_c - x_c * sin_theta - y_c * cos_theta
    pred_cos=torch.cos(pred_theta)
    pred_sin = torch.sin(pred_theta)
    target_cos = torch.cos(target_theta)
    target_sin = torch.sin(target_theta)
    pred_cos=pred_cos.unsqueeze(2)
    pred_sin=pred_sin.unsqueeze(2)
    target_cos=target_cos.unsqueeze(2)
    target_sin=target_sin.unsqueeze(2)
    # u = torch.arange(-32, 32, 1)
    # v = torch.arange(-32, 32, 1)
    u=torch.linspace(-0.5, 0.5, 32)
    v = torch.linspace(-0.5, 0.5, 32)
    # u = torch.arange(-64, 64, 1)
    # v = torch.arange(-64, 64, 1)
    # u = torch.arange(-128, 128, 1)
    # v = torch.arange(-128, 128, 1)
    # u = torch.arange(-256, 256, 1)
    # v = torch.arange(-256, 256, 1)
    u=u.unsqueeze(0)
    v=v.unsqueeze(1)
    # print(pred_cos,'\n',target_cos)
    ro_u=pred_cos*u+pred_sin*v
    ro_v=-pred_sin*u+pred_cos*v
    # print(ro_v)
    target_u=target_cos*u+target_sin*v
    target_v=-target_sin*u+target_cos*v
    # print(target_u)
    pred_w=pred_w.unsqueeze(2)
    pred_h=pred_h.unsqueeze(2)
    target_w=target_w.unsqueeze(2)
    target_h=target_h.unsqueeze(2)
    a=torch.sin(pi * ro_u * pred_w) / (pi * ro_u * pred_w+eps)
    b=torch.sin(pi * ro_v * pred_h) / (pi * ro_v * pred_h+eps)
    c=torch.sin(pi * target_u * target_w) / (pi * target_u * target_w+eps)
    d=torch.sin(pi * target_v * target_h) / (pi * target_v * target_h+eps)
    # e = torch.sin(pi * u * target_w) / (pi * u * target_w+eps)
    # f = torch.sin(pi * v * target_h) / (pi * v * target_h+eps)
    # a=a.unsqueeze(2)
    # b=b.unsqueeze(1)
    # c=c.unsqueeze(2)
    # d=d.unsqueeze(1)
    # e=e.unsqueeze(2)
    # f=f.unsqueeze(1)
    pred_fmap =a*b
    target_fmap =c*d
    # pred_fmap=torch.bmm(a, b)
    # target_fmap=torch.bmm(c,d)
    # test_fmap=torch.bmm(e,f)
    # 方法2：plt.imshow(ndarray)
    # img = image[0]  # plt.imshow()只能接受3-D Tensor，所以也要用image[0]消去batch那一维
    img2 = target_fmap[0].unsqueeze(0).numpy()  # FloatTensor转为ndarray
    img2 = np.transpose(img2, (1, 2, 0))  # 把channel那一维放到最后
    img1 = pred_fmap[0].unsqueeze(0).numpy()  # FloatTensor转为ndarray
    img1 = np.transpose(img1, (1, 2, 0))  # 把channel那一维放到最后
    # img3 = test_fmap.numpy()  # FloatTensor转为ndarray
    # img3 = np.transpose(img3, (1, 2, 0))  # 把channel那一维放到最后
    # 显示图片
    # plt.imshow(img3)  #test
    # plt.show()
    plt.subplot(1, 2, 1)
    plt.imshow(img1)  #pred
    # plt.show()
    plt.subplot(1, 2, 2)
    plt.imshow(img2)  #target
    plt.show()
    # print(torch.sum(torch.sum(torch.sum(torch.abs(pred_fmap), 0), 0),0))
    # print(torch.sum(torch.sum(torch.sum(torch.abs(target_fmap), 0), 0), 0))
    lossmap= pred_fmap- \
             target_fmap
    # print(torch.isnan(temp_loss))
    loss1=torch.sum(torch.sum(torch.sum(torch.abs(lossmap), 0), 0),0)
    # print(loss1)
    loss2=torch.sum(torch.sum(torch.abs(pred_center-target_center),0),0)
    # print(loss2)
    loss=loss1+loss2

    return loss

# @LOSSES.register_module()
# class BoundedIoULoss(nn.Module):
#
#     def __init__(self, beta=0.2, eps=1e-3, reduction='mean', loss_weight=1.0):
#         super(BoundedIoULoss, self).__init__()
#         self.beta = beta
#         self.eps = eps
#         self.reduction = reduction
#         self.loss_weight = loss_weight
#
#     def forward(self,
#                 pred,
#                 target,
#                 weight=None,
#                 avg_factor=None,
#                 reduction_override=None,
#                 **kwargs):
#         if weight is not None and not torch.any(weight > 0):
#             if pred.dim() == weight.dim() + 1:
#                 weight = weight.unsqueeze(1)
#             return (pred * weight).sum()  # 0
#         assert reduction_override in (None, 'none', 'mean', 'sum')
#         reduction = (
#             reduction_override if reduction_override else self.reduction)
#         loss = self.loss_weight * bounded_iou_loss(
#             pred,
#             target,
#             weight,
#             beta=self.beta,
#             eps=self.eps,
#             reduction=reduction,
#             avg_factor=avg_factor,
#             **kwargs)
#         return loss


# @LOSSES.register_module()
# class GIoULoss(nn.Module):
#
#     def __init__(self, eps=1e-6, reduction='mean', loss_weight=1.0):
#         super(GIoULoss, self).__init__()
#         self.eps = eps
#         self.reduction = reduction
#         self.loss_weight = loss_weight
#
#     def forward(self,
#                 pred,
#                 target,
#                 weight=None,
#                 avg_factor=None,
#                 reduction_override=None,
#                 **kwargs):
#         if weight is not None and not torch.any(weight > 0):
#             if pred.dim() == weight.dim() + 1:
#                 weight = weight.unsqueeze(1)
#             return (pred * weight).sum()  # 0
#         assert reduction_override in (None, 'none', 'mean', 'sum')
#         reduction = (
#             reduction_override if reduction_override else self.reduction)
#         if weight is not None and weight.dim() > 1:
#             # TODO: remove this in the future
#             # reduce the weight of shape (n, 4) to (n,) to match the
#             # giou_loss of shape (n,)
#             assert weight.shape == pred.shape
#             weight = weight.mean(-1)
#         loss = self.loss_weight * giou_loss(
#             pred,
#             target,
#             weight,
#             eps=self.eps,
#             reduction=reduction,
#             avg_factor=avg_factor,
#             **kwargs)
#         return loss
#
#
# @LOSSES.register_module()
# class DIoULoss(nn.Module):
#
#     def __init__(self, eps=1e-6, reduction='mean', loss_weight=1.0):
#         super(DIoULoss, self).__init__()
#         self.eps = eps
#         self.reduction = reduction
#         self.loss_weight = loss_weight
#
#     def forward(self,
#                 pred,
#                 target,
#                 weight=None,
#                 avg_factor=None,
#                 reduction_override=None,
#                 **kwargs):
#         if weight is not None and not torch.any(weight > 0):
#             if pred.dim() == weight.dim() + 1:
#                 weight = weight.unsqueeze(1)
#             return (pred * weight).sum()  # 0
#         assert reduction_override in (None, 'none', 'mean', 'sum')
#         reduction = (
#             reduction_override if reduction_override else self.reduction)
#         if weight is not None and weight.dim() > 1:
#             # TODO: remove this in the future
#             # reduce the weight of shape (n, 4) to (n,) to match the
#             # giou_loss of shape (n,)
#             assert weight.shape == pred.shape
#             weight = weight.mean(-1)
#         loss = self.loss_weight * diou_loss(
#             pred,
#             target,
#             weight,
#             eps=self.eps,
#             reduction=reduction,
#             avg_factor=avg_factor,
#             **kwargs)
#         return loss
#
#
# @LOSSES.register_module()
# class CIoULoss(nn.Module):
#
#     def __init__(self, eps=1e-6, reduction='mean', loss_weight=1.0):
#         super(CIoULoss, self).__init__()
#         self.eps = eps
#         self.reduction = reduction
#         self.loss_weight = loss_weight
#
#     def forward(self,
#                 pred,
#                 target,
#                 weight=None,
#                 avg_factor=None,
#                 reduction_override=None,
#                 **kwargs):
#         if weight is not None and not torch.any(weight > 0):
#             if pred.dim() == weight.dim() + 1:
#                 weight = weight.unsqueeze(1)
#             return (pred * weight).sum()  # 0
#         assert reduction_override in (None, 'none', 'mean', 'sum')
#         reduction = (
#             reduction_override if reduction_override else self.reduction)
#         if weight is not None and weight.dim() > 1:
#             # TODO: remove this in the future
#             # reduce the weight of shape (n, 4) to (n,) to match the
#             # giou_loss of shape (n,)
#             assert weight.shape == pred.shape
#             weight = weight.mean(-1)
#         loss = self.loss_weight * ciou_loss(
#             pred,
#             target,
#             weight,
#             eps=self.eps,
#             reduction=reduction,
#             avg_factor=avg_factor,
#             **kwargs)
#         return loss
