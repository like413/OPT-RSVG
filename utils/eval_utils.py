import torch
import numpy as np

from utils.box_utils import bbox_iou, xywh2xyxy

import numpy as np
def trans_vg_eval_val(pred_boxes, gt_boxes):
    batch_size = pred_boxes.shape[0]
    pred_boxes = xywh2xyxy(pred_boxes)
    pred_boxes = torch.clamp(pred_boxes, 0, 1)
    gt_boxes = xywh2xyxy(gt_boxes)
    iou,inter_area,u_area = bbox_iou(pred_boxes, gt_boxes)
    accu = torch.sum(iou >= 0.5) / float(batch_size)

    return iou, accu

def trans_vg_eval_test(pred_boxes, gt_boxes):
    bath_size = pred_boxes.shape[0]
    pred_boxes = xywh2xyxy(pred_boxes)
    pred_boxes = torch.clamp(pred_boxes, 0, 1)
    gt_boxes = xywh2xyxy(gt_boxes)
    iou,inter_area,u_area = bbox_iou(pred_boxes, gt_boxes)
    accu_num = torch.sum(iou >= 0.5)
    accu_num_6 = torch.sum(iou >= 0.6)
    accu_num_7 = torch.sum(iou >= 0.7)
    accu_num_8 = torch.sum(iou >= 0.8)
    accu_num_9 = torch.sum(iou >= 0.9)
    cumInterArea = np.sum(np.array(inter_area.data.cpu().numpy()))
    cumUnionArea = np.sum(np.array(u_area.data.cpu().numpy()))
    meaniou = torch.mean(iou).item()
    return accu_num,accu_num_6,accu_num_7,accu_num_8,accu_num_9,meaniou,cumInterArea/cumUnionArea
if __name__ == '__main__':

    b = torch.randn((16,20))

    d = torch.randn((16, 20))
    accu_cls = torch.sum(np.argmax(b)==d)/float(16)
    print(accu_cls)
