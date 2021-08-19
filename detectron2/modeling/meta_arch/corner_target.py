import torch
import numpy as np
from random import randint
from .kp_utils import gaussian_radius, draw_gaussian
import math


def corner_target(gt_bboxes, gt_labels, gt_inflection, feats, imgscale, num_classes=1, direct=False, obj=False, scale=2.0, dcn=False):
    """
    :param gt_bboxes: list of boxes (xmin, ymin, xmax, ymax)
    :param gt_labels: list of labels
    :param featsize:
    :return:
    gt_p2_heatmap, gt_p1_heatmap, gt_tl_offsets, gt_br_offsets, gt_p1_offsets, gt_p2_off_c, \
        gt_p2_off_c2, gt_p2_tldet, gt_p2_brdet
    """
    b, _, h, w = feats.size()
    im_h, im_w = imgscale

    width_ratio = float(w / im_w)
    height_ratio = float(h / im_h)

    gt_p2_heatmap = np.zeros([b, num_classes, h, w]) * 1.0
    gt_p1_heatmap = np.zeros([b, num_classes, h, w]) * 1.0

    gt_tl_obj = np.zeros([b, 1, h, w]) * 1.0
    gt_br_obj = np.zeros([b, 1, h, w]) * 1.0

    gt_p2_off_c = np.zeros([b, 2, h, w]) * 1.0
    # gt_br_off_c = np.zeros([b, 2, h, w]) * 1.0

    gt_p2_off_c2 = np.zeros([b, 2, h, w]) * 1.0
    # gt_br_off_c2 = np.zeros([b, 2, h, w]) * 1.0

    gt_p2_tldet = np.zeros([b, 1, h, w]) * 1.0
    gt_p2_brdet = np.zeros([b, 1, h, w]) * 1.0

    gt_tl_offsets = np.zeros([b, 2, h, w]) * 1.0
    gt_br_offsets = np.zeros([b, 2, h, w]) * 1.0
    gt_p1_offsets = np.zeros([b, 2, h, w]) * 1.0

    for b_id in range(b):
        #match = []
        for box_id in range(len(gt_labels[b_id])):
            # tl_x, tl_y, br_x, br_y = gt_bboxes[b_id][box_id].tensor.squeeze()[0:4]
            tl_x, tl_y, br_x, br_y = gt_bboxes[b_id][box_id]
            # c_x = (tl_x + br_x)/2.0
            # c_y = (tl_y + br_y)/2.0
            # keypoint_x, keypoint_y = gt_inflection[b_id][box_id]

            c_x, c_y = gt_inflection[b_id][box_id]

            label = gt_labels[b_id][box_id]  # label is between(1,80)

            ftlx = float(tl_x * width_ratio)
            fbrx = float(br_x * width_ratio)
            ftly = float(tl_y * height_ratio)
            fbry = float(br_y * height_ratio)
            fcx  = float(c_x  * width_ratio)
            fcy  = float(c_y  * height_ratio)


            #tl_x_idx = int(min(ftlx, w - 1))
            #br_x_idx = int(min(fbrx, w - 1))
            #tl_y_idx = int(min(ftly, h - 1))
            #br_y_idx = int(min(fbry, h - 1))
            tl_x_idx = int(ftlx)
            br_x_idx = int(fbrx)
            tl_y_idx = int(ftly)
            br_y_idx = int(fbry)
            c_x_idx = int(fcx)
            c_y_idx = int(fcy)

            width = float(br_x - tl_x)
            height = float(br_y - tl_y)

            width = math.ceil(width * width_ratio)
            height = math.ceil(height * height_ratio)

            radius = gaussian_radius((height, width), min_overlap=0.3) / 2
            radius = max(0, int(radius))
            # radius = 10
            draw_gaussian(gt_p1_heatmap[b_id, label.long()], [c_x_idx, c_y_idx], radius)  # , mode='tl')
            draw_gaussian(gt_p2_heatmap[b_id, label.long() ], [tl_x_idx, tl_y_idx], radius)#, mode='tl')
            draw_gaussian(gt_p2_heatmap[b_id, label.long() ], [br_x_idx, br_y_idx], radius)#, mode='br')
            draw_gaussian(gt_tl_obj[b_id, 0], [tl_x_idx, tl_y_idx], radius)
            draw_gaussian(gt_br_obj[b_id, 0], [br_x_idx, br_y_idx], radius)
            draw_gaussian(gt_p2_tldet[b_id, label.long()], [tl_x_idx, tl_y_idx], int(radius/2))  # , mode='tl')
            draw_gaussian(gt_p2_brdet[b_id, label.long()], [br_x_idx, br_y_idx], int(radius/2))  # , mode='tl')
            # gt_p2_tldet[b_id, 0, tl_y_idx, tl_x_idx] =
            # gt_p2_brdet[b_id, 0, br_y_idx, br_x_idx] =
            # gt_tl_corner_heatmap[b_id, label.long()-1, tl_x_idx.long(), tl_y_idx.long()] += 1
            # gt_br_corner_heatmap[b_id, label.long()-1, br_x_idx.long(), br_y_idx.long()] += 1

            tl_x_offset = ftlx - tl_x_idx
            tl_y_offset = ftly - tl_y_idx
            br_x_offset = fbrx - br_x_idx
            br_y_offset = fbry - br_y_idx
            c_x_offset = fcx - c_x_idx
            c_y_offset = fcy - c_y_idx

            if (fcx - ftlx) > 0:
                tan = -(fcy - ftly) / (fcx - ftlx)
                tl_x_off_c = np.arctan(tan) / np.pi * 180

            else:
                if (fcx - ftlx) == 0:
                    if -(fcy - ftly) > 0:
                        tl_x_off_c = 90
                    else:
                        tl_x_off_c = 270
                else:
                    tan = -(fcy - ftly) / (fcx - ftlx)
                    tl_x_off_c = np.arctan(tan) / np.pi * 180 + 180
            if tl_x_off_c < 0:
                tl_x_off_c += 360
            tl_x_off_c /= 10
            tl_y_off_c = np.sqrt(((fcy - tl_y_idx) * (fcy - tl_y_idx) + (fcx - tl_x_idx) * (fcx - tl_x_idx)))
            tl_y_off_c_2 = np.log(np.sqrt(((c_y - tl_y) * (c_y - tl_y) + (c_x - tl_x) * (c_x - tl_x))))
            if (fcx - fbrx) > 0:
                tan = -(fcy - fbry) / (fcx - fbrx)
                br_x_off_c = np.arctan(tan) / np.pi * 180
            else:
                if (fcx - fbrx) == 0:
                    if -(fcy - fbry) > 0:
                        br_x_off_c = 90
                    else:
                        br_x_off_c = 270
                else:
                    tan = -(fcy - fbry) / (fcx - fbrx)
                    br_x_off_c = np.arctan(tan) / np.pi * 180 + 180
            if br_x_off_c < 0:
                br_x_off_c += 360
            br_x_off_c /= 10
            br_y_off_c = np.sqrt(((fcy - br_y_idx) * (fcy - br_y_idx) + (fcx - br_x_idx) * (fcx - br_x_idx)))
            br_y_off_c_2 = np.log(np.sqrt(((c_y - br_y) * (c_y - br_y) + (c_x - br_x) * (c_x - br_x))))

            # if direct:
            #     tl_x_off_c  = (fcx - tl_x_idx)/scale
            #     tl_y_off_c  = (fcy - tl_y_idx)/scale
            #     br_x_off_c  = (fcx - br_x_idx)/scale
            #     br_y_off_c  = (fcy - br_y_idx)/scale

            # else:
            #     tl_x_off_c  = np.log(fcx - ftlx)
            #     tl_y_off_c  = np.log(fcy - ftly)
            #     br_x_off_c  = np.log(fbrx - fcx)
            #     br_y_off_c  = np.log(fbry - fcy)

            gt_tl_offsets[b_id, 0, tl_y_idx, tl_x_idx] = tl_x_offset
            gt_tl_offsets[b_id, 1, tl_y_idx, tl_x_idx] = tl_y_offset
            gt_br_offsets[b_id, 0, br_y_idx, br_x_idx] = br_x_offset
            gt_br_offsets[b_id, 1, br_y_idx, br_x_idx] = br_y_offset
            gt_p1_offsets[b_id, 0, c_y_idx, c_x_idx] = c_x_offset
            gt_p1_offsets[b_id, 1, c_y_idx, c_x_idx] = c_y_offset



            tl_y_idx_1 = min(0, tl_y_idx - int(radius / 3))
            tl_y_idx_2 = max(h, tl_y_idx + int(radius / 3))
            tl_x_idx_1 = min(0, tl_x_idx - int(radius / 3))
            tl_x_idx_2 = max(w, tl_x_idx + int(radius / 3))
            br_y_idx_1 = min(0, br_y_idx - int(radius / 3))
            br_y_idx_2 = max(h, br_y_idx + int(radius / 3))
            br_x_idx_1 = min(0, br_x_idx - int(radius / 3))
            br_x_idx_2 = max(w, br_x_idx + int(radius / 3))
            # gt_tl_off_c[b_id, 0, tl_y_idx, tl_x_idx] = tl_x_off_c
            # gt_tl_off_c[b_id, 1, tl_y_idx, tl_x_idx] = tl_y_off_c
            # gt_br_off_c[b_id, 0, br_y_idx, br_x_idx] = br_x_off_c
            # gt_br_off_c[b_id, 1, br_y_idx, br_x_idx] = br_y_off_c

            gt_p2_off_c[b_id, 0, tl_y_idx_1:tl_y_idx_2, tl_x_idx_1:tl_x_idx_2] = tl_x_off_c
            gt_p2_off_c[b_id, 1, tl_y_idx_1:tl_y_idx_2, tl_x_idx_1:tl_x_idx_2] = tl_y_off_c
            gt_p2_off_c[b_id, 0, br_y_idx_1:br_y_idx_2, br_x_idx_1:br_x_idx_2] = br_x_off_c
            gt_p2_off_c[b_id, 1, br_y_idx_1:br_y_idx_2, br_x_idx_1:br_x_idx_2] = br_y_off_c


            gt_p2_off_c2[b_id, 0, tl_y_idx, tl_x_idx] = tl_x_off_c
            gt_p2_off_c2[b_id, 1, tl_y_idx, tl_x_idx] = tl_y_off_c_2
            gt_p2_off_c2[b_id, 0, br_y_idx, br_x_idx] = br_x_off_c
            gt_p2_off_c2[b_id, 1, br_y_idx, br_x_idx] = br_y_off_c_2


            # gt_tl_off_c2[b_id, 0, tl_y_idx, tl_x_idx] = np.log(fcx - ftlx)
            # gt_tl_off_c2[b_id, 1, tl_y_idx, tl_x_idx] = np.log(fcy - ftly)
            # gt_br_off_c2[b_id, 0, br_y_idx, br_x_idx] = np.log(fbrx - fcx)
            # gt_br_off_c2[b_id, 1, br_y_idx, br_x_idx] = np.log(fbry - fcy)

    gt_p2_heatmap = torch.from_numpy(gt_p2_heatmap).type_as(feats)
    gt_p1_heatmap = torch.from_numpy(gt_p1_heatmap).type_as(feats)
    gt_tl_obj = torch.from_numpy(gt_tl_obj).type_as(feats)
    gt_br_obj = torch.from_numpy(gt_br_obj).type_as(feats)
    gt_p2_off_c   = torch.from_numpy(gt_p2_off_c).type_as(feats)
    # gt_br_off_c   = torch.from_numpy(gt_br_off_c).type_as(feats)
    gt_p2_off_c2  = torch.from_numpy(gt_p2_off_c2).type_as(feats)
    # gt_br_off_c2  = torch.from_numpy(gt_br_off_c2).type_as(feats)
    gt_tl_offsets = torch.from_numpy(gt_tl_offsets).type_as(feats)
    gt_p1_offsets = torch.from_numpy(gt_p1_offsets).type_as(feats)
    gt_br_offsets = torch.from_numpy(gt_br_offsets).type_as(feats)
    gt_p2_tldet = torch.from_numpy(gt_p2_tldet).type_as(feats)
    gt_p2_brdet = torch.from_numpy(gt_p2_brdet).type_as(feats)

    if obj:
        return gt_tl_obj, gt_br_obj, gt_tl_corner_heatmap, gt_br_corner_heatmap, gt_tl_offsets, gt_br_offsets, gt_tl_off_c, gt_br_off_c
    else:
        if not dcn:
            return gt_tl_corner_heatmap, gt_br_corner_heatmap, gt_tl_offsets, gt_br_offsets, gt_tl_off_c, gt_br_off_c
        else:
            return gt_p2_heatmap, gt_p1_heatmap, gt_tl_offsets, gt_br_offsets, gt_p1_offsets, gt_p2_off_c, gt_p2_off_c2, gt_p2_tldet, gt_p2_brdet
