import torch
import torch.nn as nn
import numpy as np
import pdb
import cv2
from mmcv.runner import get_dist_info 
import mmcv
import math
from bBox_2D import bBox_2D
def _gather_feat(feat, ind, mask=None): 
    dim = feat.size(2)
    ind = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
    feat = feat.gather(1, ind)
    if mask is not None:
        mask = mask.unsqueeze(2).expand_as(feat)
        feat = feat[mask]
        feat = feat.view(-1, dim)
    return feat


def _nms(heat, kernel=1):  # kernel size is 3 in the paper
    pad = (kernel - 1) // 2

    hmax = nn.functional.max_pool2d(heat, (kernel, kernel), stride=1, padding=pad)
    keep = (hmax == heat).float()
    return heat * keep


def _tranpose_and_gather_feat(feat, ind):
    feat = feat.permute(0, 2, 3, 1).contiguous()
    feat = feat.view(feat.size(0), -1, feat.size(3))  # x,y
    feat = _gather_feat(feat, ind)
    return feat


def _topk(scores, K=20):
    batch, cat, height, width = scores.size()  # cat is the num of categories

    topk_scores, topk_inds = torch.topk(scores.view(batch, -1), K)

    topk_clses = (topk_inds / (height * width)).int()

    topk_inds = topk_inds % (height * width)
    topk_ys = (topk_inds / width).int().float()
    topk_xs = (topk_inds % width).int().float()
    return topk_scores, topk_inds, topk_clses, topk_ys, topk_xs

def _decode_box(
        thersold, pred_p2_heatmap, pred_p2_off_c, pred_p2_off_c2, pred_p2_tldet, pred_p2_brdet, pred_p2_tloffsets,
        pred_p2_broffsets, pred_p1_heatmap, pred_p1_offsets, img_meta,
        scale_factor=None, rescale=False, obj=False, direct=True,
        linear_factor=4, K=20, kernel=3, ae_threshold=0.05, num_dets=1000
):
    K = 50
    Z = 10
    KK = 10

    batch, cat, height, width = pred_p2_heatmap.size()
    _, inp_h, inp_w = img_meta


    p2_heat = _nms(pred_p2_heatmap, kernel=kernel)
    p1_heat = _nms(pred_p1_heatmap, kernel=kernel)
    pred_p2_off_c2[:, 0, :, :] *= 10
    pred_p2_off_c2[:, 1, :, :] *= 1
    pred_p2_off_c2[:, 1, :, :] = torch.clamp(pred_p2_off_c2[:, 1, :, :], min=0.0)
    pred_p2_off_c22 = torch.zeros(pred_p2_off_c2.size())
    pred_p2_off_c22[:, 0, :, :] = torch.cos(pred_p2_off_c2[:, 0, :, :] / 180 * np.pi) * pred_p2_off_c2[:, 1, :, :]
    pred_p2_off_c22[:, 1, :, :] = -torch.sin(pred_p2_off_c2[:, 0, :, :] / 180 * np.pi) * pred_p2_off_c2[:, 1, :, :]
    pred_p2_off_c2 = pred_p2_off_c22.to("cuda")
    p2_scores, p2_inds, p2_clses, p2_ys, p2_xs = _topk(p2_heat, K=K)
    tl_det = _tranpose_and_gather_feat(pred_p2_tldet, p2_inds)
    br_det = _tranpose_and_gather_feat(pred_p2_brdet, p2_inds)
    tl_inds = p2_inds[(tl_det >= br_det).view(1, K)]
    tl_scores = p2_scores[(tl_det >= br_det).view(1, K)]
    br_inds = p2_inds[(tl_det < br_det).view(1, K)]
    a = 0
    if tl_inds.view(-1).shape[0] > br_inds.view(-1).shape[0]:
        a = br_inds.view(-1).shape[0]
    else:
        a = tl_inds.view(-1).shape[0]
    Z = a
    KK = a
    if a == 0:
        return [], [], []
    else:
        br_scores = p2_scores[(tl_det >= br_det).view(1, K)]

        tl_scores = tl_scores[:a]
        br_scores = br_scores[:a]

        p1_scores, p1_inds, p1_clses, p1_ys, p1_xs = _topk(p1_heat, K=Z)

        p2_off_c = _tranpose_and_gather_feat(pred_p2_off_c2, p2_inds)
        tl_off_c = (p2_off_c.view(K, 2)[(tl_det >= br_det).view(-1)])
        br_off_c = (p2_off_c.view(K, 2)[(tl_det < br_det).view(-1)])

        tl_off_c = tl_off_c[:a]
        br_off_c = br_off_c[:a]
        tcl_off_c = tl_off_c[:a]
        bcr_off_c = br_off_c[:a]

        tl_regr = _tranpose_and_gather_feat(pred_p2_tloffsets, tl_inds.view(1, -1))
        br_regr = _tranpose_and_gather_feat(pred_p2_broffsets, br_inds.view(1, -1))
        p1_regr = _tranpose_and_gather_feat(pred_p1_offsets, p1_inds)
        tl_regr = tl_regr[:, :a]
        br_regr = br_regr[:, :a]
        tcl_regr = tl_regr[:, :a]
        bcr_regr = br_regr[:, :a]


        tl_ys = p2_ys.view(-1)[(tl_det >= br_det).view(-1)]
        tl_xs = p2_xs.view(-1)[(tl_det >= br_det).view(-1)]
        br_ys = p2_ys.view(-1)[(tl_det < br_det).view(-1)]
        br_xs = p2_xs.view(-1)[(tl_det < br_det).view(-1)]
        tl_ys = tl_ys[:a]
        tl_xs = tl_xs[:a]
        br_ys = br_ys[:a]
        br_xs = br_xs[:a]
        tcl_ys = tl_ys[:a]
        tcl_xs = tl_xs[:a]
        bcr_ys = br_ys[:a]
        bcr_xs = br_xs[:a]

        x = tl_xs.shape[0]
        y = br_xs.shape[0]
        tl_ys1 = tl_ys.view(batch, x, 1)
        tl_xs1 = tl_xs.view(batch, x, 1)
        br_ys1 = br_ys.view(batch, 1, y)
        br_xs1 = br_xs.view(batch, 1, y)
        tl_ys = tl_ys1.expand(batch, x, x)  # expand for combine all possible boxes
        tl_xs = tl_xs1.expand(batch, x, x)
        br_ys = br_ys1.expand(batch, y, y)
        br_xs = br_xs1.expand(batch, y, y)
        pc1_ys1 = p1_ys.view(batch, 1, Z)

        pc1_xs1 = p1_xs.view(batch, 1, Z)
        pc1_ys = pc1_ys1.expand(batch, x, Z)
        pc1_xs = pc1_xs1.expand(batch, x, Z)
        p1_ys1 = p1_ys.view(batch, 1, Z)
        p1_xs1 = p1_xs.view(batch, 1, Z)
        p1_ys = p1_ys1.expand(batch, Z, Z)
        p1_xs = p1_xs1.expand(batch, Z, Z)

        tcl_ys1 = tcl_ys.view(batch, x, 1)
        tcl_xs1 = tcl_xs.view(batch, x, 1)
        bcr_ys1 = bcr_ys.view(batch, y, 1)
        bcr_xs1 = bcr_xs.view(batch, y, 1)
        tcl_ys = tcl_ys1.expand(batch, x, Z)
        tcl_xs = tcl_xs1.expand(batch, x, Z)
        bcr_ys = bcr_ys1.expand(batch, y, Z)
        bcr_xs = bcr_xs1.expand(batch, y, Z)

        tcl_off_c = tcl_off_c.view(batch, x, 1, 2)
        bcr_off_c = bcr_off_c.view(batch, y, 1, 2)
        tcl_regr = tcl_regr.view(batch, x, 1, 2)
        bcr_regr = bcr_regr.view(batch, y, 1, 2)
        pc1_regr = p1_regr.view(batch, 1, Z, 2)

        tl_off_c = tl_off_c.view(batch, x, 1, 2)
        br_off_c = br_off_c.view(batch, 1, y, 2)
        tl_regr = tl_regr.view(batch, x, 1, 2)
        br_regr = br_regr.view(batch, 1, y, 2)
        p1_regr = p1_regr.view(batch, 1, Z, 2)


        tl_cxs = tl_xs + tl_off_c[..., 0] + tl_regr[..., 0]
        tl_cys = tl_ys + tl_off_c[..., 1] + tl_regr[..., 1]
        br_cxs = br_xs + br_off_c[..., 0] + br_regr[..., 0]
        br_cys = br_ys + br_off_c[..., 1] + br_regr[..., 1]


        tcl_cxs = tcl_xs + tcl_off_c[..., 0] + tcl_regr[..., 0]
        tcl_cys = tcl_ys + tcl_off_c[..., 1] + tcl_regr[..., 1]
        bcr_cxs = bcr_xs + bcr_off_c[..., 0] + bcr_regr[..., 0]
        bcr_cys = bcr_ys + bcr_off_c[..., 1] + bcr_regr[..., 1]

        tl_xs = tl_xs + tl_regr[..., 0]
        tl_ys = tl_ys + tl_regr[..., 1]
        br_xs = br_xs + br_regr[..., 0]
        br_ys = br_ys + br_regr[..., 1]
        c_xs = p1_xs + p1_regr[..., 0]
        c_ys = p1_ys + p1_regr[..., 1]

        tcl_xs = tcl_xs + tcl_regr[..., 0]
        tcl_ys = tcl_ys + tcl_regr[..., 1]
        bcr_xs = bcr_xs + bcr_regr[..., 0]
        bcr_ys = bcr_ys + bcr_regr[..., 1]
        cc_xs = pc1_xs + pc1_regr[..., 0]
        cc_ys = pc1_ys + pc1_regr[..., 1]

        tl_xs *= (inp_w / width)
        tl_ys *= (inp_h / height)
        br_xs *= (inp_w / width)
        br_ys *= (inp_h / height)
        c_xs *= (inp_w / width)
        c_ys *= (inp_h / height)

        tl_cxs *= (inp_w / width)
        tl_cys *= (inp_h / height)
        br_cxs *= (inp_w / width)
        br_cys *= (inp_h / height)

        tl_cxs *= tl_cxs.gt(0.0).type_as(tl_cxs)
        tl_cys *= tl_cys.gt(0.0).type_as(tl_cys)
        br_cxs *= br_cxs.gt(0.0).type_as(br_cxs)
        br_cys *= br_cys.gt(0.0).type_as(br_cys)
        tl_xs *= tl_xs.gt(0.0).type_as(tl_xs)
        tl_ys *= tl_ys.gt(0.0).type_as(tl_ys)
        br_xs *= br_xs.gt(0.0).type_as(br_xs)
        br_ys *= br_ys.gt(0.0).type_as(br_ys)
        c_xs *= c_xs.gt(0.0).type_as(c_xs)
        c_ys *= c_ys.gt(0.0).type_as(c_ys)

        bboxes = torch.stack((tl_xs, tl_ys, br_xs, br_ys), dim=3)

        cl_group_bboxes = torch.stack((tcl_xs, tcl_ys, cc_xs, cc_ys), dim=3) * 4
        cr_group_bboxes = torch.stack((bcr_xs, bcr_ys, cc_xs, cc_ys), dim=3) * 4

        cl_group_bboxes = cl_group_bboxes.view(-1, 4)
        cr_group_bboxes = cr_group_bboxes.view(-1, 4)
        tcl_group_bboxes = cl_group_bboxes[:, :2] - cl_group_bboxes[:, -2:]
        bcr_group_bboxes = cr_group_bboxes[:, :2] - cr_group_bboxes[:, -2:]

        tcl_group_bboxes_mask_1 = torch.sqrt((tcl_group_bboxes[:, 0] * tcl_group_bboxes[:, 0]) +\
                                  (tcl_group_bboxes[:, 1] * tcl_group_bboxes[:, 1])).gt(144 / 1.953)
        tcl_group_bboxes_mask_2 =torch.sqrt((tcl_group_bboxes[:, 0] * tcl_group_bboxes[:, 0]) +\
                                  (tcl_group_bboxes[:, 1] * tcl_group_bboxes[:, 1])).lt(12 / 1.953)
        tcl_group_bboxes_mask = (tcl_group_bboxes_mask_1 | tcl_group_bboxes_mask_2)
        tcl_group_bboxes_mask = tcl_group_bboxes_mask.reshape(1, -1)


        # remove too big or too small 
        bcr_group_bboxes_mask_1 = torch.sqrt((bcr_group_bboxes[:, 0] * bcr_group_bboxes[:, 0]) +\
                                  (bcr_group_bboxes[:, 1] * bcr_group_bboxes[:, 1])).gt(144 / 1.953)
        bcr_group_bboxes_mask_2 = torch.sqrt((bcr_group_bboxes[:, 0] * bcr_group_bboxes[:, 0]) +\
                                  (bcr_group_bboxes[:, 1] * bcr_group_bboxes[:, 1])).lt(12 / 1.953)
        bcr_group_bboxes_mask =( bcr_group_bboxes_mask_1 | bcr_group_bboxes_mask_2)
        bcr_group_bboxes_mask = bcr_group_bboxes_mask.reshape(a, Z)
        tcl_t_mask = tcl_group_bboxes_mask.expand(a, a * Z)
        bcr_t_mask = bcr_group_bboxes_mask
        for z in range(a - 1):
            bcr_t_mask = torch.cat((bcr_t_mask, bcr_group_bboxes_mask), 1)


        # Calculate the angle between vectors
        bcr_group_bboxes = bcr_group_bboxes.cpu().numpy()
        tcl_group_bboxes = tcl_group_bboxes.cpu().numpy()
        tcl_group_bboxes_n = np.linalg.norm(tcl_group_bboxes, axis=1, keepdims=True)
        bcr_group_bboxes_n = np.linalg.norm(bcr_group_bboxes, axis=1, keepdims=True)
        tcl_group_bboxes = tcl_group_bboxes / tcl_group_bboxes_n
        bcr_group_bboxes = bcr_group_bboxes / bcr_group_bboxes_n
        tcl_group_bboxes = tcl_group_bboxes.reshape(1, -1, 2)
        bcr_group_bboxes = bcr_group_bboxes.reshape(a, Z, 2)
        tcl_group_bboxes = torch.from_numpy(tcl_group_bboxes)
        bcr_group_bboxes = torch.from_numpy(bcr_group_bboxes)
        tcl_t = tcl_group_bboxes.expand(a, a * Z, 2)
        bcr_t = bcr_group_bboxes
        for z in range(a - 1):
            bcr_t = torch.cat((bcr_t, bcr_group_bboxes), 1)
        t_t = torch.sum(tcl_t * bcr_t, 2)
        t_t_mask = abs(t_t).gt(0.259)
        t_t_mask_2 = abs(t_t).lt(0.8)

        #calculate scores
        tcl_scores = tl_scores.view(batch, a, 1).expand(batch, a, Z)
        bcr_scores = br_scores.view(batch, a, 1).expand(batch, a, Z).view(a, Z)
        pc_scores = p1_scores.view(batch, 1, Z).expand(batch, a, Z)
        tc_scores = (tcl_scores + pc_scores)
        tc_scores = tc_scores.view(1, -1)
        tc_scores = tc_scores.expand(a, a * Z)
        bc_scores = bcr_scores
        for z in range(a - 1):
            bc_scores = torch.cat((bc_scores, bcr_scores), 1)
        cc_scores = (tc_scores + bc_scores) / 3

        #second group
        tcl_cxs *= 4
        tcl_cys *= 4
        cc_xs *= 4
        cc_ys *= 4
        bcr_cxs *= 4
        bcr_cys *= 4
        tcl_xs *= 4
        tcl_ys *= 4
        bcr_xs *= 4
        bcr_ys *= 4
        lc_center = torch.stack((tcl_cxs, tcl_cys, cc_xs, cc_ys), dim=3)
        lc_center = lc_center.view(-1, 4)
        rc_center = torch.stack((bcr_cxs, bcr_cys, cc_xs, cc_ys), dim=3)
        rc_center = rc_center.view(-1, 4)
        larea_bbox = torch.sqrt(((tcl_xs - cc_xs) * (tcl_xs - cc_xs)) + ((tcl_ys - cc_ys) * (tcl_ys - cc_ys))) + 1e-16
        lc_bbox = torch.sqrt(((tcl_cxs - cc_xs) * (tcl_cxs - cc_xs)) + ((tcl_cys - cc_ys) * (tcl_cys - cc_ys))) + 1e-16
        lc_scores = lc_bbox / (larea_bbox / 4) + 1e-16
        lc_scores = 1 / lc_scores
        lc_scores = torch.log(lc_scores)
        lc_scores = torch.clamp(lc_scores, min=0.0).reshape(1, -1)
        lc_scores = lc_scores.expand(a, a * Z)
        lc_scores_mask = lc_scores.le(0.0)
        rarea_bbox = torch.sqrt(((bcr_xs - cc_xs) * (bcr_xs - cc_xs)) + ((bcr_ys - cc_ys) * (bcr_ys - cc_ys))) + 1e-16
        rc_bbox = torch.sqrt(((bcr_cxs - cc_xs) * (bcr_cxs - cc_xs)) + ((bcr_cys - cc_ys) * (bcr_cys - cc_ys))) + 1e-16
        rc_scores = rc_bbox / (rarea_bbox / 4) + 1e-16
        rc_scores = 1 / rc_scores
        rc_scores = torch.log(rc_scores)
        rc_scores = torch.clamp(rc_scores, min=0.0).reshape(a, Z)
        rc_scores_1 = rc_scores
        for q in range(a - 1):
            rc_scores_1 = torch.cat((rc_scores_1, rc_scores), 1)
        rc_scores = rc_scores_1
        rc_scores_mask = rc_scores.le(0.0)
        lr_sores = rc_scores + lc_scores
        lr_scores_mask = lc_scores_mask | rc_scores_mask
        t_t = t_t.cuda()
        cc_scores[lr_scores_mask.cuda() & t_t_mask.cuda()] = -1
        cc_scores[t_t_mask_2] = cc_scores[t_t_mask_2] + 1 * (1 - abs(t_t[t_t_mask_2])) * (
                    cc_scores[t_t_mask_2] - cc_scores[t_t_mask_2] * cc_scores[t_t_mask_2])
        cc_scores[~lr_scores_mask] = cc_scores[~lr_scores_mask] + (1 - torch.exp(-lr_sores[~lr_scores_mask])) * (
                    cc_scores[~lr_scores_mask] - cc_scores[~lr_scores_mask] * cc_scores[~lr_scores_mask])


        group_bboxes = torch.stack((tl_xs, tl_ys, br_xs, br_ys), dim=3)
        center = torch.stack((tl_cxs, tl_cys, br_cxs, br_cys), dim=3)

        cre = torch.zeros_like(center)
        area_bbox = torch.sqrt(((br_xs - tl_xs) * (br_xs - tl_xs)) + ((tl_ys - br_ys) * (tl_ys - br_ys))) + 1e-16
        c_bbox = torch.sqrt(((br_cxs - tl_cxs) * (br_cxs - tl_cxs)) + ((tl_cys - br_cys) * (tl_cys - br_cys))) + 1e-16
        mask1 = area_bbox.le(18)
        # c_scores= c_bbox
        c_scores = c_bbox / (area_bbox / 2) + 1e-16
        # c_scores[mask1] = c_bbox[mask1] / (area_bbox[mask1] / 1) + 1e-16
        region_mask = area_bbox.gt(82)
        region_mask_1 = area_bbox.lt(9)
        c_scores = 1 / c_scores
        c_scores = torch.log(c_scores)
        c_scores = torch.clamp(c_scores, min=0.0)
        c_scores[region_mask] = 0
        c_scores[region_mask_1] = 0

        cc_scores[bcr_t_mask] = -1
        cc_scores[tcl_t_mask] = -1
        pen_scores = cc_scores.clone()


        dig_scores, dig_inds, dig_clses, dig_ys, dig_xs = _topk(cc_scores.unsqueeze(0).unsqueeze(0), K=KK)
        #     print(dig_scores)
        ldig_xs = (dig_xs / Z).int().view(KK, -1)
        cdig_xs = (dig_xs % Z).int().view(KK, -1)
        rdig_xs = dig_ys.int().view(KK, -1)
        #     print(ldig_xs)
        #     print(rdig_xs)
        #     print(rdig_xs)
        p_box = torch.cat((ldig_xs, rdig_xs, cdig_xs), dim=1)
        #     print(p_box)
        fp_box = []
        fdig_scores = []
        l_ready = []
        r_ready = []
        c_ready = []
        for r, w in enumerate(p_box):
            if (w[0] in l_ready) or (w[1] in r_ready) or (w[2] in c_ready):
                continue
            #         print(w[0])
            l_ready.append(w[0])
            r_ready.append(w[1])
            c_ready.append(w[2])
            fp_box.append(w.tolist())
            #         print(dig_scores.shape)
            fdig_scores.append(dig_scores[0, r].tolist())
        #     print(p_box)
        dig_scores = torch.Tensor(fdig_scores).cuda()
        #     print(dig_scores)
        p_box = torch.Tensor(fp_box).cuda()
        #     print(p_box)


        tcl_xs_2 = tcl_xs[:, :, 0].view(a, -1)
        tcl_ys_2 = tcl_ys[:, :, 0].view(a, -1)
        tcl_2 = torch.cat((tcl_xs_2, tcl_ys_2), 1)
        bcr_xs_2 = bcr_xs[:, :, 0].view(a, -1)
        bcr_ys_2 = bcr_ys[:, :, 0].view(a, -1)
        bcr_2 = torch.cat((bcr_xs_2, bcr_ys_2), 1)
        cc_xs_2 = cc_xs[:, 0, :].view(Z, -1)
        cc_ys_2 = cc_ys[:, 0, :].view(Z, -1)
        cc_2 = torch.cat((cc_xs_2, cc_ys_2), 1)
        lrc_box = torch.zeros((p_box.size()[0], 6))
        for t in range(p_box.size()[0]):
            lrc_box[t][0:2] = tcl_2[p_box[t][0].long()]
            lrc_box[t][2:4] = bcr_2[p_box[t][1].long()]
            lrc_box[t][4:6] = cc_2[p_box[t][2].long()]
        c_cen = (lrc_box[:, :2] + lrc_box[:, 2:4]) / 2
        c_result = torch.cat((lrc_box, c_cen), 1)
        c_ten = c_result[:, 4:6] - c_result[:, 6:]
        c_rad = ((c_result[:, :2] - c_result[:, 2:4]) / 2)
        c_rad = torch.sqrt(torch.sum(c_rad * c_rad, 1))
        c_dis = torch.sqrt(torch.sum(c_ten * c_ten, 1))
        c_coefficient = c_rad / c_dis
        c_ten = c_ten * c_coefficient.view(-1, 1)
        c_result[:, 4:6] = c_ten + c_result[:, 6:]
        c_an_point = -c_ten + c_result[:, 6:]
        c_fianlly = torch.cat((c_result[:, :2], c_result[:, 4:6], c_result[:, 2:4], c_result[:, -2:]), 1)
        c_w = torch.sqrt(
            torch.sum((c_fianlly[:, :2] - c_fianlly[:, 2:4]) * (c_fianlly[:, :2] - c_fianlly[:, 2:4]), 1)).view(-1, 1)
        c_h = torch.sqrt(
            torch.sum((c_fianlly[:, 4:6] - c_fianlly[:, 2:4]) * (c_fianlly[:, 4:6] - c_fianlly[:, 2:4]), 1)).view(-1, 1)
        #         print(w[2])
        c_theta = c_fianlly[:, 2:4] - c_fianlly[:, :2]
        c_theta[:, 1] = -c_theta[:, 1]
        c_theta1 = []
        for y in c_theta:
            c_theta1.append(math.degrees(math.atan2(y[1], y[0])))
        c_theta1 = torch.Tensor(c_theta1).view(-1, 1).to("cuda")
        #     print(c_cen.device, c_w.device, c_h.device, theta1.device)
        c_boxes = torch.cat((c_cen.cuda(), c_w.cuda(), c_h.cuda(), c_theta1.cuda()), 1)
        # print(c_boxes)

        #     print(c_scores)
        for i in range(batch):
            if c_scores[i].max() == 0:
                c_scores[i] = c_scores[i]
            else:
                c_scores[i] = c_scores[i] / c_scores[i].max()
        tl_scores = tl_scores.view(batch, a, 1).expand(batch, a, a)
        br_scores = br_scores.view(batch, 1, a).expand(batch, a, a)
        aa = (tl_scores + br_scores) / 2
        # scores = (tl_scores + br_scores) / 2 * c_scores  # scores for all possible boxes

        c_scores = c_scores + (c_scores - c_scores * c_scores)
        aa *= c_scores
        aa = aa + (aa - aa * aa)

        # scores = scores + (scores - scores * scores)
        scores = aa
        scores = scores.view(batch, -1)
        aa = aa.view(batch, -1, 1)

        scores, inds = torch.topk(scores, a)
        scores = scores.view(batch, -1)

        aa = _gather_feat(aa, inds)
        bboxes = bboxes.view(batch, -1, 4)
        bboxes = _gather_feat(bboxes, inds)

        center = center.view(batch, -1, 4)
        center = _gather_feat(center, inds)

        thersold_1 = thersold
        thersold = thersold
        tl_scores = tl_scores.reshape(batch, -1, 1)
        tl_scores = _gather_feat(tl_scores, inds)
        br_scores = br_scores.reshape(batch, -1, 1)
        br_scores = _gather_feat(br_scores, inds)
        br_scores = br_scores.view(-1)
        tl_scores = tl_scores.view(-1)
        d = aa
        c = d.gt(thersold)
        c = c.view(batch, -1)
        rbox = []
        source = []
        rresult = []
        boxes = torch.Tensor(()).cuda()
        aaaaa = torch.Tensor(()).cuda()
        for i in range(1):
            d = torch.nonzero(c)
            if len(d) == 0:
                continue
            thersold = torch.nonzero(c[i])[-1]
            centers = center[i, :thersold + 1]
            box = bboxes[i, :thersold + 1]
            br_scores1 = br_scores[:thersold + 1].view(-1, 1)
            tl_scores1 = tl_scores[:thersold + 1].view(-1, 1)
            aaaaa = (aa[i].view(-1)[:thersold + 1])

            centers1 = (centers[:, :2] * tl_scores1 + centers[:, 2:] * br_scores1) / (tl_scores1 + br_scores1)
            result = torch.cat((box, centers1), 1)

            fp_box_1 = []
            scores_1 = []
            l_ready_1 = []
            r_ready_1= []
            for r, w in enumerate(result):
                if (w[:2].tolist() in l_ready_1) or (w[2:4].tolist() in r_ready_1):
                    continue
                l_ready_1.append(w[:2].tolist())
                r_ready_1.append(w[2:4].tolist())
                fp_box_1.append(w.tolist())
                scores_1.append(aaaaa[r].tolist())
            aaaaa = torch.Tensor(scores_1).view(-1).cuda()
            result = torch.Tensor(fp_box_1).cuda()

            cen = (result[:, :2] + result[:, 2:4]) / 2
            result = torch.cat((result, cen), 1)
            ten = result[:, 4:6] - result[:, 6:]
            rad = ((result[:, :2] - result[:, 2:4]) / 2)
            rad = torch.sqrt(torch.sum(rad * rad, 1))
            dis = torch.sqrt(torch.sum(ten * ten, 1))
            coefficient = rad / dis
            ten = ten * coefficient.view(-1, 1)
            result[:, 4:6] = ten + result[:, 6:]
            an_point = -ten + result[:, 6:]
            fianlly = torch.cat((result[:, :2], result[:, 4:6], result[:, 2:4], result[:, -2:]), 1)
            w = torch.sqrt(
                torch.sum((fianlly[:, :2] - fianlly[:, 2:4]) * (fianlly[:, :2] - fianlly[:, 2:4]), 1)).view(-1, 1)
            h = torch.sqrt(
                torch.sum((fianlly[:, 4:6] - fianlly[:, 2:4]) * (fianlly[:, 4:6] - fianlly[:, 2:4]), 1)).view(-1, 1)
            #         print(w[2])
            theta = fianlly[:, 2:4] - fianlly[:, :2]
            theta[:, 1] = -theta[:, 1]
            theta1 = []
            for j in theta:
                theta1.append(math.degrees(math.atan2(j[1], j[0])))
            theta1 = torch.Tensor(theta1).view(-1, 1).to("cuda")
            boxes = torch.cat((cen, w, h, theta1), 1)
            rresult.append(fianlly)

        dig_scores_mask = dig_scores.gt(thersold_1)
        c_boxes = c_boxes[dig_scores_mask]
        dig_scores = dig_scores[dig_scores_mask]
        boxes = c_boxes
        aaaaa = dig_scores

        # # first
        # if boxes == None:
        #     boxes = torch.cat((c_boxes, pen_c_boxes), 0)
        #     aaaaa = torch.cat((dig_scores, pen_dig_scores))
        # else:
        #     boxes = torch.cat((c_boxes, pen_c_boxes, ), 0)
        #     aaaaa = torch.cat((dig_scores, pen_dig_scores, ))

        # second
        # if boxes == None:
        #     boxes = torch.cat((c_boxes, pen_c_boxes), 0)
        #     aaaaa = torch.cat((dig_scores, pen_dig_scores))
        # else:
        #     boxes = torch.cat((boxes), 0)
        #     aaaaa = torch.cat((aaaaa))
        # if aaaaa != None:
        #     for i, j in enumerate(aaaaa):
        #         if j == 0:
        #             boxes = boxes[:i]
        #             aaaaa = aaaaa[:i]
        #             break
        rbox.append(boxes)
        source.append(aaaaa)

        return rbox, source, rresult

def overlay_boxes(image, predictions, source, anntype, xylimits, res):
    """
    Adds the predicted boxes on top of the image
    Arguments:
        image (np.ndarray): an image as returned by OpenCV
        predictions (BoxList): the result of the computation by the model.
            It should contain the field `labels`.
    """
    # labels = predictions.get_field("labels")
    imgsrc = image.copy()
    predictions = np.array(predictions)
    oriens = predictions[:, -1]
    boxes = predictions[:, :-1]
    scores = (source)
    xclist = []
    yclist = []
    alphalist = []
    detections_per_frame = []
    j = 0
    # print('\noriens:',oriens.size(),'boxes:',boxes.size(),'==========\n')
    for box, orien, score in zip(boxes, oriens, scores):
        color = {'targets': (155, 255, 255), 'output': (155, 255, 55)}
        offset = {'targets': 0, 'output': 0}

        # if not (xylimits[0] < xc < xylimits[1] and xylimits[2] < yc < xylimits[3]):
        #     # continue
        #     pass
        # cv2.line(image, (xylimits[0], xylimits[3]), (xylimits[1], xylimits[3]), color[anntype], 1, cv2.LINE_AA)
        # cv2.line(image, (xylimits[0], xylimits[2]), (xylimits[1], xylimits[2]), color[anntype], 1, cv2.LINE_AA)
        # cv2.line(image, (xylimits[0], xylimits[3]), (xylimits[0], xylimits[2]), color[anntype], 1, cv2.LINE_AA)
        # cv2.line(image, (xylimits[1], xylimits[3]), (xylimits[1], xylimits[2]), color[anntype], 1, cv2.LINE_AA)

        # if l * w <= 1:
        #     continue

        box = bBox_2D(box[2], box[3], box[0], box[1], orien)
        box.scale(1000 / 512, 0, 0)
        # box.resize(1.2)
        box.bBoxCalcVertxex()

        rad = box.alpha * math.pi / 180
        cv2.line(image, box.vertex1, box.vertex2, color[anntype], 1, cv2.LINE_AA)
        cv2.line(image, box.vertex2, box.vertex4, color[anntype], 1, cv2.LINE_AA)
        cv2.line(image, box.vertex3, box.vertex1, color[anntype], 1, cv2.LINE_AA)
        cv2.line(image, box.vertex4, box.vertex3, color[anntype], 1, cv2.LINE_AA)
        cv2.putText(image, str(j), box.vertex1, cv2.FONT_HERSHEY_PLAIN,
                    1.0, (255, 255, 255), thickness = 1)
        j += 1

        # if anntype == 'output':
        #     print('+++++')
        #     print(box.vertex4, box.vertex3, box.vertex2, box.vertex1, '====', l * w, '\t', l, '\t', w, '\t angle',
        #           box.alpha, ' score ', score)
        #     detections_per_frame.append([score, (box.yc - 100) / 30.0, (box.xc - 500) / 30.0, rad])
        # else:
        #     # print(box.vertex4, box.vertex3, box.vertex2, box.vertex1, '====', l * w, '\t', l, '\t', w, '\t angle',
        #     # box.alpha)
        #     pass

        point = int(box.xc - box.length * 0.8 * np.sin(rad)), int(box.yc + box.length * 0.8 * np.cos(rad))
        cv2.line(image, (int(box.xc), int(box.yc)),
                 point,
                 color[anntype], 2, cv2.LINE_AA)
        # if anntype == 'output':
        #     cv2.putText(image, str(score.numpy()), point, fontFace=1, fontScale=1.5, color=(255, 0, 255))

    image = cv2.addWeighted(imgsrc, 0.4, image, 0.6, 0)
    if anntype == 'output':
        detections_per_frame = np.array(detections_per_frame)
    else:
        detections_per_frame = []
    return np.array([xclist, yclist], dtype=float), np.array(alphalist, dtype=float), detections_per_frame


def _neg_loss(preds, gt):
    pos_inds = gt.eq(1)
    neg_inds = gt.lt(1)
    #
    neg_weights = torch.pow(1 - gt[neg_inds], 4)
    #
    loss = 0
    for pred in preds:
        pos_pred = pred[pos_inds]
        neg_pred = pred[neg_inds]
        #
        pos_loss = torch.log(pos_pred) * torch.pow(1 - pos_pred, 2)
        neg_loss = torch.log(1 - neg_pred) * torch.pow(neg_pred, 2) * neg_weights
        #
        num_pos = pos_inds.float().sum()
        pos_loss = pos_loss.sum()
        neg_loss = neg_loss.sum()
        #
        # avoid the error when num_pos is zero
        if pos_pred.nelement() == 0:
            loss = loss - neg_loss
        else:
            loss = loss - (pos_loss + neg_loss) / num_pos
    return loss


def _sigmoid(x):
    x = torch.clamp(x.sigmoid_(), min=1e-4, max=1 - 1e-4)
    return x


def _ae_loss(tag0, tag1, mask):  # mask means only consider the loss of positive corner
    num = mask.sum(dim=1, keepdim=True).float()
    tag0 = tag0.squeeze()
    tag1 = tag1.squeeze()
    #
    tag_mean = (tag0 + tag1) / 2
    #
    tag0 = torch.pow(tag0 - tag_mean, 2) / (num + 1e-4)
    tag0 = tag0[mask].sum()
    tag1 = torch.pow(tag1 - tag_mean, 2) / (num + 1e-4)
    tag1 = tag1[mask].sum()
    pull = tag0 + tag1  # this is pull loss, smaller means tag0 and tag1 are more similiar
    #
    mask = mask.unsqueeze(1) + mask.unsqueeze(2)
    mask = mask.eq(2)
    num = num.unsqueeze(2)
    num2 = (num - 1) * num
    dist = tag_mean.unsqueeze(1) - tag_mean.unsqueeze(2)
    dist = 1 - torch.abs(dist)
    dist = nn.functional.relu(dist, inplace=True)
    dist = dist - 1 / (num + 1e-4)
    dist = dist / (num2 + 1e-4)
    dist = dist[mask]
    push = dist.sum()
    return pull, push


def _regr_loss(regr, gt_regr, mask):  # regression loss
    num = mask.float().sum()
    mask = mask.unsqueeze(2).expand_as(gt_regr)

    regr = regr[mask]
    gt_regr = gt_regr[mask]

    regr_loss = nn.functional.smooth_l1_loss(regr, gt_regr, size_average=False)
    regr_loss = regr_loss / (num + 1e-4)
    return regr_loss


def gaussian2D(shape, sigma=1):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m+1,-n:n+1]
    #
    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))

    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h

def draw_gaussian(heatmap, center, radius, k=1):
    diameter = 2 * radius + 1
    gaussian  = gaussian2D((diameter, diameter), sigma=diameter / 6)
    x, y = center
    #
    height, width = heatmap.shape[0:2]
    #process the border
    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)
    #
    masked_heatmap  = heatmap[y - top:y + bottom, x - left:x + right]
    masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right]
    np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)

def gaussian_radius(det_size, min_overlap):
    height, width = det_size

    a1  = 1
    b1  = (height + width)
    c1  = width * height * (1 - min_overlap) / (1 + min_overlap)
    sq1 = np.sqrt(b1 ** 2 - 4 * a1 * c1)
    r1  = (b1 - sq1) / (2 * a1)

    a2  = 4
    b2  = 2 * (height + width)
    c2  = (1 - min_overlap) * width * height
    sq2 = np.sqrt(b2 ** 2 - 4 * a2 * c2)
    r2  = (b2 - sq2) / (2 * a2)

    a3  = 4 * min_overlap
    b3  = -2 * min_overlap * (height + width)
    c3  = (min_overlap - 1) * width * height
    sq3 = np.sqrt(b3 ** 2 - 4 * a3 * c3)
    r3  = (b3 + sq3) / (2 * a3)
    return min(r1, r2, r3)
