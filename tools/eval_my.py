import logging
import os
from collections import OrderedDict
import torch
import sys
import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, hooks, launch
from detectron2.evaluation import (
    CityscapesEvaluator,
    COCOEvaluator,
    COCOPanopticEvaluator,
    DatasetEvaluators,
    LVISEvaluator,
    PascalVOCDetectionEvaluator,
    SemSegEvaluator,
    verify_results,
)
from detectron2.modeling import GeneralizedRCNNWithTTA

import numpy as np
import numpy.linalg as la
import pandas as pd
import datetime
import shutil

from tqdm import tqdm
from detectron2.structures import pairwise_iou_rotated
from detectron2.structures import RotatedBoxes

from bBox_2D import bBox_2D
import math
import cv2

import matplotlib.pyplot as plt
import random
from time import time

########## load model and data ##########
class Trainer(DefaultTrainer):
    """
    We use the "DefaultTrainer" which contains pre-defined default logic for
    standard training workflow. They may not work for you, especially if you
    are working on a new research project. In that case you can use the cleaner
    "SimpleTrainer", or write your own training loop. You can use
    "tools/plain_train_net.py" as an example.
    """

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        """
        Create evaluator(s) for a given dataset.
        This uses the special metadata "evaluator_type" associated with each builtin dataset.
        For your own dataset, you can simply create an evaluator manually in your
        script and do not have to worry about the hacky if-else logic here.
        """
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        evaluator_list = []
        evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
        if evaluator_type in ["sem_seg", "coco_panoptic_seg"]:
            evaluator_list.append(
                SemSegEvaluator(
                    dataset_name,
                    distributed=True,
                    num_classes=cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES,
                    ignore_label=cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE,
                    output_dir=output_folder,
                )
            )
        if evaluator_type in ["coco", "coco_panoptic_seg"]:
            evaluator_list.append(COCOEvaluator(dataset_name, cfg, True, output_folder))
        if evaluator_type == "coco_panoptic_seg":
            evaluator_list.append(COCOPanopticEvaluator(dataset_name, output_folder))
        elif evaluator_type == "cityscapes":
            assert (
                    torch.cuda.device_count() >= comm.get_rank()
            ), "CityscapesEvaluator currently do not work with multiple machines."
            return CityscapesEvaluator(dataset_name)
        elif evaluator_type == "pascal_voc":
            return PascalVOCDetectionEvaluator(dataset_name)
        elif evaluator_type == "lvis":
            return LVISEvaluator(dataset_name, cfg, True, output_folder)
        if len(evaluator_list) == 0:
            raise NotImplementedError(
                "no Evaluator for the dataset {} with the type {}".format(
                    dataset_name, evaluator_type
                )
            )
        elif len(evaluator_list) == 1:
            return evaluator_list[0]
        return DatasetEvaluators(evaluator_list)

    @classmethod
    def test_with_TTA(cls, cfg, model):
        logger = logging.getLogger("detectron2.trainer")
        # In the end of training, run an evaluation with TTA
        # Only support some R-CNN models.
        logger.info("Running inference with test-time augmentation ...")
        model = GeneralizedRCNNWithTTA(cfg, model)
        evaluators = [
            cls.build_evaluator(
                cfg, name, output_folder=os.path.join(cfg.OUTPUT_DIR, "inference_TTA")
            )
            for name in cfg.DATASETS.TEST
        ]
        res = cls.test(cfg, model, evaluators)
        res = OrderedDict({k + "_TTA": v for k, v in res.items()})
        return res

    def test_ex(self):
        data = next(self._data_loader_iter)
        head_out, batched_inputs, a, b = self.model(data)
        #         print(loss_dict)
        return head_out, batched_inputs, a, b

    def inference(self):
        return self.data_loader, self.model


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    #     cfg.merge_from_file(args.config_file)
    #     cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def main(args):
    cfg = setup(args)

    #     if args.eval_only:
    #         model = Trainer.build_model(cfg)
    #         DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
    #             cfg.MODEL.WEIGHTS, resume=args.resume
    #         )
    #         res = Trainer.test(cfg, model)
    #         if cfg.TEST.AUG.ENABLED:
    #             res.update(Trainer.test_with_TTA(cfg, model))
    #         if comm.is_main_process():
    #             verify_results(cfg, res)
    #         return res

    """
    If you'd like to do anything fancier than the standard training logic,print
    consider writing your own training loop or subclassing the trainer.
    """
    trainer = Trainer(cfg, False)
    # trainer.resume_or_load(resume=None)
    if cfg.TEST.AUG.ENABLED:
        trainer.register_hooks(""
                               [hooks.EvalHook(0, lambda: trainer.test_with_TTA(cfg, trainer.model))]
                               )
    trainer.resume_or_load()
    trainer.model.training = False
    return trainer.inference()


args = None
port = 2 ** 15 + 2 ** 14 + hash(os.getuid() if sys.platform != "win32" else 1) % 2 ** 14
default_dist_url = "tcp://127.0.0.1:{}".format(port)
print("Command Line Args:", args)

a = main(None)
b, c = a
data_loader = iter(b)
model = c


########## generate the prediction boxes and calculate the AP ##########
def overlay_boxes(image, predictions, source, anntype, num, targ):
    """
    Adds the predicted boxes on top of the image
    Arguments:
        image (np.ndarray): an image as returned by OpenCV
        predictions (BoxList): the result of the computation by the model.
            It should contain the field `labels`.
    """
    # labels = predictions.get_field("labels")
    imgsrc = image.copy()
    #     imgsrc = 255 - imgsrc
    predictions = np.array(predictions).reshape(-1, 5)
    oriens = predictions[:, -1]
    boxes = predictions[:, :-1]
    scores = (source)
    oriens_tar = targ[:, -1]
    boxes_tar = targ[:, :-1]
    xclist = []
    yclist = []
    alphalist = []
    detections_per_frame = []
    j = 0
    # print('\noriens:',oriens.size(),'boxes:',boxes.size(),'==========\n')
    #     print(">>>>>>>>>")
    #     print(boxes)
    #     print(oriens)
    #     print(scores)
    for box, orien, score in zip(boxes, oriens, scores):
        color = {'targets': (155, 255, 255), 'output': (64, 145, 61)}
        offset = {'targets': 0, 'output': 0}
        orien = -orien
        # if not (xylimits[0] < xc < xylimits[1] and xylimits[2] < yc < xylimits[3]):
        #     # continue
        #     pass
        # cv2.line(image, (xylimits[0], xylimits[3]), (xylimits[1], xylimits[3]), color[anntype], 1, cv2.LINE_AA)
        # cv2.line(image, (xylimits[0], xylimits[2]), (xylimits[1], xylimits[2]), color[anntype], 1, cv2.LINE_AA)
        # cv2.line(image, (xylimits[0], xylimits[3]), (xylimits[0], xylimits[2]), color[anntype], 1, cv2.LINE_AA)
        # cv2.line(image, (xylimits[1], xylimits[3]), (xylimits[1], xylimits[2]), color[anntype], 1, cv2.LINE_AA)

        # if l * w <= 1:
        #     continue
        #         print(box[2], box[3], box[0], box[1], orien)
        box = bBox_2D(box[3], box[2], box[0], box[1], orien)
        #         box.scale(1000 / 512, 0, 0)
        # box.resize(1.2)
        box.bBoxCalcVertxex()
        #         print(box.vertex1, box.vertex2, box.vertex3, box.vertex4)
        rad = box.alpha * math.pi / 180
        #         cv2.line(imgsrc, (100, 100), (20, 56), (155, 255, 55), 1, cv2.LINE_AA)
        cv2.line(imgsrc, box.vertex1, box.vertex2, color[anntype], 1, cv2.LINE_AA)
        cv2.line(imgsrc, box.vertex2, box.vertex4, color[anntype], 1, cv2.LINE_AA)
        cv2.line(imgsrc, box.vertex3, box.vertex1, color[anntype], 1, cv2.LINE_AA)
        cv2.line(imgsrc, box.vertex4, box.vertex3, color[anntype], 1, cv2.LINE_AA)
        j += 1
        point = int(box.xc - box.length * 0.8 * np.sin(rad)), int(box.yc + box.length * 0.8 * np.cos(rad))
        point2 = int(box.xc + box.length * 0.8 * np.sin(rad)), int(box.yc - box.length * 0.8 * np.cos(rad))
        cv2.line(imgsrc, (int(box.xc), int(box.yc)),
                 point,
                 color[anntype], 2, cv2.LINE_AA)
        cv2.line(imgsrc, (int(box.xc), int(box.yc)),
                 point2,
                 color[anntype], 2, cv2.LINE_AA)
    for box, orien in zip(boxes_tar, oriens_tar):
        color = {'targets': (155, 255, 255), 'output': (155, 255, 55)}
        offset = {'targets': 0, 'output': 0}
        orien = -orien
        # if not (xylimits[0] < xc < xylimits[1] and xylimits[2] < yc < xylimits[3]):
        #     # continue
        #     pass
        # cv2.line(image, (xylimits[0], xylimits[3]), (xylimits[1], xylimits[3]), color[anntype], 1, cv2.LINE_AA)
        # cv2.line(image, (xylimits[0], xylimits[2]), (xylimits[1], xylimits[2]), color[anntype], 1, cv2.LINE_AA)
        # cv2.line(image, (xylimits[0], xylimits[3]), (xylimits[0], xylimits[2]), color[anntype], 1, cv2.LINE_AA)
        # cv2.line(image, (xylimits[1], xylimits[3]), (xylimits[1], xylimits[2]), color[anntype], 1, cv2.LINE_AA)

        # if l * w <= 1:
        #     continue
        #         print(box[2], box[3], box[0], box[1], orien)
        box = bBox_2D(box[3], box[2], box[0], box[1], orien)
        #         box.scale(1000 / 512, 0, 0)
        # box.resize(1.2)
        box.bBoxCalcVertxex()
        #         print(box.vertex1, box.vertex2, box.vertex3, box.vertex4)
        rad = box.alpha * math.pi / 180
    #         cv2.line(imgsrc, (100, 100), (20, 56), (155, 255, 55), 1, cv2.LINE_AA)

    #         cv2.line(imgsrc, box.vertex1, box.vertex2, (155, 255, 255), 1, cv2.LINE_AA)
    #         cv2.line(imgsrc, box.vertex2, box.vertex4, (155, 255, 255), 1, cv2.LINE_AA)
    #         cv2.line(imgsrc, box.vertex3, box.vertex1, (155, 255, 255), 1, cv2.LINE_AA)
    #         cv2.line(imgsrc, box.vertex4, box.vertex3, (155, 255, 255), 1, cv2.LINE_AA)
    #         cv2.putText(imgsrc, "t", box.vertex1, cv2.FONT_HERSHEY_PLAIN,
    #                     1.0, (255, 255, 255), thickness = 1)

    # if anntype == 'output':
    #     print('+++++')
    #     print(box.vertex4, box.vertex3, box.vertex2, box.vertex1, '====', l * w, '\t', l, '\t', w, '\t angle',
    #           box.alpha, ' score ', score)
    #     detections_per_frame.append([score, (box.yc - 100) / 30.0, (box.xc - 500) / 30.0, rad])
    # else:
    #     # print(box.vertex4, box.vertex3, box.vertex2, box.vertex1, '====', l * w, '\t', l, '\t', w, '\t angle',
    #     # box.alpha)
    #     pass

    #         point = int(box.xc - box.length * 0.8 * np.sin(rad)), int(box.yc + box.length * 0.8 * np.cos(rad))
    #         cv2.line(imgsrc, (int(box.xc), int(box.yc)),
    #                  point,
    #                  color[anntype], 2, cv2.LINE_AA)
    #         # if anntype == 'output':
    #     cv2.putText(image, str(score.numpy()), point, fontFace=1, fontScale=1.5, color=(255, 0, 255))

    #     imgsrc = cv2.addWeighted(imgsrc, 0.4, imgsrc, 0.6, 0)
    cv2.imwrite("./result/img_" + str(num) + ".jpg", imgsrc)
    if anntype == 'output':
        detections_per_frame = np.array(detections_per_frame)
    else:
        detections_per_frame = []


def test(thresh):
    model.eval()
    results_dict = {}
    cpu_device = torch.device("cpu")
    eval_distance = []
    eval_angle = []
    outcenterlist = 0
    tarcenterlist = 0
    infertimelist = []
    detections_total = {}
    for i, batch in enumerate(tqdm(b.dataset)):
        images, targets, image_ids = batch["image"], batch["instances"].gt_boxes, batch["image_id"]
        images = images
        with torch.no_grad():
            output = model.inference(thresh, batch)
            #             print(output)
            #             time_end = time.time()
            if not output[0]:
                output = torch.Tensor([[]])
                ss = torch.Tensor([[]])
            else:
                ss = output[1][0]
                output = output[0][0].to(cpu_device)
        #             print(output)
        images = images.permute(1, 2, 0).numpy()
        images = images.astype(np.uint8)
        output_1 = output.numpy()
        ss_1 = ss.cpu().numpy()
        targ = targets.tensor.cpu().numpy()
        overlay_boxes(images, output_1, ss_1, "output", i, targ)
        results_dict.update({image_ids: output})
        output = output.reshape(-1, 5)

        outcenter = output[:, :2].numpy()
        outalpha = output[:, -1:].numpy()
        tarcenter = targets.tensor[:, :2].numpy()
        taralpha = targets.tensor[:, -1:].numpy()
        outcenter = outcenter.T
        tarcenter = tarcenter.T
        m, n = outcenter.shape
        o, p = tarcenter.shape
        tarcenterlist += p
        if n == 0:
            continue
        outcenterlist += n
        if p == 0:
            continue

        D = np.zeros([n, p])
        A = np.zeros([n, p])
        for q in range(n):
            for j in range(p):

                D[q, j] = la.norm(outcenter[:, q] - tarcenter[:, j])  # distance matrix
                # A[q, j] = outalpha[q] - taralpha[j]
                #                 print(D[q, j])
                alpha = []
                if outalpha[q] > 0 and taralpha[j] > 0 or outalpha[q] < 0 and taralpha[j] < 0:
                    alpha.append(abs(outalpha[q] - taralpha[j]))
                else:
                    alpha.append(abs(outalpha[q] + taralpha[j]))
                if taralpha[j] < 0:
                    taralpha_r = 180 + taralpha[j]
                else:
                    taralpha_r = -(180 - taralpha[j])
                if outalpha[q] > 0 and taralpha_r > 0 or outalpha[q] < 0 and taralpha_r < 0:
                    alpha.append(abs(outalpha[q] - taralpha_r))
                else:
                    alpha.append(abs(outalpha[q] + taralpha_r))
                A[q, j] = min(alpha)

        for ii in range(p):
            eval_distance.append(D[np.argmin(D, axis=0)[ii]][ii])
            eval_angle.append(A[np.argmin(D, axis=0)[ii]][ii])

    dislist = [0.15, 0.3, 0.45]
    anglist = [5, 15, 25, 360]
    prerec = []
    for dis in dislist:
        for ang in anglist:
            predinrange = sum(
                (np.array(eval_distance) < dis * 60) & (np.array(eval_angle) < ang))  # calc matched predictions
            prednum = outcenterlist
            tarnumraw = tarcenterlist
            print(predinrange, prednum, tarnumraw, '++++', sum(np.array(eval_distance) < dis * 60),
                  sum(np.array(eval_angle) < ang))
            pre = predinrange / prednum if prednum != 0 else 1
            rec = predinrange / tarnumraw if tarnumraw != 0 else 1
            print(' precision: %.6f' % pre, ' racall: %.6f' % rec, ' with dis', dis, 'ang', ang)
            prerec.append([pre, rec, dis, ang])
    return np.array(prerec)


def randomcolor():
    colorArr = ['1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F']
    color = ""
    for i in range(6):
        color += colorArr[random.randint(0, 14)]
    return "#" + color


def plot_pr_array(pr_array, dislist, radlist):
    pr_dict = {}

    for prerec in pr_array:
        for dis in dislist:
            for ang in radlist:
                if str(dis) + '_' + str(ang) not in pr_dict:
                    pr_dict[str(dis) + '_' + str(ang)] = []
                p_r = prerec[np.where(np.logical_and(prerec[:, 2] == dis, prerec[:, 3] == ang) == True)[0], :]
                pr_dict[str(dis) + '_' + str(ang)].append(p_r)

    for dis in dislist:
        for ang in radlist:
            curve = np.array(pr_dict[str(dis) + '_' + str(ang)])
            curve = curve.clip(max=1)
            ap = calcAP(curve) * 100
            plt.plot(curve[:, :, 1], curve[:, :, 0], color=randomcolor(),
                     label=str(dis) + '_' + str('%.2f' % ang) + '_%.1f' % ap)
            plt.xlim((0, 1))
            plt.ylim((0, 1))
    plt.legend()
    plt.xticks([x / 10.0 for x in range(11)])
    plt.yticks([x / 10.0 for x in range(11)])
    plt.grid()
    plt.savefig(str(time()) + '.png')


def calcAP(prcurve):
    recall = prcurve[:, :, 1]
    precision = prcurve[:, :, 0]
    precision = precision[::-1]
    recall = recall[::-1]

    acum_area = 0
    prevpr, prevre = 1, 0
    for pr, re in zip(precision, recall):
        if re[0] > prevre and (not math.isnan(pr[0])) and (not math.isnan(re[0])):
            acum_area += 0.5 * (pr[0] + prevpr) * (re[0] - prevre)
            prevpr = pr[0]
            prevre = re[0]

    return min(acum_area, 1.0)

# thresh_list = [0]
thresh_list = [0, 0.005, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99, 0.999,0.9999, 0.99999, 1]
# thresh_list = [0.6,]
# prerec structure per N thresh (N*12*4):
#
#   precision   recall  dis orien
#                       15  5
#                       15  15
#                       15  25
#                       15  360
#                       30  5
#                       30  15
#                       30  25
#                       30  360
#                       45  5
#                       45  15
#                       45  25
#                       45  360
#

pr_array = []
dislist = [0.15, 0.3, 0.45]
anglist = [5, 15, 25, 360]

for thresh in tqdm(thresh_list):
    prerec = test(thresh)
    pr_array.append(prerec)
    print(prerec)
    print(thresh, '================================================')
np.save('prerec', pr_array)

# plotting pr_array
plot_pr_array(pr_array, dislist, anglist)