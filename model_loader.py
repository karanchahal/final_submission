"""
You need to implement all four functions in this file and also put your team info as a variable
Then you should submit the python file with your model class, the state_dict, and this file
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from final_model import SingleModel, GoodSegModel
from helper import collate_fn, draw_box
import numpy as np
from data_helper import UnlabeledDataset, LabeledDataset
import matplotlib.patches as patches
import matplotlib.pyplot as plt 
import math

##### ANGLES

def convert(point_squence):

    fr_l_x = point_squence[0][0]
    fr_r_x = point_squence[0][1]
    bk_r_x = point_squence[0][2]
    bk_l_x = point_squence[0][3]

    fr_l_y = point_squence[1][0]
    fr_r_y = point_squence[1][1]
    bk_r_y = point_squence[1][2]
    bk_l_y = point_squence[1][3]

    ex1 = min(fr_l_x, fr_r_x, bk_r_x, bk_l_x)
    ex2 = max(fr_l_x, fr_r_x, bk_r_x, bk_l_x)
    ey1 = min(fr_l_y, fr_r_y, bk_r_y, bk_l_y)
    ey2 = max(fr_l_y, fr_r_y, bk_r_y, bk_l_y)

    return torch.tensor([ex1, ey1, ex2, ey2]).view(4)

def angle_util(x, y):
    myradians = math.atan2(y, x)
    mydegrees = math.degrees(myradians)
    return mydegrees

def check(angle1, angle2, angle3, angle4, min, max):
    if angle1 >= min and angle1 <= max:
        if angle2 >= min and angle2 <= max:
            if angle3 >= min and angle3 <= max:
                if angle4 >= min and angle4 <= max:
                    return True
                else:
                    return False
            else:
                return False
        else:
            return False
    else:
        return False

def checkUtil(angle, min, max):
    if angle >= min and angle <= max:
        return True
    else:
        return False



def get_angle_actual(box):
    # print(box)
    min_x = min(box[0], box[2])
    max_x = max(box[0], box[2])
    min_y = min(box[1], box[3])
    max_y = max(box[1], box[3])

    center_x = min_x + 0.5*(max_x - min_x)
    center_y = min_y + 0.5*(max_y - min_y)

    angle = angle_util(center_x, center_y)

    if checkUtil(angle, min=-35, max=35):
        # print('GREEN Angles: {} | Center x: {} | Center y: {} '.format(angle, center_x, center_y))
        return 'green' # front left

    if checkUtil(angle, min=25, max=95):
        # print('RED Angles: {} | Center x: {} | Center y: {} '.format(angle, center_x, center_y))
        return 'red' # front

    if checkUtil(angle, min=85, max=155):
        # print('BLACK Angles: {} | Center x: {} | Center y: {} '.format(angle, center_x, center_y))
        return 'black' # back left

    if checkUtil(angle, min=-95, max=-25):
        # print('BLACK Angles: {} | Center x: {} | Center y: {} '.format(angle, center_x, center_y))
        return 'yellow' # front right

    if checkUtil(angle, min=-155, max=-85):
        # print('BLACK Angles: {} | Center x: {} | Center y: {} '.format(angle, center_x, center_y))
        return 'white' # back right
    
    if checkUtil(angle, min=-180, max=-145):
        return 'blue' # back

    if checkUtil(angle, min=145, max=180):
        return 'blue' # back
    
    return 'white'



def get_angle(box):

    min_x = min(box[0][0], box[0][1], box[0][2], box[0][3])
    max_x = max(box[0][0], box[0][1], box[0][2], box[0][3])
    min_y = min(box[1][0], box[1][1], box[1][2], box[1][3])
    max_y = max(box[1][0], box[1][1], box[1][2], box[1][3])

    center_x = min_x + 0.5*(max_x - min_x)
    center_y = min_y + 0.5*(max_y - min_y)

    angle = angle_util(center_x, center_y)

    if checkUtil(angle, min=-35, max=35):
        # print('GREEN Angles: {} | Center x: {} | Center y: {} '.format(angle, center_x, center_y))
        return 'green' # front left

    if checkUtil(angle, min=25, max=95):
        # print('RED Angles: {} | Center x: {} | Center y: {} '.format(angle, center_x, center_y))
        return 'red' # front

    if checkUtil(angle, min=85, max=155):
        # print('BLACK Angles: {} | Center x: {} | Center y: {} '.format(angle, center_x, center_y))
        return 'black' # back left

    if checkUtil(angle, min=-95, max=-25):
        # print('BLACK Angles: {} | Center x: {} | Center y: {} '.format(angle, center_x, center_y))
        return 'yellow' # front right

    if checkUtil(angle, min=-155, max=-85):
        # print('BLACK Angles: {} | Center x: {} | Center y: {} '.format(angle, center_x, center_y))
        return 'white' # back right
    
    if checkUtil(angle, min=-180, max=-145):
        return 'blue' # back

    if checkUtil(angle, min=145, max=180):
        return 'blue' # back
    
    return 'white'


def get_bbox_split(bboxes):
    split_boxes = [ [] for i in range(6)]
    for box in bboxes:    
        min_x = min(box[0][0], box[0][1], box[0][2], box[0][3])
        max_x = max(box[0][0], box[0][1], box[0][2], box[0][3])
        min_y = min(box[1][0], box[1][1], box[1][2], box[1][3])
        max_y = max(box[1][0], box[1][1], box[1][2], box[1][3])

        center_x = min_x + 0.5*(max_x - min_x)
        center_y = min_y + 0.5*(max_y - min_y)

        angle = angle_util(center_x, center_y)

        if checkUtil(angle, min=-35, max=35):
            # print('GREEN Angles: {} | Center x: {} | Center y: {} '.format(angle, center_x, center_y))
            split_boxes[1].append(box) # front 

        if checkUtil(angle, min=25, max=95):
            # print('RED Angles: {} | Center x: {} | Center y: {} '.format(angle, center_x, center_y))
            split_boxes[0].append(box) # front left

        if checkUtil(angle, min=85, max=155):
            # print('BLACK Angles: {} | Center x: {} | Center y: {} '.format(angle, center_x, center_y))
            split_boxes[3].append(box) # back left

        if checkUtil(angle, min=-95, max=-25):
            # print('BLACK Angles: {} | Center x: {} | Center y: {} '.format(angle, center_x, center_y))
            split_boxes[2].append(box) # front right

        if checkUtil(angle, min=-155, max=-85):
            # print('BLACK Angles: {} | Center x: {} | Center y: {} '.format(angle, center_x, center_y))
            split_boxes[5].append(box) # back right
        
        if checkUtil(angle, min=-180, max=-145):
            split_boxes[4].append(box) # back

        if checkUtil(angle, min=145, max=180):
            split_boxes[4].append(box) # back
        
    return split_boxes



##### Matcher

from torch.jit.annotations import List, Tuple
from torch import Tensor

@torch.jit.script
class Matcher(object):
    """
    This class assigns to each predicted "element" (e.g., a box) a ground-truth
    element. Each predicted element will have exactly zero or one matches; each
    ground-truth element may be assigned to zero or more predicted elements.
    Matching is based on the MxN match_quality_matrix, that characterizes how well
    each (ground-truth, predicted)-pair match. For example, if the elements are
    boxes, the matrix may contain box IoU overlap values.
    The matcher returns a tensor of size N containing the index of the ground-truth
    element m that matches to prediction n. If there is no match, a negative value
    is returned.
    """

    BELOW_LOW_THRESHOLD = -1
    BETWEEN_THRESHOLDS = -2

    __annotations__ = {
        'BELOW_LOW_THRESHOLD': int,
        'BETWEEN_THRESHOLDS': int,
    }

    def __init__(self, high_threshold=0.7, low_threshold=0.3, allow_low_quality_matches=False):
        # type: (float, float, bool)
        """
        Args:
            high_threshold (float): quality values greater than or equal to
                this value are candidate matches.
            low_threshold (float): a lower quality threshold used to stratify
                matches into three levels:
                1) matches >= high_threshold
                2) BETWEEN_THRESHOLDS matches in [low_threshold, high_threshold)
                3) BELOW_LOW_THRESHOLD matches in [0, low_threshold)
            allow_low_quality_matches (bool): if True, produce additional matches
                for predictions that have only low-quality match candidates. See
                set_low_quality_matches_ for more details.
        """
        self.BELOW_LOW_THRESHOLD = -1
        self.BETWEEN_THRESHOLDS = -2
        assert low_threshold <= high_threshold
        self.high_threshold = high_threshold
        self.low_threshold = low_threshold
        self.allow_low_quality_matches = allow_low_quality_matches

    def __call__(self, match_quality_matrix):
        """
        Args:
            match_quality_matrix (Tensor[float]): an MxN tensor, containing the
            pairwise quality between M ground-truth elements and N predicted elements.
        Returns:
            matches (Tensor[int64]): an N tensor where N[i] is a matched gt in
            [0, M - 1] or a negative value indicating that prediction i could not
            be matched.
        """
        if match_quality_matrix.numel() == 0:
            # empty targets or proposals not supported during training
            if match_quality_matrix.shape[0] == 0:
                raise ValueError(
                    "No ground-truth boxes available for one of the images "
                    "during training")
            else:
                raise ValueError(
                    "No proposal boxes available for one of the images "
                    "during training")

        # match_quality_matrix is M (gt) x N (predicted)
        # Max over gt elements (dim 0) to find best gt candidate for each prediction
        matched_vals, matches = match_quality_matrix.max(dim=0)
        if self.allow_low_quality_matches:
            all_matches = matches.clone()
        else:
            all_matches = None

        # Assign candidate matches with low quality to negative (unassigned) values
        below_low_threshold = matched_vals < self.low_threshold
        between_thresholds = (matched_vals >= self.low_threshold) & (
            matched_vals < self.high_threshold
        )
        matches[below_low_threshold] = torch.tensor(self.BELOW_LOW_THRESHOLD)
        matches[between_thresholds] = torch.tensor(self.BETWEEN_THRESHOLDS)

        if self.allow_low_quality_matches:
            assert all_matches is not None
            self.set_low_quality_matches_(matches, all_matches, match_quality_matrix)

        return matches

    def set_low_quality_matches_(self, matches, all_matches, match_quality_matrix):
        """
        Produce additional matches for predictions that have only low-quality matches.
        Specifically, for each ground-truth find the set of predictions that have
        maximum overlap with it (including ties); for each prediction in that set, if
        it is unmatched, then match it to the ground-truth with which it has the highest
        quality value.
        """
        # For each gt, find the prediction with which it has highest quality
        highest_quality_foreach_gt, _ = match_quality_matrix.max(dim=1)
        # Find highest quality match available, even if it is low, including ties
        gt_pred_pairs_of_highest_quality = torch.nonzero(
            match_quality_matrix == highest_quality_foreach_gt[:, None]
        )
        # Example gt_pred_pairs_of_highest_quality:
        #   tensor([[    0, 39796],
        #           [    1, 32055],
        #           [    1, 32070],
        #           [    2, 39190],
        #           [    2, 40255],
        #           [    3, 40390],
        #           [    3, 41455],
        #           [    4, 45470],
        #           [    5, 45325],
        #           [    5, 46390]])
        # Each row is a (gt index, prediction index)
        # Note how gt items 1, 2, 3, and 5 each have two ties

        pred_inds_to_update = gt_pred_pairs_of_highest_quality[:, 1]
        matches[pred_inds_to_update] = all_matches[pred_inds_to_update]


#####



##### Box IOU

import torch
from torch.jit.annotations import Tuple
from torch import Tensor
import torchvision


def nms(boxes, scores, iou_threshold):
    # type: (Tensor, Tensor, float)
    """
    Performs non-maximum suppression (NMS) on the boxes according
    to their intersection-over-union (IoU).
    NMS iteratively removes lower scoring boxes which have an
    IoU greater than iou_threshold with another (higher scoring)
    box.
    Parameters
    ----------
    boxes : Tensor[N, 4])
        boxes to perform NMS on. They
        are expected to be in (x1, y1, x2, y2) format
    scores : Tensor[N]
        scores for each one of the boxes
    iou_threshold : float
        discards all overlapping
        boxes with IoU > iou_threshold
    Returns
    -------
    keep : Tensor
        int64 tensor with the indices
        of the elements that have been kept
        by NMS, sorted in decreasing order of scores
    """
    return torch.ops.torchvision.nms(boxes, scores, iou_threshold)


def batched_nms(boxes, scores, idxs, iou_threshold):
    # type: (Tensor, Tensor, Tensor, float)
    """
    Performs non-maximum suppression in a batched fashion.
    Each index value correspond to a category, and NMS
    will not be applied between elements of different categories.
    Parameters
    ----------
    boxes : Tensor[N, 4]
        boxes where NMS will be performed. They
        are expected to be in (x1, y1, x2, y2) format
    scores : Tensor[N]
        scores for each one of the boxes
    idxs : Tensor[N]
        indices of the categories for each one of the boxes.
    iou_threshold : float
        discards all overlapping boxes
        with IoU > iou_threshold
    Returns
    -------
    keep : Tensor
        int64 tensor with the indices of
        the elements that have been kept by NMS, sorted
        in decreasing order of scores
    """
    if boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.int64, device=boxes.device)
    # strategy: in order to perform NMS independently per class.
    # we add an offset to all the boxes. The offset is dependent
    # only on the class idx, and is large enough so that boxes
    # from different classes do not overlap
    max_coordinate = boxes.max()
    offsets = idxs.to(boxes) * (max_coordinate + 1)
    boxes_for_nms = boxes + offsets[:, None]
    keep = nms(boxes_for_nms, scores, iou_threshold)
    return keep


def remove_small_boxes(boxes, min_size):
    # type: (Tensor, float)
    """
    Remove boxes which contains at least one side smaller than min_size.
    Arguments:
        boxes (Tensor[N, 4]): boxes in (x1, y1, x2, y2) format
        min_size (float): minimum size
    Returns:
        keep (Tensor[K]): indices of the boxes that have both sides
            larger than min_size
    """
    ws, hs = boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1]
    keep = (ws >= min_size) & (hs >= min_size)
    keep = keep.nonzero().squeeze(1)
    return keep


def clip_boxes_to_image(boxes, size):
    # type: (Tensor, Tuple[int, int])
    """
    Clip boxes so that they lie inside an image of size `size`.
    Arguments:
        boxes (Tensor[N, 4]): boxes in (x1, y1, x2, y2) format
        size (Tuple[height, width]): size of the image
    Returns:
        clipped_boxes (Tensor[N, 4])
    """
    dim = boxes.dim()
    boxes_x = boxes[..., 0::2]
    boxes_y = boxes[..., 1::2]
    height, width = size

    if torchvision._is_tracing():
        boxes_x = torch.max(boxes_x, torch.tensor(0, dtype=boxes.dtype, device=boxes.device))
        boxes_x = torch.min(boxes_x, torch.tensor(width, dtype=boxes.dtype, device=boxes.device))
        boxes_y = torch.max(boxes_y, torch.tensor(0, dtype=boxes.dtype, device=boxes.device))
        boxes_y = torch.min(boxes_y, torch.tensor(height, dtype=boxes.dtype, device=boxes.device))
    else:
        boxes_x = boxes_x.clamp(min=0, max=width)
        boxes_y = boxes_y.clamp(min=0, max=height)

    clipped_boxes = torch.stack((boxes_x, boxes_y), dim=dim)
    return clipped_boxes.reshape(boxes.shape)


def box_area(boxes):
    """
    Computes the area of a set of bounding boxes, which are specified by its
    (x1, y1, x2, y2) coordinates.
    Arguments:
        boxes (Tensor[N, 4]): boxes for which the area will be computed. They
            are expected to be in (x1, y1, x2, y2) format
    Returns:
        area (Tensor[N]): area for each box
    """
    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])


# implementation from https://github.com/kuangliu/torchcv/blob/master/torchcv/utils/box.py
# with slight modifications
def box_iou(boxes1, boxes2):
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Arguments:
        boxes1 (Tensor[N, 4])
        boxes2 (Tensor[M, 4])
    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    iou = inter / (area1[:, None] + area2 - inter).type(torch.float32)
    return iou

def test():
    a = torch.tensor([319.73915, 674.9350, 272.4419, 656.2753]).view(1,-1).type(torch.float64)
    b = torch.tensor([319.7395, 674.9350, 272.4419, 656.2753]).view(1,-1).type(torch.float64)
    print(box_iou(a, b)) 




#### ANCHORS


def get_offsets(gt_boxes, ex_boxes):

    ex_width = ex_boxes[:, 2] - ex_boxes[:, 0]
    ex_height = ex_boxes[:, 3] - ex_boxes[:, 1]
    ex_center_x = ex_boxes[:, 0] + 0.5*ex_width
    ex_center_y = ex_boxes[:, 1] + 0.5*ex_height

    gt_width = gt_boxes[:, 2] - gt_boxes[:, 0]
    gt_height = gt_boxes[:, 3] - gt_boxes[:, 1]
    gt_center_x = gt_boxes[:, 0] + 0.5*gt_width
    gt_center_y = gt_boxes[:, 1] + 0.5*gt_height


    delta_x = (gt_center_x - ex_center_x) / ex_width
    delta_y = (gt_center_y - ex_center_y) / ex_height
    delta_scaleX = torch.log(gt_width / ex_width)
    delta_scaleY = torch.log(gt_height / ex_height)

    offsets = torch.cat([delta_x.unsqueeze(0), 
                    delta_y.unsqueeze(0),
                    delta_scaleX.unsqueeze(0),
                    delta_scaleY.unsqueeze(0)],
                dim=0)
    return offsets.permute(1,0)

def plotMap(x, y):
    plt.ylim(800, 0)  # decreasing time
    plt.plot(x.numpy(), y.numpy(), 'o', color='black')
    plt.show()
 
def get_bbox_gt(bboxes1, gt_boxes, sz, device):
    bboxes = bboxes1.clone()
    ex_boxes = []
    for box in bboxes:
        ex_box = convert_box_format(box)
        ex_boxes.append(ex_box)
    ex_boxes = torch.stack(ex_boxes)
    high_threshold = 0.1
    low_threshold = 0.01

    ex_width = ex_boxes[:, 2] - ex_boxes[:, 0]
    ex_height = ex_boxes[:, 3] - ex_boxes[:, 1]
    ex_center_x = ex_boxes[:, 0] + 0.5*ex_width
    ex_center_y = ex_boxes[:, 1] + 0.5*ex_height

    gt_widths = gt_boxes[:, 2] - gt_boxes[:, 0]
    gt_heights = gt_boxes[:, 3] - gt_boxes[:, 1]
    gt_center_x = gt_boxes[:, 0] + 0.5*gt_widths
    gt_center_y = gt_boxes[:, 1] + 0.5*gt_heights

    ious = box_iou(gt_boxes, ex_boxes)
    ious2 = box_iou(ex_boxes, gt_boxes)
    vals, inds = torch.max(ious, dim=1)

    vals2, inds2 = torch.max(ious2, dim=1)
    shape_i = gt_boxes.shape[0]
    gt_classes = torch.zeros((shape_i)).type(torch.long).to(device)
    gt_offsets = torch.zeros((shape_i, 4)).type(torch.float32).to(device)

    gt_classes[vals > high_threshold] = 1
    gt_classes[vals < low_threshold] = 0 # background anchors
    gt_classes[(vals >= low_threshold) & (vals < high_threshold)] = -1 # anchors to ignore
    k = 0
    for box, val, ind in zip(ex_boxes, vals2, inds2):
        
        if val.item() < high_threshold:
            k += 1
            gt_classes[ind] = 1
            offset = get_offsets(gt_boxes[ind].unsqueeze(0), box.unsqueeze(0)).squeeze(0)
            gt_offsets[ind] = offset

    actual_boxes = ex_boxes[inds[vals > high_threshold]]
    ref_boxes = gt_boxes[vals > high_threshold]
    g_offsets = get_offsets(ref_boxes, actual_boxes)
    gt_offsets[vals > high_threshold] = g_offsets

    return gt_classes, gt_offsets


def get_bbox_gt_actual(ex_boxes, gt_boxes, sz, device):
    high_threshold = 0.7
    low_threshold = 0.3

    ex_width = ex_boxes[:, 2] - ex_boxes[:, 0]
    ex_height = ex_boxes[:, 3] - ex_boxes[:, 1]
    ex_center_x = ex_boxes[:, 0] + 0.5*ex_width
    ex_center_y = ex_boxes[:, 1] + 0.5*ex_height

    gt_widths = gt_boxes[:, 2] - gt_boxes[:, 0]
    gt_heights = gt_boxes[:, 3] - gt_boxes[:, 1]
    gt_center_x = gt_boxes[:, 0] + 0.5*gt_widths
    gt_center_y = gt_boxes[:, 1] + 0.5*gt_heights

    ious = box_iou(gt_boxes, ex_boxes)
    ious2 = box_iou(ex_boxes, gt_boxes)
    vals, inds = torch.max(ious, dim=1)

    vals2, inds2 = torch.max(ious2, dim=1)
    shape_i = gt_boxes.shape[0]
    gt_classes = torch.zeros((shape_i)).type(torch.long).to(device)
    gt_offsets = torch.zeros((shape_i, 4)).type(torch.float32).to(device)

    gt_classes[vals > high_threshold] = 1
    gt_classes[vals < low_threshold] = 0 # background anchors
    gt_classes[(vals >= low_threshold) & (vals < high_threshold)] = -1 # anchors to ignore
    k = 0

    for box, val, ind in zip(ex_boxes, vals2, inds2):
        if val.item() < high_threshold:
            k += 1
            gt_classes[ind] = 1
            offset = get_offsets(gt_boxes[ind].unsqueeze(0), box.unsqueeze(0)).squeeze(0)
            gt_offsets[ind] = offset
            
    actual_boxes = ex_boxes[inds[vals > high_threshold]]
    ref_boxes = gt_boxes[vals > high_threshold]
    g_offsets = get_offsets(ref_boxes, actual_boxes)
    gt_offsets[vals > high_threshold] = g_offsets

    return gt_classes, gt_offsets



def modify(gt_boxes, gt_offsets, sz=800, vis=True, device="cuda"):
    delta_x = gt_offsets[:,0]
    delta_y = gt_offsets[:,1]
    delta_scaleX = gt_offsets[:,2]
    delta_scaleY = gt_offsets[:,3]
    gt_widths = gt_boxes[:, 2] - gt_boxes[:, 0]
    gt_heights = gt_boxes[:, 3] - gt_boxes[:, 1]
    gt_center_x = gt_boxes[:, 0] + 0.5 * gt_widths
    gt_center_y = gt_boxes[:, 1] + 0.5 * gt_heights

    ex_width = gt_widths / torch.exp(delta_scaleX)
    ex_height = gt_heights / torch.exp(delta_scaleY)
    ex_center_x = gt_center_x - delta_x*ex_width
    ex_center_y = gt_center_y - delta_y*ex_height

    ex1 = ex_center_x - 0.5*ex_width
    ex2 = ex_center_x + 0.5*ex_width
    ey1 = ex_center_y - 0.5*ex_height
    ey2 = ex_center_y + 0.5*ex_height

    pred_boxes = torch.cat([ex1.unsqueeze(0), ey1.unsqueeze(0), ex2.unsqueeze(0), ey2.unsqueeze(0)], dim=0).permute(1,0)
    inds = nms(pred_boxes, torch.ones((pred_boxes.size(0))).to(device), 1.0)
    pred_boxes = pred_boxes[inds]
    if vis == True:
        fig,ax = plt.subplots(1)
        a = torch.zeros(sz,sz)
        ax.imshow(a)
        for box in pred_boxes:
            x1, y1, x2, y2 = box[0], box[1], box[2], box[3]
            rect = patches.Rectangle((x1,y1),abs(x1 - x2),abs(y1 - y2),linewidth=1,edgecolor='r',facecolor='none')
            ax.add_patch(rect)
        plt.show()
    return pred_boxes

def visualizeFast(gt_boxes, gt_offsets, gt_classes, sz=800):
    inds = (gt_classes > 0)
    threshold = 0.7
    gt_boxes = gt_boxes[inds]
    gt_offsets = gt_offsets[inds]
    gt_classes = gt_classes[inds]

    # print('Number of g boxes: ', gt_boxes.size())
    if gt_boxes.shape[0] == 0:
        print('No gt boxes matched !')
        return None, None

    delta_x = gt_offsets[:,0]
    delta_y = gt_offsets[:,1]
    delta_scaleX = gt_offsets[:,2]
    delta_scaleY = gt_offsets[:,3]
    gt_widths = gt_boxes[:, 2] - gt_boxes[:, 0]
    gt_heights = gt_boxes[:, 3] - gt_boxes[:, 1]
    gt_center_x = gt_boxes[:, 0] + 0.5 * gt_widths
    gt_center_y = gt_boxes[:, 1] + 0.5 * gt_heights

    ex_width = gt_widths / torch.exp(delta_scaleX)
    ex_height = gt_heights / torch.exp(delta_scaleY)
    ex_center_x = gt_center_x - delta_x*ex_width
    ex_center_y = gt_center_y - delta_y*ex_height

    ex1 = ex_center_x - 0.5*ex_width
    ex2 = ex_center_x + 0.5*ex_width
    ey1 = ex_center_y - 0.5*ex_height
    ey2 = ex_center_y + 0.5*ex_height

    pred_boxes = torch.cat([ex1.unsqueeze(0), ey1.unsqueeze(0), ex2.unsqueeze(0), ey2.unsqueeze(0)], dim=0).permute(1,0)
    pred_boxes = pred_boxes.type(torch.float32)
    gt_classes = gt_classes.type(torch.float32)

    fig,ax = plt.subplots(1)
    a = torch.zeros(sz,sz)
    ax.imshow(a)
    for box in pred_boxes:
        x1, y1, x2, y2 = box[0], box[1], box[2], box[3]
        rect = patches.Rectangle((x1,y1),abs(x1 - x2),abs(y1 - y2),linewidth=1,edgecolor='r',facecolor='none')
        ax.add_patch(rect)

    return fig, ax
    # plt.show()

def convert_box_format(box):

    point_squence = torch.stack([box[:, 0], box[:, 1], box[:, 3], box[:, 2], box[:, 0]]).T
    point_squence[0] *= 10
    point_squence[0] += 400

    point_squence[1] = -point_squence[1] * 10  + 400

    fr_l_x = point_squence[0][0]
    fr_r_x = point_squence[0][1]
    bk_r_x = point_squence[0][2]
    bk_l_x = point_squence[0][3]

    fr_l_y = point_squence[1][0]
    fr_r_y = point_squence[1][1]
    bk_r_y = point_squence[1][2]
    bk_l_y = point_squence[1][3]

    ex1 = min(fr_l_x, fr_r_x, bk_r_x, bk_l_x)
    ex2 = max(fr_l_x, fr_r_x, bk_r_x, bk_l_x)
    ey1 = min(fr_l_y, fr_r_y, bk_r_y, bk_l_y)
    ey2 = max(fr_l_y, fr_r_y, bk_r_y, bk_l_y)

    return torch.tensor([ex1, ey1, ex2, ey2]).view(4)

def convert_box_format_actual(box):

    point_squence = torch.stack([box[:, 0], box[:, 1], box[:, 3], box[:, 2], box[:, 0]]).T
    point_squence[0] *= 1

    point_squence[1] = -point_squence[1] * 1

    fr_l_x = point_squence[0][0]
    fr_r_x = point_squence[0][1]
    bk_r_x = point_squence[0][2]
    bk_l_x = point_squence[0][3]

    fr_l_y = point_squence[1][0]
    fr_r_y = point_squence[1][1]
    bk_r_y = point_squence[1][2]
    bk_l_y = point_squence[1][3]

    ex1 = min(fr_l_x, fr_r_x, bk_r_x, bk_l_x)
    ex2 = max(fr_l_x, fr_r_x, bk_r_x, bk_l_x)
    ey1 = min(fr_l_y, fr_r_y, bk_r_y, bk_l_y)
    ey2 = max(fr_l_y, fr_r_y, bk_r_y, bk_l_y)

    return torch.tensor([ex1, ey1, ex2, ey2]).view(4)


def convert_x(x):
    return 10*x + 400

def convert_y(y):
    return 10*-y + 400

def plotAngleLines(ax, x, x_new, y, angle, color):
    y_new_1 = x_new*math.tan(math.radians(angle))
    ax.plot( [convert_x(x), convert_x(x_new)], [convert_y(y), convert_y(y_new_1)], color=color)

def plotAngleLinesNoConvert(ax, x, x_new, y, angle, color):
    y_new_1 = x_new*math.tan(math.radians(angle))
    ax.plot( [x, x_new], [y, y_new_1], color=color)



def plotDefault(ax, bboxes):
    a = torch.zeros(800,800)
    ax.imshow(a)
    for box in bboxes:
        draw_box(ax, box, color="white")

def visActual(cornerz, road=None):
    
    if road:
        fig, ((ax, ax2, ax3)) = plt.subplots(1, 3)
    else:
        fig, ((ax, ax2)) = plt.subplots(1, 2)

    a = torch.zeros(800,800)

    # plot with given visualisation code to validate our calculations
    plotDefault(ax2, cornerz)
    ax.imshow(a)
    if road:
        ax3.imshow(road)

    # plot angles
    x = 0
    y = 0
    plotAngleLines(ax, x=x, x_new=10, y=y, angle=35, color='green')
    plotAngleLines(ax, x=x, x_new=10, y=y, angle=-35, color='green')
    plotAngleLines(ax, x=x, x_new=10, y=y, angle=25, color='red')
    plotAngleLines(ax, x=x, x_new=-2, y=y, angle=95, color='red')
    plotAngleLines(ax, x=x, x_new=10, y=y, angle=-25, color='yellow')
    plotAngleLines(ax, x=x, x_new=-2, y=y, angle=-95, color='yellow')
    plotAngleLines(ax, x=x, x_new=2, y=y, angle=85, color='black')
    plotAngleLines(ax, x=x, x_new=-10, y=y, angle=155, color='black')
    plotAngleLines(ax, x=x, x_new=-10, y=y, angle=145, color='blue')
    plotAngleLines(ax, x=x, x_new=-10, y=y, angle=-145, color='blue')
    plotAngleLines(ax, x=x, x_new=-10, y=y, angle=-155, color='white')
    plotAngleLines(ax, x=x, x_new=2, y=y, angle=-85, color='white')

    # plot the boxes

    for box in cornerz:
        color = get_angle(box)
        box = box.clone()

        point_squence = torch.stack([box[:, 0], box[:, 1], box[:, 3], box[:, 2], box[:, 0]]).T
        point_squence[0] *= 10
        point_squence[0] += 400

        point_squence[1] = -point_squence[1] * 10  + 400

        fr_l_x = point_squence[0][0]
        fr_r_x = point_squence[0][1]
        bk_r_x = point_squence[0][2]
        bk_l_x = point_squence[0][3]

        fr_l_y = point_squence[1][0]
        fr_r_y = point_squence[1][1]
        bk_r_y = point_squence[1][2]
        bk_l_y = point_squence[1][3]

        ex1 = min(fr_l_x, fr_r_x, bk_r_x, bk_l_x)
        ex2 = max(fr_l_x, fr_r_x, bk_r_x, bk_l_x)
        ey1 = min(fr_l_y, fr_r_y, bk_r_y, bk_l_y)
        ey2 = max(fr_l_y, fr_r_y, bk_r_y, bk_l_y)

        rect = patches.Rectangle((ex1,ey1),abs(ex1 - ex2),abs(ey1 - ey2),linewidth=2, edgecolor=color, fill=False)
        ax.add_patch(rect)

    return fig, ((ax, ax2))


def visActualReal(cornerz):
    

    fig, ax = plt.subplots(1)

    a = torch.zeros(800,800)

    # plot with given visualisation code to validate our calculations
    ax.imshow(a)
    
    # plot angles
    x = 0
    y = 0
    plotAngleLines(ax, x=x, x_new=10, y=y, angle=35, color='green')
    plotAngleLines(ax, x=x, x_new=10, y=y, angle=-35, color='green')
    plotAngleLines(ax, x=x, x_new=10, y=y, angle=25, color='red')
    plotAngleLines(ax, x=x, x_new=-2, y=y, angle=95, color='red')
    plotAngleLines(ax, x=x, x_new=10, y=y, angle=-25, color='yellow')
    plotAngleLines(ax, x=x, x_new=-2, y=y, angle=-95, color='yellow')
    plotAngleLines(ax, x=x, x_new=2, y=y, angle=85, color='black')
    plotAngleLines(ax, x=x, x_new=-10, y=y, angle=155, color='black')
    plotAngleLines(ax, x=x, x_new=-10, y=y, angle=145, color='blue')
    plotAngleLines(ax, x=x, x_new=-10, y=y, angle=-145, color='blue')
    plotAngleLines(ax, x=x, x_new=-10, y=y, angle=-155, color='white')
    plotAngleLines(ax, x=x, x_new=2, y=y, angle=-85, color='white')

    # plot the boxes

    for box in cornerz:
        # print(len(box))
        color = get_angle_actual(box)
        box = box.clone()

        ex1 = min(box[0], box[2])*10 + 400
        ex2 = max(box[0], box[2])*10 + 400
        ey1 = -min(box[1], box[3])*10 + 400
        ey2 = -max(box[1], box[3])*10 + 400

        rect = patches.Rectangle((ex1,ey1),abs(ex1 - ex2),abs(ey1 - ey2),linewidth=2, edgecolor=color, fill=False)
        ax.add_patch(rect)

    return fig, ax


def convertBack(boxes):
    final_boxes = []
    for box in boxes:
        ex1 = box[0]
        ex2 = box[2]
        ey1 = box[1]
        ey2 = box[3]
        conv_box = torch.tensor([[ex1, ex1, ex2, ex2], [ey1, ey2, ey1, ey2]]).view(2,4)
        final_boxes.append(conv_box)
    if len(final_boxes) == 0:
        return []
    return torch.stack(final_boxes)


def visGT(bboxes):
    fig,ax = plt.subplots(1)
    a = torch.zeros(800,800)
    ax.imshow(a)
    for box in bboxes:
        box = box.clone()
        ex1 = box[0]
        ey1 = box[1]
        ex2 = box[2]
        ey2 = box[3]
        if ey1 == 680:
            rect = patches.Rectangle((ex1,ey1),abs(ex1 - ex2),abs(ey1 - ey2),linewidth=1,edgecolor='r',facecolor='none')
            ax.add_patch(rect)
    
    plt.show()


def draw_box_util(ax, box, fill=False):
    ex1 = box[0]
    ey1 = box[1]
    ex2 = box[2]
    ey2 = box[3]
    rect = patches.Rectangle((ex1,ey1),abs(ex1 - ex2),abs(ey1 - ey2),linewidth=1, edgecolor='r',fill=fill, facecolor='r')
    ax.add_patch(rect)


def visHeatMap(gt_boxes, gt_classes):
    fig,ax = plt.subplots(1)
    a = torch.zeros(40,40)
    ax.imshow(a)

    foreground = gt_classes[gt_classes > 0]
    foreground_boxes = gt_boxes[gt_classes > 0]

    for box in foreground_boxes:
        draw_box_util(ax, box, fill=True)
    
    plt.show()

def draw_box_actual(ax, box):
    # box = convert_box_format_actual(box)
    ex1 = box[0]
    ey1 = box[1]
    ex2 = box[2]
    ey2 = box[3]

    rect = patches.Rectangle((ex1,ey1),abs(ex1 - ex2),abs(ey1 - ey2),linewidth=1,edgecolor='r',facecolor='none')
    ax.add_patch(rect)

def translate(box, angle, trans=True):
    if trans == True:
        box = convert_box_format_actual(box)
    ex1 = box[0]
    ey1 = box[1]
    ex2 = box[2]
    ey2 = box[3]
    theta = math.radians(angle)
    new_ex1 = 0+(ex1-0)*math.cos(theta)+(ey1-0)*math.sin(theta)
    new_ey1 = 0-(ex1-0)*math.sin(theta)+(ey1-0)*math.cos(theta)

    new_ex2 = 0+(ex2-0)*math.cos(theta)+(ey2-0)*math.sin(theta)
    new_ey2 = 0-(ex2-0)*math.sin(theta)+(ey2-0)*math.cos(theta)

    ex1 = min(new_ex1, new_ex2)
    ex2 = max(new_ex1, new_ex2)

    ey1 = min(new_ey1, new_ey2)
    ey2 = max(new_ey1, new_ey2)

    new_box = torch.tensor([ex1, ey1, ex2, ey2]).view(4)

    return new_box

def visSepBoxes(split_boxes, fig1, axs1, vis=False, trans=True):
    if vis:
        fig, (axs) = plt.subplots(2,3)
        for j in range(2):
            for i in range(3):
                plotAngleLinesNoConvert(axs[j,i], x=0, x_new=40, y=0, angle=0, color='green')
                plotAngleLinesNoConvert(axs[j,i], x=0, x_new=40, y=0, angle=70, color='green')

    new_boxes = [ [], [], [], [], [], [] ]

    for i, boxes in enumerate(split_boxes):
        if i == 1:
            angle = -35
        elif i == 0:
            angle = -( 25 + 70)
        elif i == 2:
            angle = 25
        elif i == 3:
            angle = - (25 + 60 + 70)
        elif i == 4:
            angle = - ( 180 + 35)
        elif i == 5:
            angle = 85

        for box in boxes:
            # translate boxes to go -35% in angle
            n_box = translate(box, angle, trans)
            new_boxes[i].append(n_box)

        if len(new_boxes[i]) != 0:
            new_boxes[i] = torch.stack(new_boxes[i], dim=0)
        
    # Visualization code
    if vis:
        for i, s in enumerate(new_boxes):
            a = torch.zeros((40,40))

            # get ids for axs, not imp
            if i < 3:
                j = 0
            else:
                j = 1
            i = i % 3

            axs[j, i].imshow(a)
            for box in s:
                draw_box_actual(axs[j, i], box)
        if vis:
            plt.show()
    
        return new_boxes, fig, axs
    
    return new_boxes, None, None



def reverse_boxes(boxes, k):
   
    new_boxes = []

    if k == 1:
        angle = +35
    elif k == 0:
        angle = +( 25 + 70)
    elif k == 2:
        angle = -25
    elif k == 3:
        angle = + (25 + 60 + 70)
    elif k == 4:
        angle = + ( 180 + 35)
    elif k == 5:
        angle = -85

    for box in boxes:
        # translate boxes to go -35% in angle
        n_box = translate(box, angle, trans=False)
        new_boxes.append(n_box)

    return new_boxes


####

# Put your transform function here, we will use it for our dataloader
def get_transform(): 
    return torchvision.transforms.ToTensor()
def get_transform_task1(): 
    return torchvision.transforms.ToTensor()
# For road map task
def get_transform_task2(): 
    return torchvision.transforms.ToTensor()

class ModelLoader():
    # Fill the information for your team
    team_name = 'team_name'
    team_number = 1
    round_number = 1
    team_member = ['Karanbir Singh Chahal', 'Soham Tamba', 'Natalie Frank']
    contact_email = 'karanchahal@nyu.edu'

    def __init__(self, model_file='final_model.pth'):
        self.device = "cuda"
        obj = torch.load(model_file, map_location=self.device)
        self.bbox_model = SingleModel().cuda()
        self.bbox_model.load_state_dict(obj["obj"])
        self.seg_model = GoodSegModel().cuda()
        self.seg_model.load_state_dict(obj["seg"])
        self.gt_boxes = self.get_gt_boxes()
        
        # call cuda
        # pass
    
    def get_gt_boxes(self):
        stride = 1
        scales = torch.tensor([1,2,4])
        ratios = torch.tensor([[1,2], [2,1], [1,1], [4,1], [1,4]]).view(5,2)
        ref_boxes = []
        self.map_sz = 40
        for x in range(0, self.map_sz, stride):
            for y in range(0, self.map_sz, stride):
                for scale in scales:
                    for r in ratios:
                        x_r = r[0]
                        y_r = r[1]
                        x_b = x + scale*x_r
                        y_b = y + scale*y_r
                        box = torch.tensor([x, y, x_b, y_b]).view(4)
                        ref_boxes.append(box)
        
        gt_boxes = torch.stack(ref_boxes).view(-1,4).type(torch.float32).to(self.device)
        return gt_boxes

    def get_bounding_boxes(self, samples):
        # samples is a cuda tensor with size [batch_size, 6, 3, 256, 306]
        # You need to return a tuple with size 'batch_size' and each element is a cuda tensor [N, 2, 4]
        # where N is the number of object
        final_out_boxes = []
        for sample in samples:
            yo_boxes = []
            for i, im in enumerate(sample):
                out_pred, out_bbox = self.bbox_model(im.unsqueeze(0))
                out_pred = torch.sigmoid(out_pred)[0]

                out_bbox = out_bbox[0]
                out_pred[out_pred < 0.5] = 0
                out_pred[out_pred != 0] = 1

                out_bbox = out_bbox[out_pred == 1]
                gt_bbox = self.gt_boxes[out_pred == 1].clone()

                final_boxes = modify(gt_bbox, out_bbox, sz=40, vis=False)
                new_boxes = reverse_boxes(final_boxes, i)
                new_boxes = convertBack(new_boxes)
                for box in new_boxes:
                    yo_boxes.append(box)
            out_boxes = torch.stack(yo_boxes)
            final_out_boxes.append(out_boxes)

        return final_out_boxes

    def get_binary_road_map(self, samples):
        # samples is a cuda tensor with size [batch_size, 6, 3, 256, 306]
        # You need to return a cuda tensor with size [batch_size, 800, 800] 
        out = self.seg_model(samples).squeeze(1)
        seg_masks = torch.argmax(out, dim=1)
        return seg_masks


def test():
    image_folder = '../dl/data'
    annotation_csv = '../dl/data/annotation.csv'
    labeled_scene_index = np.arange(106, 128)
    labeled_scene_index_test = np.arange(130, 131)
    transform = get_transform()
    labeled_trainset = LabeledDataset(image_folder=image_folder,
                                    annotation_file=annotation_csv,
                                    scene_index=labeled_scene_index,
                                    transform=transform,
                                    extra_info=True
                                    )

    trainloader = torch.utils.data.DataLoader(labeled_trainset, batch_size=2, shuffle=True, num_workers=0, collate_fn=collate_fn)
    
    sample, target, road_image, extra = iter(trainloader).next()
    loader = ModelLoader()

#test()
