from __future__ import division
import numpy as np
from numpy import array
from ..local_imutils import (
    connected_component_analysis_sk_with_labels as cca_sk,)

from ..bboxs_tools.bbox_operations import (get_area_bounding_box,
                                           get_x_y_start_stop,
                                           )
from ...generic_utils import indexing_with_bounding_2d, sorted_index
from .common_sg import (filter_out_big_and_small_index, get_merged_edges,
                        filter_out_childs, filter_out_small_width_height_index,
                        merge_intersected)
from ..local_imutils import (is_horizontal_line, is_vertical_line)

import cv2


def TreeNode(object):
    def __init__(self, parent, bbox):
        self.parent = parent
        self.bbox = bbox
        self.childs = []


def get_median_background(gray, bbox):
    x_start, x_stop, y_start, y_stop = get_x_y_start_stop(bbox)
    indices = array(
        [[y_start-1, x_start-1], [y_start, x_start-1],
         [y_start-1, x_start], [y_start-1, x_stop+1],
         [y_start-1, x_stop], [y_start, x_stop+1],
         [y_stop+1, x_start-1], [y_stop, x_start-1],
         [y_stop+1, x_start], [y_stop+1, x_stop+1],
         [y_stop+1, x_stop], [y_stop, x_stop+1]])
    array_val = indexing_with_bounding_2d(gray, indices)
    return np.median(array_val)


def binarize_sg(colored_image_array, thres1=50, thres2=200,
                upper_child_bound=5, input_binarized=False, debug=False):
    if len(colored_image_array.shape) > 2:
        gray = cv2.cvtColor(colored_image_array, cv2.COLOR_BGR2GRAY)
    else:
        gray = colored_image_array
    img_shape = gray.shape

    edges = get_merged_edges(colored_image_array, thres1, thres2, debug)

    labels, bboxs = cca_sk(edges)
    areas = [get_area_bounding_box(bbox) for bbox in bboxs]
    sorted_index_text_only = sorted_index(areas, inverse=True)
    sorted_index_text_only = filter_out_big_and_small_index(
        sorted_index_text_only, areas, 15, np.prod(img_shape)/5, debug)
    sorted_index_text_only = filter_out_small_width_height_index(
        sorted_index_text_only, bboxs, 3, 4, debug)
    # Filter
    sorted_index_text_only_final = filter_out_childs(
        bboxs, sorted_index_text_only, upper_child_bound, debug)
    sorted_index_text_only_final = [
        index for index in sorted_index_text_only_final
        if not is_horizontal_line(bboxs[index]) and
        not is_vertical_line(bboxs[index])]
    if input_binarized:
        bboxs, sorted_index_text_only_final = merge_intersected(
            bboxs,
            sorted_index_text_only_final,
            debug)
    # Estimate foreground and background for each connected component
    binary_img = np.ones(img_shape)*255
    for index in sorted_index_text_only_final:
        positions = np.where(labels == (index+1))
        i_foreground = np.sum(gray[positions])
        num_fg = len(positions[0])
        if num_fg > 0:
            i_foreground /= num_fg
        i_background = get_median_background(gray, bboxs[index])
        x_start, x_stop, y_start, y_stop = get_x_y_start_stop(bboxs[index])
        if debug:
            print("Foreground: ", i_foreground, ",Background: ", i_background)
        if i_foreground < i_background:
            if input_binarized:
                i_foreground = 0
            binary_img[y_start:y_stop, x_start:x_stop] =\
                ((gray[y_start:y_stop, x_start:x_stop] > i_foreground) *
                 (binary_img[y_start:y_stop, x_start:x_stop] != 0))*255
        else:
            if input_binarized:
                i_foreground = 255
            binary_img[y_start:y_stop, x_start:x_stop] =\
                ((gray[y_start:y_stop, x_start:x_stop] < i_foreground) *
                 (binary_img[y_start:y_stop, x_start:x_stop] != 0))*255

    if not input_binarized:
        new_binarize_img = np.ones(binary_img.shape)*255
        labels, bboxs = cca_sk((255-binary_img))
        areas = [get_area_bounding_box(bbox) for bbox in bboxs]
        sorted_index_text_only = sorted_index(areas, inverse=True)
        sorted_index_text_only = filter_out_big_and_small_index(
            sorted_index_text_only, areas,
            15, np.prod(img_shape)/20, debug)
        sorted_index_text_only = filter_out_small_width_height_index(
            sorted_index_text_only, bboxs,
            3, 4, debug)
        sorted_index_text_only = filter_out_childs(
            bboxs, sorted_index_text_only, debug)
        sorted_index_text_only = [
            index for index in sorted_index_text_only
            if not is_horizontal_line(bboxs[index]) and
            not is_vertical_line(bboxs[index])]

        for index in sorted_index_text_only:
            x_start, x_stop, y_start, y_stop = get_x_y_start_stop(bboxs[index])
            new_binarize_img[y_start:y_stop, x_start:x_stop] = binary_img[
                y_start:y_stop,
                x_start: x_stop]
        binary_img = new_binarize_img
    if debug:
        print("Done binarizing")
        cv2.imwrite("debug_binary.png", binary_img)
        #cv2.imwrite("debug_binary_" + str(thres1) +
        #           "_" + str(thres2) + ".jpg", binary_img)
    return binary_img
