import numpy as np
import cv2
from numpy import prod

from ..local_imutils import (
    connected_component_analysis_sk_with_labels as cca_sk,)
from ..gradients_tools.edges_utils import get_edge_index_flatten
from ..bboxs_tools.bbox_operations import (get_area_bounding_box,
                                           check_bbox_contains_each_other,
                                           get_width_height_bounding_boxs,
                                           check_intersect_x,
                                           check_intersect_y,
                                           get_minimum_bounding_rect)


def filter_out_small(edges, labels, areas):
    sorted_index_reject_small = [index for index in range(len(areas))
                                 if areas[index] < 15]
    for index in sorted_index_reject_small:
        filtereds_positions = np.where(labels == index+1)
        edges[filtereds_positions] = 0


def filter_out_big(edges, labels, areas, img_shape):
    sorted_index_reject_big = [index for index in range(len(areas))
                               if areas[index] > prod(img_shape)/5]
    for index in sorted_index_reject_big:
        filtereds_positions = np.where(labels == index+1)
        edges[filtereds_positions] = 0


def filter_out_small_index(indices, areas, thres_areas=15, debug=False):
    """
    Input:
        indices: index in areas
        areas: areas of bboxs
        thres_areas: threshold
    Output:
        Filtered Indices
    """
    if debug:
        print("Len before small filter: ", len(indices))
    indices = [index for index in indices if areas[index] > thres_areas]
    if debug:
        print("Len after small filter: ", len(indices))
    return indices


def filter_out_big_index(indices, areas, thres_areas, debug=False):
    """
    Input:
        indices: index in areas
        areas: areas of bboxs
        thres_areas: threshold
    Output:
        Filtered Indices
    """
    if debug:
        print("Len before big filter: ", len(indices))
    indices = [index for index in indices if areas[index] < thres_areas]
    if debug:
        print("Len after big filter: ", len(indices))
    return indices


def filter_out_big_and_small_index(indices, areas,
                                   thres_under, thres_upper, debug=False):
    """
    Input:
        indices: index in areas
        areas: areas of bboxs
        thres_under: minimum area allowed
        thres_upper: maximum area allowed
    Output:
        Filtered Indices
    """
    indices = filter_out_small_index(indices, areas, thres_under, debug)
    indices = filter_out_big_index(indices, areas, thres_upper, debug)
    return indices


def filter_out_small_width_height_index(indices, bboxs,
                                        width_under=3, height_under=4,
                                        debug=False):
    """
    Input:
        indices: index in areas
        bboxs: bboxs
        width_under: minimum width allowed
        height_under: minimum height allowed
    """
    if debug:
        print("Prefilter width, height small: ", len(indices))
    widths, heights = get_width_height_bounding_boxs(bboxs)
    indices = [index for index in indices
               if widths[index] > width_under and heights[index] > height_under]
    if debug:
        print("After filter width, height small: ", len(indices))
    return indices


def filter_out_big_and_small(edges, debug=False):
    img_shape = edges.shape
    labels, bboxs = cca_sk(edges)
    areas = [get_area_bounding_box(bbox) for bbox in bboxs]

    # Filter 1:
    filter_out_small(edges, labels, areas)
    filter_out_big(edges, labels, areas, img_shape)
    if debug:
        print("Done filter out big and small")
        cv2.imwrite("debug_edges_filter_big_and_small.jpg", edges)
    return edges


def get_merged_edges(colored_image_array, thres1, thres2, debug=False):
    img_shape = colored_image_array.shape[:2]
    # Perform canny on each
    if len(colored_image_array.shape) > 2:
        edge_blue, edge_green, edge_red = [get_edge_index_flatten(
            colored_image_array,
            index,
            thres1, thres2) for index in range(3)]

        # Merge
        edges = np.maximum(edge_blue, edge_green)
        edges = np.maximum(edges, edge_red).reshape(img_shape)
    else:
        edges = cv2.Canny(colored_image_array, thres1, thres2)
    if debug:
        print("Done edging")
        cv2.imwrite("debug_edges_" + str(thres1) + "_" + str(thres2) + ".jpg",
                    edges)

    return edges


def recursive_filter_bboxs_kansas(sorted_index_text_only, bboxs,
                                  upper_child_bound=5, debug=False):
    sorted_index_text_only_final = []
    index = sorted_index_text_only[0]
    list_reject = []
    list_childs = [other_index
                   for other_index in sorted_index_text_only[1:]
                   if check_bbox_contains_each_other(
                       bboxs[index],
                       bboxs[other_index])[0]
                   ]
    if len(list_childs) < upper_child_bound:
        list_reject.extend(list_childs)
        return [index], list_reject
    else:
        if debug:
            print("list child lens: ", len(list_childs))
        list_reject.append(index)
        while(len(list_childs) > 0):
            sorted_index_text_only_final_new, list_reject_new =\
                recursive_filter_bboxs_kansas(list_childs, bboxs)
            sorted_index_text_only_final.extend(
                sorted_index_text_only_final_new)
            list_reject.extend(list_reject_new)
            list_childs = [x for x in list_childs
                           if x not in sorted_index_text_only_final_new and
                           x not in list_reject_new]
    return sorted_index_text_only_final, list_reject


def filter_out_childs(bboxs, sorted_index_text_only, upper_child_bound,
                      debug=False):
    sorted_index_text_only_final = []
    if debug:
        print("Prefiltered: ", len(sorted_index_text_only))
    while(len(sorted_index_text_only) > 0):
        list_add_new, list_reject_new = recursive_filter_bboxs_kansas(
            sorted_index_text_only, bboxs, upper_child_bound, debug)
        sorted_index_text_only_final.extend(list_add_new)
        sorted_index_text_only = [x for x in sorted_index_text_only
                                  if x not in list_add_new and
                                  x not in list_reject_new]

    if debug:
        print("Afterfiltered: ", len(sorted_index_text_only_final))
    return sorted_index_text_only_final


def merge_intersected(bboxs, sorted_index, debug=False):
    i = 0
    while i < len(sorted_index)-1:
        the_index = sorted_index[i]
        intersected_indices = [
            index
            for index in sorted_index[i+1:]
            if check_intersect_x(bboxs[the_index], bboxs[index]) and
            check_intersect_y(bboxs[the_index], bboxs[index])]
        sorted_index[i+1:] = [
            index for index in sorted_index[i+1:]
            if index not in intersected_indices
        ]
        intersected_indices.append(the_index)
        bboxs[the_index] = get_minimum_bounding_rect(
            [bboxs[intersected_index]
             for intersected_index in intersected_indices])
        if debug:
            print("Merging: ", i)
        i += 1
    return bboxs, sorted_index
