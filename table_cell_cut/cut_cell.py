from matplotlib import pyplot as plt
from matplotlib import gridspec
import cv2
import os
import sys
import numpy as np
from ocrolib.table_cell_cut.merging import *
from scipy.signal import argrelextrema
import logging
#logging.basicConfig(level = #logging.INFO)

from ocrolib.table_cell_cut.kernel_density_estimate import kernel_density_estimate
sys.path.append('F:/Hellios-workspace/Photoreader')

def show(window_name,img):
    pass
    #cv2.imshow(window_name, img)
    #cv2.waitKey(0)

def reduction(pivot_x, nearest_size):
    reduced_pivot = []
    if len(pivot_x) == 0: return reduced_pivot
    if len(pivot_x) == 1: return pivot_x
    Max = pivot_x[0]
    for i in range(1,len(pivot_x)):
        if abs(pivot_x[i]-pivot_x[i-1])<= nearest_size:
            Max = max(pivot_x[i], Max)
            # print('max', Max)
        else:
            # print('Max', Max)
            reduced_pivot.extend([Max])
            Max = pivot_x[i]
        if (i == len(pivot_x) - 1):
            reduced_pivot.extend([Max])
            print('Max1', Max)
    return reduced_pivot

def draw_line(img, pivots, wid, hei, direction):
    if direction == 0:
        '''horizontal'''
        print('horizontal')
        for pivot in pivots:
            # print('pi',pivot)
            cv2.line(img, (pivot,0),(pivot,hei),(255,0,0), 4)
    else:
        '''vertical'''
        print('vertical')
        for pivot in pivots:
            # print('pi1',pivot)
            cv2.line(img, (0,pivot),(wid, pivot),(255,0,0), 4)
    return img

def horizontalCutCells(table_path, plot):
    img = cv2.imread(table_path)
    img1 = np.asarray(img.copy())

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hei,wid = gray.shape
    _, bin_img = cv2.threshold(gray, 0, 1, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    print('shape ', bin_img.shape)
    '''projecting'''
    hist = np.sum(bin_img, axis=0, dtype = np.int32)
    list_bin = range(wid)
    '''calculate diff'''
    diff = (np.diff(hist))
    list_bin_diff = range(0,wid - 1)
    '''select top'''
    top = 20
    top_pivot = np.argsort(diff)[-top:]
    top_diff = np.sort(diff)[-top:]
    print('top pivot', top_pivot)
    print('top diff', top_diff)
    '''remove pivot < 40'''
    table_name_area_pivot = 40
    top_diff = top_diff[top_pivot > table_name_area_pivot]
    top_pivot = top_pivot[np.where(top_pivot>table_name_area_pivot)]
    print('top pivot1', top_pivot)
    print('top diff1', top_diff)
    print('top pivot', top_pivot)
    # print('hist', hist.shape, hist)
    # print('diff', diff.shape, diff)
    threshold_diff = 50
    pivot_x = top_pivot[np.where(top_diff>= threshold_diff)]
    # top_diff = [diff]
    # trip_hist = hist[int(wid / 3):int(wid * 2 / 3)]
    # x_min = np.argmax(trip_hist) + int(wid / 3)
    # print('trip ', trip_hist.shape)
    '''reduce pivot'''
    # reduced_diff = [diff for diff in pivot_x if ]
    pivot_x = sorted(pivot_x)
    print('pivot x', pivot_x)
    # mi, ma = kernel_density_estimate(calculus_hist, wid, 20)
    # thresh_interval_right = 160
    # thresh_interval_left = 120
    # list_arg_pivot_min = mi
    nearest_size = 50
    reduced_pivot = reduction(pivot_x, nearest_size)
    print('reduce', reduced_pivot)
    '''plot'''
    if(plot):
        # plt.ylim(ymin =0)
        fig = plt.figure()
        # plt.xlim([0, wid])
        # plt.ylim([0, hei])
        ax1 = plt.subplot(221)
        # plt.plot(x_min, hist[x_min], 'ro')
        ax1.set_title('Projection')
        ax1.set_xlim([0,wid])
        ax1.set_ylim([0,hei])
        plt.plot(list_bin, hist)

        ax2 = plt.subplot(222)
        # plt.plot(x_min, hist[x_min], 'ro')
        ax2.set_title('Differentiate')
        ax2.set_xlim([0, wid-1])
        ax2.set_ylim([0, hei])
        plt.plot(list_bin_diff, diff)

        for pivot in pivot_x:
            # print('pi',pivot)
            cv2.line(img, (pivot,0),(pivot,hei),(255,0,0), 2)
        ax3 = plt.subplot(223)
        ax3.set_title('Optimums')
        ax3.set_xlim([0, wid])
        ax3.set_ylim([hei, 0])
        plt.imshow(img)

        for pivot in reduced_pivot:
            # print('pi',pivot)
            cv2.line(img1, (pivot,0),(pivot,hei),(255,0,0), 2)
        ax4 = plt.subplot(224)
        ax4.set_title('Reduced optimums')
        ax4.set_xlim([0, wid])
        ax4.set_ylim([hei, 0])

        # plt.ylim(ymin=0)
        plt.xlim([0, wid])
        # plt.ylim([0, hei])
        plt.imshow(img1)
        plt.show()
        # cv2.imshow('table', img)

def verticalCutCells(table_path, plot):
    img = cv2.imread(table_path)
    img1 = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hei,wid = gray.shape
    _, bin_img = cv2.threshold(gray, 0, 1, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    print('shape ', bin_img.shape)
    '''projecting'''
    hist = np.sum(bin_img, axis=1, dtype = np.int32)
    list_bin = range(hei)
    '''calculate diff'''
    diff = np.diff(hist)
    list_bin_diff = range(0,hei - 1)
    '''select top'''
    top = 30
    top_pivot = np.argsort(diff)[-top:]
    top_diff = np.sort(diff)[-top:]
    print('top pivot', top_pivot)
    print('top diff', top_diff)
    '''remove pivot < 40'''
    table_name_area_pivot = 40
    top_diff = top_diff[top_pivot > table_name_area_pivot]
    top_pivot = top_pivot[np.where(top_pivot>table_name_area_pivot)]
    print('top pivot1', top_pivot)
    print('top diff1', top_diff)
    indices_pivot = np.arange(top)
    print('top pivot', top_pivot)
    # print('hist', hist.shape, hist)
    # print('diff', diff.shape, diff)
    threshold_diff = 120
    pivot_x = top_pivot[np.where(top_diff>= threshold_diff)]
    # top_diff = [diff]
    # trip_hist = hist[int(wid / 3):int(wid * 2 / 3)]
    # x_min = np.argmax(trip_hist) + int(wid / 3)
    # print('trip ', trip_hist.shape)
    '''reduce pivot'''
    # reduced_diff = [diff for diff in pivot_x if ]
    pivot_x = sorted(pivot_x)
    print('pivot x', pivot_x)
    nearest_size = 10
    reduced_pivot = reduction(pivot_x,nearest_size)
    # Max = pivot_x[0]
    # for i in range(1,len(pivot_x)):
    # 	if abs(pivot_x[i]-pivot_x[i-1])<= nearest_size:
    # 		Max = max(pivot_x[i], Max)
    # 	else:
    # 		print('gray', gray[Max,:].shape)
    # 		line_std = np.std(gray[Max,:])
    # 		print('line_std', line_std)
    # 		if line_std < 100:
    # 			reduced_pivot.extend([Max])
    # 			Max = pivot_x[i]
    # 	if (i == len(pivot_x) - 1):
    # 		reduced_pivot.extend([Max])
    # 		print('Max1', Max)
    print('reduce', reduced_pivot)
    '''plot'''
    if(plot):
        gs1 = gridspec.GridSpec(2,2, height_ratios=[1,1])
        # plt.ylim(ymin =0)
        fig = plt.figure()
        # plt.xlim([0, wid])
        # plt.ylim([0, hei])
        # ax1 = plt.subplot(gs1[0])
        ax1 = plt.subplot2grid((2,2),(0,0))
        # plt.plot(x_min, hist[x_min], 'ro')
        ax1.set_title('Projection')
        ax1.set_xlim([0,wid])
        ax1.set_ylim([0,hei])
        # plt.axes(ax1)
        plt.plot(hist, list_bin)

        for pivot in pivot_x:
            # print('pi',pivot)
            cv2.line(img, (0,pivot),(wid,pivot),(255,0,0), 1)
        # ax2 = plt.subplot(gs1[1])
        ax2 = plt.subplot2grid((2, 2), (0, 1), rowspan= 1,colspan=1)
        ax2.set_title('Optimums')
        ax2.set_xlim([0, wid])
        ax2.set_ylim([0, hei])
        ax2.imshow(img)
        ax2.set_aspect(1)
        plt.axes( [10, 20, 60, 90])

        # ax3 = plt.subplot(gs1[2])
        ax3 = plt.subplot2grid((2,2),(1,0))
        # plt.plot(x_min, hist[x_min], 'ro')
        ax3.set_title('Differentiate')
        ax3.set_xlim([0, wid - 1])
        ax3.set_ylim([0, hei])
        plt.plot(diff, list_bin_diff)

        for pivot in reduced_pivot:
            # print('pi',pivot)
            cv2.line(img1, (0,pivot),(wid,pivot),(255,0,0), 2)
        # ax4 = plt.subplot(gs1[3])
        ax4 = plt.subplot2grid((2, 2), (1,1))
        ax4.set_title('Reduced optimums')
        ax4.set_xlim([0, wid])
        ax4.set_ylim([0, hei])

        plt.tight_layout()
        # plt.ylim(ymin=0)
        plt.xlim([0, wid])
        # plt.ylim([0, hei])
        plt.imshow(img1)
        plt.show()
    # cv2.imshow('table', img)

def pivotDetection(origin_image, edge_image, plot):
    #logging.info('Pivots detection')
    fig_dir = '{}/debugCellCut/'.format(os.getcwd())
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)
    img1 = edge_image.copy()
    gray = cv2.cvtColor(edge_image, cv2.COLOR_BGR2GRAY)
    hei, wid = gray.shape
    _, bin_img = cv2.threshold(gray, 0, 1, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    print('shape ', bin_img.shape)
    #logging.info('shape')
    #logging.info(bin_img.shape)

    '''horizontal projecting'''
    hist_x = np.sum(bin_img, axis=0, dtype=np.int32)
    list_bin_x =  np.arange(0, wid)

    invert_hist_x = hei - hist_x

    '''select top'''
    top = 30
    top_pivot = np.argsort(invert_hist_x)[-top:]
    logging.info('top pivot')
    logging.info(top_pivot)
    logging.info(top_pivot.shape)
    #logging.info('top diff')
    #logging.info(top_diff)

    '''remove pivot < 40'''
    table_name_area_pivot = 40
    top_pivot = top_pivot[np.where((top_pivot > table_name_area_pivot)& (top_pivot < wid- table_name_area_pivot))]

    #logging.info('top pivot after first filter')
    #logging.info(top_pivot)
    # threshold_diff = np.mean(top_pivot)
    # print('threshold_diff ', threshold_diff)
    # pivot_x = top_pivot[invert_hist_x[top_pivot]>= threshold_diff -10]

    '''reduce pivot'''
    pivot_x = sorted(top_pivot)
    logging.info('pivot before reduction')
    logging.info(pivot_x)
    nearest_size = 30
    reduced_pivot_x = np.int32(reduction(pivot_x, nearest_size))
    logging.info('reduce')
    print('rd ', reduced_pivot_x)
    logging.info(reduced_pivot_x)

    '''vertical projecting'''
    hist_y = np.sum(bin_img, axis=1, dtype=np.int32)
    #logging.info('hist_y')
    #logging.info(hist_y)
    list_bin_y = range(hei)
    invert_hist_y = wid - hist_y

    '''select top'''
    top = 40
    top_pivot = np.argsort(invert_hist_y)[-top:]
    print('top pivot', top_pivot)
    #logging.info('top pivot')
    #logging.info(top_pivot)

    '''remove pivot < 40'''
    table_name_area_pivot = 40
    top_pivot_y = top_pivot[np.where((top_pivot > table_name_area_pivot)& (top_pivot < hei- table_name_area_pivot))]
    print('top pivot1', top_pivot_y)
    threshold_diff = 50
    pivot_y = top_pivot_y[invert_hist_y[top_pivot_y] >= threshold_diff]

    '''reduce pivot'''
    pivot_y = sorted(pivot_y)
    print('pivot y', pivot_y)
    nearest_size_horizontal = 20
    reduced_pivot_y = np.int32(reduction(pivot_y, nearest_size_horizontal))
    print('reduce', reduced_pivot_y)

    if(plot == 1):
        '''plot'''
        fig = plt.figure()
        # plt.xlim([0, wid])
        # plt.ylim([0, hei])
        ax1 = plt.subplot(321)
        # plt.plot(x_min, hist[x_min], 'ro')
        ax1.set_title('Projection_x')
        ax1.set_xlim([0, wid])
        # ax1.set_ylim([0, hei])
        plt.plot(list_bin_x,invert_hist_x)

        draw_line(origin_image, reduced_pivot_x, 0, hei, 0)
        draw_line(origin_image, reduced_pivot_y, wid, 0, 1)
        ax2 = plt.subplot(322)
        # plt.plot(x_min, hist[x_min], 'ro')
        ax2.set_title('Result')
        ax2.set_xlim([0, wid - 1])
        ax2.set_ylim([0, hei])
        plt.imshow(origin_image)

        ax3 = plt.subplot(323)
        # plt.plot(x_min, hist[x_min], 'ro')
        ax3.set_title('Projection_y')
        ax3.set_xlim([0, wid])
        ax3.set_ylim([0, hei])
        plt.plot(invert_hist_y,list_bin_y)

        # ax4 = plt.subplot(324)
        # # plt.plot(x_min, hist[x_min], 'ro')
        # ax4.set_title('Differentiate_y')
        # ax4.set_xlim([0, wid - 1])
        # ax4.set_ylim([0, hei])
        # plt.plot(diff_y, list_bin_diff_y)

        # for pivot in pivot_y:
        # 	# print('pi',pivot)
        # 	cv2.line(img, (pivot, 0), (pivot, hei), (255, 0, 0), 2)
        # draw_line(img, pivot_x, 0, hei, 0)
        draw_line(edge_image, top_pivot_y, wid, 0, 1)
        ax5 = plt.subplot(325)
        ax5.set_title('Optimums')
        ax5.set_xlim([0, wid])
        ax5.set_ylim([hei, 0])
        plt.imshow(edge_image)

        # for pivot in reduced_pivot_x:
        # 	# print('pi',pivot)
        # 	cv2.line(img, (pivot, 0), (pivot, hei), (255, 0, 0), 2)
        draw_line(img1, reduced_pivot_x, 0, hei, 0)
        draw_line(img1, reduced_pivot_y, wid,0, 1)
        ax6 = plt.subplot(326)
        ax6.set_title('Reduced optimums')
        ax6.set_xlim([0, wid])
        ax6.set_ylim([hei, 0])

        figManager = plt.get_current_fig_manager()
        figManager.window.showMaximized()
        plt.tight_layout()
        # plt.ylim(ymin=0)
        plt.xlim([0, wid])
        # plt.ylim([0, hei])
        plt.imshow(img1)
        fig.savefig('{}cell_plot.png'.format(fig_dir), dpi = 1200)
        plt.show()
    return origin_image, bin_img, reduced_pivot_x, reduced_pivot_y

def lineCutCells(im_gray, im_edge):
    img = np.array(im_gray * 255, dtype=np.uint8)
    im_table = np.array(im_edge * 255, dtype=np.uint8)
    # horizontal_arg = horizontal_pivot(img, table_file, filename)
    # vertical_arg = vertical_pivot(img, table_file, filename)
    debug = 0
    blank_origin, blank_table , horizontal_arg, vertical_arg = pivotDetection(img, im_table, debug)
    hei, wid = img.shape
    hei_bl, wid_bl = blank_table.shape
    print('gray shape', hei, wid, hei_bl, wid_bl)
    return cutCells(blank_origin, blank_table, horizontal_arg, vertical_arg, wid, hei)
    # for i in horizontal_arg:
    # 	cv2.line(img, (i, hei), (i ,0), (255,0,0), 4)
    #
    # for i in vertical_arg:
    # 	cv2.line(img, (wid, i), (0, i), (255,0,0), 4)
    #
    # cv2.imshow('',img)
    # cv2.waitKey(0)
    # cv2.imwrite('{}/Data/output/table/cells/{}/{}'.format(sys.path[-1],format, filename), img)

def cutCells(img, blank_table, list_pivot_x, list_pivot_y, wid, hei, cell_type):
    full_list_x = np.concatenate(([0],list_pivot_x,[wid-1]))
    full_list_y = np.concatenate(([0], list_pivot_y, [hei-1]))

    print('full', full_list_x.shape," ", full_list_y.shape)
    print('ful', full_list_x, " ", full_list_y)
    edge_matrix, width_mt, height_mt = createEdgeMatrix(full_list_x, full_list_y)
    updated_edge_matrix = updateEdgeMatrix(edge_matrix, width_mt, height_mt, full_list_x,
                                           full_list_y, blank_table, cell_type)
    list_edge_coordinates = detectBlocks(updated_edge_matrix)
    list_vertices_start_y = np.expand_dims(np.take(full_list_y, list_edge_coordinates[:, 0]), -1)
    list_vertices_end_y = np.expand_dims(np.take(full_list_y, list_edge_coordinates[:, 1]), -1)
    list_vertices_start_x = np.expand_dims(np.take(full_list_x, list_edge_coordinates[:, 2]), -1)
    list_vertices_end_x = np.expand_dims(np.take(full_list_x, list_edge_coordinates[:, 3]),-1)
    print('list_vertices_start_y\n', list_vertices_start_y )
    list_coordinates = np.hstack((list_vertices_start_x, list_vertices_start_y, list_vertices_end_x, list_vertices_end_y))
    print('list_coordinate', list_coordinates)
    for coordinates in list_coordinates:
        print('coordinate', coordinates)
        #cv2.rectangle(img, (coordinates[2], coordinates[0]),(coordinates[3], coordinates[1]),(255,0,2), 3)

    # show('blank', img)
    #cv2.imwrite('{}/Data/output/2/blank.PNG'.format(sys.path[-1]), img)
    #logging.info('length of list block')
    #logging.info(len(list_coordinates))
    return list_coordinates

def cutCellsAllTables(gray_table_dir, blank_table_dir):
    # for table_file in os.listdir(dir):
    for table_file in ['1.PNG']:
        blank_table_path = '{}{}'.format(blank_table_dir,table_file)
        gray_table_path = '{}{}'.format(gray_table_dir,table_file)
        if os.path.isfile(blank_table_path):
            lineCutCells(gray_table_path, blank_table_path)

# format = '1'
# format = '2'
# format = '3'
# format = '4'
# format = '5'
# format = '6'
# format = '7'
# format = '8'
# format = '9'
# format = '11'
# format = '12'
# gray_dir = '{}/Data/output/{}/cut/'.format(sys.path[-1], format)
# blank_dir = '{}/Data/{}/'.format(sys.path[-1], format)
# dir = '{}/Data/scratch/'.format(sys.path[-1])
# cutCellsAllTables(gray_dir, blank_dir)