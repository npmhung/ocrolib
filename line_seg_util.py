from __future__ import print_function
from __future__ import division

from ocrolib import psegutils, morph, sl
from scipy.ndimage.filters import gaussian_filter, uniform_filter, maximum_filter
from scipy.ndimage import measurements, interpolation
from scipy.misc import imsave
from numpy import (amax, minimum, maximum, array, zeros, where, transpose)
from matplotlib.mlab import find
import numpy as np

def DSAVE(title, image):
    if type(image) == list:
        assert len(image) == 3
        image = transpose(array(image), [1, 2, 0])
    fname = title + ".png"
    imsave(fname, image)

def compute_colseps_mconv(binary, scale=1.0):
    """Find column separators using a combination of morphological
    operations and convolution."""
    smoothed = gaussian_filter(1.0 * binary, (scale, scale * 0.5))
    smoothed = uniform_filter(smoothed, (5.0 * scale, 1))
    thresh = (smoothed < amax(smoothed) * 0.1)
    blocks = morph.r_closing(binary, (int(4 * scale), int(4 * scale)))
    seps = minimum(blocks, thresh)
    seps = morph.select_regions(seps, sl.dim0, min=10 * scale, nbest=3)
    blocks = morph.r_dilation(blocks, (5, 5))
    seps = maximum(seps, 1 - blocks)
    return seps

def compute_colseps_conv(binary, scale=1.0):
    """Find column separators by convoluation and
    thresholding."""
    # find vertical whitespace by thresholding
    smoothed = gaussian_filter(1.0 * binary, (scale, scale * 0.5))
    smoothed = uniform_filter(smoothed, (5.0 * scale, 1))
    thresh = (smoothed < amax(smoothed) * 0.1)
    # find column edges by filtering
    grad = gaussian_filter(1.0 * binary, (scale, scale * 0.5), order=(0, 1))
    grad = uniform_filter(grad, (10.0 * scale, 1))
    # grad = abs(grad) # use this for finding both edges
    grad = (grad > 0.5 * amax(grad))
    # combine edges and whitespace
    seps = minimum(thresh, maximum_filter(grad, (int(scale), int(5 * scale))))
    seps = maximum_filter(seps, (int(2 * scale), 1))
    # select only the biggest column separators
    seps = morph.select_regions(seps, sl.dim0, min=10 * scale, nbest=3)
    return seps

def compute_colseps(binary, scale):
    """Computes column separators either from vertical black lines or whitespace."""
    colseps = compute_colseps_conv(binary, scale)

    return colseps, binary

################################################################
### Text Line Finding.
###
### This identifies the tops and bottoms of text lines by
### computing gradients and performing some adaptive thresholding.
### Those components are then used as seeds for the text lines.
################################################################

def compute_gradmaps(binary, scale):
    # use gradient filtering to find baselines
    boxmap = psegutils.compute_boxmap(binary, scale)
    cleaned = boxmap * binary
    grad = gaussian_filter(1.0 * cleaned, (max(4, 0.3 * scale),
                                            scale), order=(1, 0))
    grad = uniform_filter(grad, (1.0, 1.0 * 6 * scale))
    bottom = norm_max((grad < 0) * (-grad))
    top = norm_max((grad > 0) * grad)
    return bottom, top, boxmap


def compute_line_seeds(binary, bottom, top, colseps, scale):
    """Base on gradient maps, computes candidates for baselines
    and xheights.  Then, it marks the regions between the two
    as a line seed."""
    t = 0.1
    vrange = int(1.0 * scale)
    bmarked = maximum_filter(bottom == maximum_filter(bottom, (vrange, 0)), (2, 2))
    bmarked = bmarked * (bottom > t * amax(bottom) * t) * (1 - colseps)
    tmarked = maximum_filter(top == maximum_filter(top, (vrange, 0)), (2, 2))
    tmarked = tmarked * (top > t * amax(top) * t / 2) * (1 - colseps)
    tmarked = maximum_filter(tmarked, (1, 20))
    seeds = zeros(binary.shape, 'i')
    delta = max(3, int(scale / 2))
    for x in range(bmarked.shape[1]):
        transitions = sorted([(y, 1) for y in find(bmarked[:, x])] + [(y, 0) for y in find(tmarked[:, x])])[::-1]
        transitions += [(0, 0)]
        for l in range(len(transitions) - 1):
            y0, s0 = transitions[l]
            if s0 == 0: continue
            seeds[y0 - delta:y0, x] = 1
            y1, s1 = transitions[l + 1]
            if s1 == 0 and (y0 - y1) < 5 * scale: seeds[y1:y0, x] = 1
    seeds = maximum_filter(seeds, (1, int(1 + scale)))
    seeds = seeds * (1 - colseps)
    seeds, _ = morph.label(seeds)
    return seeds  # [seeds, 0.3 * tmarked + 0.7 * bmarked, binary]

################################################################
### Line segmentation from binary image
################################################################

def text_line_segmentation(binary, scale=None, gray=None, num_col = 1):
    """Given a binary image, compute a complete segmentation into
    lines, computing both columns and text lines."""
    binary = array(binary, 'B')
    if scale is None:
        scale = psegutils.estimate_scale(binary)

    # do the column finding
    if num_col > 1:
        colseps, binary = compute_colseps(binary, scale)
    else:
        colseps = np.zeros(binary.shape)

    # now compute the text line seeds
    bottom, top, boxmap = compute_gradmaps(binary, scale)
    seeds = compute_line_seeds(binary, bottom, top, colseps, scale)

    # spread the text line seeds to all the remaining components
    llabels = morph.propagate_labels(boxmap, seeds, conflict=0)
    spread = morph.spread_labels(seeds, maxdist=scale)
    llabels = where(llabels > 0, llabels, spread * binary)
    segmentation = llabels * binary

    lines = psegutils.compute_lines(segmentation, scale, 0.8)
    line_ims = []

    for l in lines:
        if gray is None:
            binline = psegutils.extract_masked(1-binary, l, pad=0)
        else:
            binline = psegutils.extract_masked(gray, l, pad=0)
        binline = pad_by(binline, 10, invert=False)
        line_ims.append(binline)

    return line_ims, lines

### dash-separated line estimation
def compute_scale_from_grid(binary):
    # project to X Y axis to compute scale between seps
    binary = np.array((255 - binary) > 0).astype('uint8') # invert
    proj_x = np.sum(binary, axis=0)
    h, w = binary.shape
    mask_x = proj_x > h * 0.2
    sep_count = len(np.argwhere(mask_x))
    if sep_count > 0:
        sep_x = np.diff([i[0] for i in np.argwhere(mask_x)])
        sep_x = sep_x[sep_x > 10]
        if len(sep_x) > 0:
            scale = np.median(sep_x).astype(int)
        else:
            if np.argwhere(mask_x)[0][0] > 10 and np.argwhere(mask_x)[0][0] < w - 10:
                scale = w // 2
            else:
                scale = w
    else:
            scale = w

    return scale

def compute_checkbox_position(blank_im):
    binary = convert_to_binary(255 - blank_im)
    labels, n = morph.label(binary)
    h, w = binary.shape
    minsize = 40

    # find small dash in img
    sums = measurements.sum(binary, labels, range(n + 1))
    sums = sums[labels]
    good = minimum(binary, 1 - (sums > 0) * (sums < minsize))

    junk_cc = np.bitwise_xor(good, binary)
    # temporary fix: add bottom line
    junk_cc[h-1:, :] = np.ones((1, w))
    junk_cc = morph.r_dilation(junk_cc, (7,7))
    junk_cc = morph.r_closing(junk_cc, (9,9))

    # find hole using morphology
    hole = morph.fill_hole(junk_cc)
    hole = hole - junk_cc

    # locate holes position
    labels, n = morph.label(hole)
    objects = morph.find_objects(labels)
    objects = sorted(objects, key=lambda b: sl.center(b))
    area_thres = 0.4 * (amax([sl.area(b) for b in objects]) if len(objects) > 0 else 0)
    boxes = [[b[0].start, b[1].start, b[0].stop, b[1].stop] for b in objects if sl.area(b) > area_thres]

    return boxes, convert_binary_to_normal_im(hole)

def read_check_mark_position(im, boxes, shift):
    # if no boxes in blank template, skip
    if len(boxes) == 0:
        return []

    binary = convert_to_binary(255 - im)
    shift_x, shift_y = shift
    shifted_boxes = [[b[0]+shift_y, b[1]+shift_x, b[2]+shift_y, b[3]+shift_x] for b in boxes]
    h, w = binary.shape
    pad = 5
    expand_boxes = [[max(b[0] - pad, 0), max(b[1] - pad, 0), min(b[2] + pad, h), min(b[3] + pad, w)] for b in shifted_boxes]
    counts = np.array([np.sum(binary[b[0]:b[2], b[1]:b[3]]) for b in expand_boxes])
    thres = max(amax(counts) * 0.8, 12)

    return np.argwhere(counts > thres)

def remove_small_noise(binary, minsize = 50):
    labels, n = morph.label(binary)
    h, w = binary.shape
    objects = morph.find_objects(labels)
    space_to_edge = 10
    sums = measurements.sum(binary, labels, range(n + 1))
    sums = sums[labels]
    good = minimum(binary, 1 - (sums > 0) * (sums < minsize))

    for i, b in enumerate(objects):
        cy, cx = sl.center(b)
        # if component is small and close to edge
        if (sl.area(b) < minsize * 1.2 and ((cx < space_to_edge or cx > w - space_to_edge) or (cy < space_to_edge or cy > h - space_to_edge))):
            good[b][labels[b] == i+1] = 0

    return good

def convert_to_binary(im, thres = 0.6):
    return np.array(im>thres*(np.amin(im)+np.amax(im))).astype('B')

def convert_to_norm_gray(im):
    return im / 255.0

def remove_small_noise_and_seps(im, num_cells, minsize = 30):
    im = 255 - im
    binary = np.array(im>0.5*(np.amin(im)+np.amax(im))).astype('uint8')  # invert
    h, w = im.shape
    scale = int(w / num_cells)

    labels, n = morph.label(binary)
    objects = morph.find_objects(labels)

    # remove small noise using connected components
    sums = measurements.sum(binary, labels, range(n + 1))
    sums = sums[labels]
    good = minimum(binary, 1 - (sums > 0) * (sums < minsize))

    # calculate sep bar positions from junk cc
    junk_cc = np.bitwise_xor(binary, good)

    # remove long connected component (solid separators)
    proj_x = np.sum(binary, axis=0)
    mask_x = np.tile((proj_x > h * 0.8).astype('B'), h)
    solid_sep_pos = [j[0] for j in np.argwhere(proj_x > h * 0.6)]
    good[mask_x] = 0
    '''for i, b in enumerate(objects):
            if sl.width(b) < 6 and sl.height(b) > h * 0.9:
                good[b][labels[b] == i + 1] = 0
        '''

    if np.sum(junk_cc) > 140:
        # only detect sep bars if has enough pixels
        proj_x = np.sum(junk_cc, axis=0)
        mask_x = proj_x > np.amax(proj_x) * 0.2
        sep_pos = np.array([i[0] for i in np.argwhere(mask_x)])
        start = [True] + [True if abs(sep_pos[i] - sep_pos[i-1] - scale) < 5 or abs(sep_pos[i] - sep_pos[i-1] - 2 * scale) < 5 else False for i in range(1,len(sep_pos))]
    else:
        sep_pos = []

    if len(sep_pos) > 0:
        start_sep_pos = sep_pos[start]
        #print(start_sep_pos)

        # fill-in missing pos
        '''for i in range(1,len(start_sep_pos)):
            if start_sep_pos[i] - start_sep_pos[i-1] > scale + 4:
                mid = (start_sep_pos[i] + start_sep_pos[i-1]) // 2
                good[0:h, mid:mid + 5] = 0
        '''

        # fill seps start from begin sep with scale space
        if len(start_sep_pos) > 0 and len(solid_sep_pos) > 0:
            pos_x = start_sep_pos[0]
            scale = int(round(1.0 * w / num_cells) + 0.1)
            while pos_x < w:
                if any(x in solid_sep_pos for x in range(pos_x-3,pos_x+4)):
                    pos_x = min([x for x in range(pos_x-3,pos_x+4) if x in solid_sep_pos])
                    good[0:h,pos_x:pos_x+5] = 0
                pos_x += scale
        else:
            # handle special case for 2 cells
            if w / scale > 1.5 and w / scale < 2.6:
                mid = w // 2
                good[0:h, mid:mid + 5] = 0

    else:
        # fill seps start from solid sep with scale space
        proj_x = np.sum(good, axis=0)
        mask_x = proj_x > h * 0.9
        sep_pos = np.array([i[0] for i in np.argwhere(mask_x)])
        pos_x = scale if len(sep_pos) == 0 else sep_pos[0]
        while pos_x < w:
            good[0:h, pos_x:pos_x + 5] = 0
            pos_x += scale + 1

    return np.array((1-good) * 255).astype('uint8')

def cut_dash_line(im, num_cells):
    binary = convert_to_binary(255-im, thres=0.5)
    labels, _ = morph.label(binary)
    objects = morph.find_objects(labels)

    scale = int(round(1.0 * binary.shape[1] / num_cells + 0.2))
    h = binary.shape[0] - 1
    # list to store objects for each cell
    cells = [[] for _ in range(num_cells)]
    cell_ims = []

    for i, b in enumerate(objects):
        # only process object with width < 2 x scale
        if sl.width(b) < 2 * scale:
            x1, x2 = b[1].start, b[1].stop
            mid_x = (x1 + x2) // 2
            cell_index = np.median([x1 // scale, x2 // scale, mid_x // scale]).astype(int)
            #print(cell_index)
            # handle case where digit from 2 cells connected
            if x2 - (cell_index + 1) * scale > 0.3 * scale:
                temp_b = (b[0], slice(b[1].start, (cell_index + 1) * (scale + 1), None))
                print("2 char connected!!!")
            else:
                temp_b = b
            cells[cell_index].append(temp_b)

    for i, c in enumerate(cells):
        if len(c) > 0:
            x1 = min([obj[1].start for obj in c])
            x2 = max([obj[1].stop for obj in c])
            cell_ims.append(normalize_cell_img(im[0:h, x1:x2]))
        else:
            blank = np.zeros((h, scale))
            cell_ims.append(normalize_cell_img(convert_binary_to_normal_im(blank)))

    return cell_ims

def extract_region_from_image_cc(im, rect):
    pad = 10
    y0, x0, y1, x1 = rect

    local_im = im[y0-pad:y1+pad, x0-pad:x1+pad].copy()
    h, w = local_im.shape

    box = (slice(pad, h - pad, None), slice(pad, w - pad, None))
    binary = convert_to_binary(255 - local_im)
    labels, _ = morph.label(binary)
    objects = morph.find_objects(labels)
    region_objs = []

    for i, b in enumerate(objects):
        if 1.0 * sl.xoverlap(b, box) * sl.yoverlap(b, box) / sl.area(b) > 0.55:
            region_objs.append(b)
        else:
            binary[b][labels[b]==i+1] = 0

    x2 = min([obj[1].start for obj in region_objs] + [pad])
    x3 = max([obj[1].stop for obj in region_objs] + [w - pad])
    y2 = min([obj[0].start for obj in region_objs] + [pad])
    y3 = max([obj[0].stop for obj in region_objs] + [h - pad])

    return convert_binary_to_normal_im(binary[y2:y3, x2:x3]), (pad-x2, x3-w+pad, pad-y2, y3-h+pad)

def expand_blank_im(im, blank_bin, expand):
    h, w = im.shape
    bh, bw = blank_bin.shape
    pad_left, pad_right, pad_top, pad_bottom = expand
    print(expand)
    pad_left = int(1.0 * pad_left * bw / w)
    pad_right = int(1.0 * pad_right * bw / w)
    pad_top = int(1.0 * pad_top * bh / h)
    pad_bottom = int(1.0 * pad_bottom * bh / h)
    new_bh = bh + pad_top + pad_bottom
    new_bw = bw + pad_left + pad_right
    new_blank = np.zeros((new_bh, new_bw)).astype('B')
    new_blank[pad_top:pad_top+bh, pad_left:pad_left+bw] = blank_bin

    return new_blank, (new_bw, new_bh)

def match_image_with_blank(im, blank_bin):
    binary = convert_to_binary(255 - im, thres=0.6)
    h, w = binary.shape
    max_shift_ratio = 0.06
    range_x = int(max_shift_ratio * w)
    range_y = int(max_shift_ratio * h)
    sum_blank = np.sum(blank_bin)
    if sum_blank < 10:
        return im, None
    thres = 0.2
    max_match = -1
    shift_x, shift_y = 0, 0
    for x in range(-range_x, range_x):
        for y in range(-range_y, range_y):
            temp_blank = shift_binary(blank_bin, (y, x)) #interpolation.shift(blank_bin, (y, x))
            match = np.sum(temp_blank & binary)
            if 1.0 * match / sum_blank > thres and match > max_match:
                max_match = match
                shift_x, shift_y = x, y
    if max_match != -1:
        #print(shift_x, shift_y)
        temp_blank = shift_binary(blank_bin,  (shift_y, shift_x))  #interpolation.shift(blank_bin, (shift_y, shift_x))
        temp_blank = morph.r_dilation(temp_blank, (5,5))
        binary = binary & (1 - temp_blank) #0.5 * binary + 0.5 * temp_blank

    return convert_binary_to_normal_im(binary), (shift_y, shift_x)

# center and crop cell image for better OCR performance
def normalize_cell_img(im, pad = 12.0):
    binary = convert_to_binary(255-im)
    # calculate center of mass
    center = measurements.center_of_mass(binary)
    shift = [binary.shape[0] / 2 - center[0], binary.shape[1] / 2 - center[1]]
    binary = interpolation.shift(binary, shift)

    # Mask of non-black pixels (assuming image has a single channel).
    content = binary > 0
    # Coordinates of non-black pixels.
    coords = np.argwhere(content)
    # Bounding box of non-black pixels.
    if coords.shape[0] > 0:
        y0, x0 = coords.min(axis=0)
        y1, x1 = coords.max(axis=0) + 1  # slices are exclusive at the top
    else:
        x0, y0 = 0, 0
        x1, y1 = binary.shape[1], binary.shape[0]

    # crop to content
    binary = binary[y0:y1, x0:x1]

    # pad image
    pad_ratio = pad / 28 / 2
    pad_x = int(pad_ratio * binary.shape[1])
    pad_y = int(pad_ratio * binary.shape[0])
    binary = np.pad(binary, ((pad_y,pad_y), (pad_x,pad_x)), mode='constant', constant_values=0)

    # pad to square
    h, w = binary.shape
    pad_x = (h - w) // 2 if h > w else 0
    pad_y = (w - h) // 2 if h < w else 0
    binary = np.pad(binary, ((pad_y, pad_y), (pad_x, pad_x)), mode='constant', constant_values=0)

    return ((1-binary)*255).astype('uint8')

def convert_binary_to_normal_im(binary):
    return ((1-binary)*255).astype('uint8')

def shift_binary(origin_bin, shift):
    binary = origin_bin.copy()

    h, w = binary.shape
    binary = np.roll(binary, shift, axis=(0,1))
    if shift[0] > 0:
        y0, y1 = 0, shift[0]
    else:
        y0, y1 = h + shift[0], h

    if shift[1]>0:
        x0, x1 = 0, shift[1]
    else:
        x0, x1 = w + shift[1], w

    binary[y0:y1, :] = np.zeros((y1-y0, w))
    binary[:, x0:x1] = np.zeros((h, x1-x0))

    return binary