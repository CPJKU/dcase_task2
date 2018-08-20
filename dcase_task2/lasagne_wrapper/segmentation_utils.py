# -*- coding: utf-8 -*-

import lasagne
import numpy as np


def dice(Seg, GT):
    """ compute dice coefficient between current segmentation result and groundtruth (GT)"""

    sum_GT = np.sum(GT)
    sum_Seg = np.sum(Seg)

    if (sum_GT + sum_Seg) == 0:
        dice = 1.0
    else:
        dice = (2.0 * np.sum(Seg[GT == 1])) / (sum_Seg + sum_GT)

    return dice


def compute_label_assignment(seg, gt):
    """ assign segmentation labels to ground truth """
    import munkres
    from skimage.measure import regionprops

    seg_rprops = regionprops(seg)
    gt_rprops = regionprops(gt)

    if len(seg_rprops) == 0 or len(gt_rprops) == 0:
        assignment = np.zeros((0, 2), dtype=np.int)

    else:

        D = np.zeros((len(seg_rprops), len(gt_rprops)), dtype=np.float)
        for i, rprops_seg in enumerate(seg_rprops):
            for j, rprops_gt in enumerate(gt_rprops):

                min_row_s, min_col_s, max_row_s, max_col_s = rprops_seg.bbox
                min_row_g, min_col_g, max_row_g, max_col_g = rprops_gt.bbox

                def get_overlap(a, b):
                    return max(0, min(a[1], b[1]) - max(a[0], b[0]))

                overlap_row = get_overlap([min_row_s, max_row_s], [min_row_g, max_row_g])
                overlap_col = get_overlap([min_col_s, max_col_s], [min_col_g, max_col_g])

                if overlap_row > 0 and overlap_col > 0:
                    D[i, j] = dice(seg == rprops_seg.label, gt == rprops_gt.label)
                else:
                    D[i, j] = 0

        # init minimum assignment algorithm
        D *= -1
        m = munkres.Munkres()

        if D.shape[0] <= D.shape[1]:
            assignment = np.asarray(m.compute(D.copy()))
        else:
            assignment = np.asarray(m.compute(D.T.copy()))
            assignment = assignment[:, ::-1]

        # remove high cost assignments
        costs = np.asarray([D[i, j] for i, j in assignment])
        assignment = assignment[costs < 0]

        # for i, j in assignment:
        #     print(i, j, seg_labels[i], gt_labels[j], D[i, j])
        #
        # import matplotlib.pyplot as plt
        # plt.figure()
        #
        # plt.subplot(2, 2, 1)
        # plt.imshow(D, interpolation="nearest")
        # for i, j in assignment:
        #     plt.plot(j, i, 'mo', alpha=0.5)
        # plt.colorbar()
        # plt.ylabel("gt %d" % len(gt_labels))
        # plt.xlabel("seg %d" % len(seg_labels))
        # plt.title("%d assignments" % len(assignment))
        #
        # plt.subplot(2, 2, 3)
        # plt.imshow(seg, interpolation="nearest")
        #
        # plt.subplot(2, 2, 4)
        # plt.imshow(gt, interpolation="nearest")
        #
        # plt.show()

    # prepare labels
    seg_labels = np.asarray([r.label for r in seg_rprops])
    gt_labels = np.asarray([r.label for r in gt_rprops])

    return assignment, seg_labels, gt_labels


def compute_recognition_assignment(assignment, seg_labels, gt_labels):
    """ comput fp, fn, tp """
    seg_visited = np.zeros(len(seg_labels))
    gt_visited = np.zeros(len(gt_labels))

    for i_seg, i_gt in assignment:
        seg_visited[i_seg] = 1
        gt_visited[i_gt] = 1

    tp_idx = np.nonzero(seg_visited)[0]
    fp_idx = np.nonzero(seg_visited == 0)[0]
    fn_idx = np.nonzero(gt_visited == 0)[0]

    return tp_idx, fp_idx, fn_idx


def pixelwise_softmax(net):
    """
    Apply pixelwise softmax
    """
    
    # get number of classes
    Nc = net.output_shape[1]
    
    # reshape 2 softmax
    shape = net.output_shape
    net = lasagne.layers.ReshapeLayer(net, shape=(-1, Nc, shape[2]*shape[3]))
    net = lasagne.layers.DimshuffleLayer(net, (0,2,1))
    net = lasagne.layers.ReshapeLayer(net, shape=(-1, Nc))

    net = lasagne.layers.NonlinearityLayer(net, nonlinearity=lasagne.nonlinearities.softmax)    

    # reshape 2 image
    net = lasagne.layers.ReshapeLayer(net, shape=(-1, shape[2]*shape[3], Nc))
    net = lasagne.layers.DimshuffleLayer(net, (0,2,1))
    net = lasagne.layers.ReshapeLayer(net, shape=(-1,shape[1],shape[2],shape[3]))
    
    return net