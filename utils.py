def compute_iou(box_a, box_b):
    """
    :param box_a: the first bounding box as a 1-D Numpy array
    :param box_b: the second bounding box as a 1-D Numpy array
    :return: IOU score
    """
    # Determine the (x, y)-coordinates of the intersection rectangle
    x_a = max(box_a[0], box_b[0])
    y_a = max(box_a[1], box_b[1])
    x_b = min(box_a[2], box_b[2])
    y_b = min(box_a[3], box_b[3])

    # Compute the area of intersection rectangle
    intersect = abs(max((x_b - x_a, 0)) * max((y_b - y_a), 0))
    if intersect == 0:
        return 0

    # Compute the area of both box A and box B
    box_a_area = abs((box_a[2] - box_a[0]) * (box_a[3] - box_a[1]))
    box_b_area = abs((box_b[2] - box_b[0]) * (box_b[3] - box_b[1]))

    # Compute the intersection over union by taking the intersection area and dividing it by (box A + box B - intersection area)
    iou = intersect / float(box_a_area + box_b_area - intersect)

    return iou


def check_duplicate(in_box, list_boxes, iou_thres=0.7):
    """
    :param in_box: input bounding box as a 1-D Numpy array
    :param list_boxes: list of bounding boxes
    :param iou_thres: IOU threshold
    :return: index of duplicate bounding box
    """
    for idx in range(len(list_boxes)):
        if compute_iou(list_boxes[idx], in_box) >= iou_thres:
            return idx
    return None