import cv2
import numpy as np
import math


def find_long_side_points(polygon: np.array):
    """Given a polygon in shape (8, ), fine the side with the longest length.
    Return the coordinates of the two points that define the longest side.
    """
    _polygon = polygon.reshape(-1, 2)
    _polygon = np.array(_polygon, dtype=np.float32)
    _polygon = np.concatenate((_polygon, _polygon[:1, :]), axis=0)
    _polygon = np.linalg.norm(_polygon[1:] - _polygon[:-1], axis=1)
    long_side_idx = np.argmax(_polygon)
    # return coordinates of the two points that define the longest side
    return polygon[long_side_idx], polygon[long_side_idx + 1]


def find_theta(line1: np.array,
               line2: np.array = np.array([0, 0, 0, 1]),
               if_switch: bool = False):
    """Fine the angle between two lines with direction. The angle range
    should be in [0, 180)
    """
    line1_copy = line1.copy().reshape(-1, 2)
    line1 = line1.reshape(-1, 2)
    line2 = line2.reshape(-1, 2)
    line1 = np.array(line1, dtype=np.float32)
    line2 = np.array(line2, dtype=np.float32)
    line1 = line1[1] - line1[0]
    line2 = line2[1] - line2[0]
    line1 = line1 / np.linalg.norm(line1)
    line2 = line2 / np.linalg.norm(line2)
    theta = np.arccos(np.dot(line1, line2))
    theta = theta / math.pi * 180
    if if_switch:
        if theta <= 90:
            return line1_copy[0], line1_copy[1], theta
        else:
            return line1_copy[1], line1_copy[0], 180 - theta
    else:
        return theta


def padding(img, polygon: np.array):
    """Padding around the image to make sure after rotation for any angle,
    the polygon will still be inside the image
    """
    polygon = polygon.reshape(-1, 2)
    # padding around the image
    img_h, img_w = img.shape[:2]
    max_side = max(img_h, img_w)
    img = cv2.copyMakeBorder(
        img,
        max_side,
        max_side,
        max_side,
        max_side,
        cv2.BORDER_CONSTANT,
        value=(0, 0, 0))
    polygon[:, 0] += max_side
    polygon[:, 1] += max_side
    return img, polygon


def rotate_crop(img, polygon: np.array):

    # find min area rect
    polygon = polygon.reshape(-1, 2)
    rect = cv2.minAreaRect(polygon)
    # calculate coordinates of the rotated rect
    box = cv2.boxPoints(rect)
    polygon = np.int0(box)
    # fine the side with the longest length
    p1, p2 = find_long_side_points(polygon)
    # find the angle between the longest side and the positive y-axis
    A, B, theta = find_theta(
        np.array([p1[0], p1[1], p2[0], p2[1]]),
        np.array([0, 0, 0, 1]),
        if_switch=True)
    alpha = find_theta(
        np.array([A[0], A[1], B[0], B[1]]),
        np.array([0, 0, 1, 0]),
        if_switch=False)
    # if alpha < 90, rotate the image and the polygon anti-clockwise by alpha
    if alpha < 90:
        alpha = alpha
    # if alpha >= 90, rotate the image and the polygon clockwise by 180 - alpha
    else:
        alpha = alpha + 180
    # get rotation matrix
    M = cv2.getRotationMatrix2D(rect[0], alpha, 1)
    # rotate the image and the polygon
    img_rotated = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))
    polygon_rotated = cv2.transform(np.array([polygon]),
                                    M).squeeze().astype(np.int32)
    # crop the image
    x, y, w, h = cv2.boundingRect(polygon_rotated)
    img_croped = img_rotated[y:y + h, x:x + w]
    return img_croped
