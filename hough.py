import cv2
import numpy as np


def hough_lines_p(img):
    """
    Grabs the hough lines from an image
    :param img: Image
    :return: Array of lines
    """
    min_line_length = 0
    max_line_gap = 10

    lines = cv2.HoughLinesP(img, 1, np.pi/180, 20, min_line_length, max_line_gap)

    return lines


def canny(img):
    """
    Returns edges found in an image
    :param img: Image
    :return: Processed image only showing edges
    """
    return cv2.Canny(img, 100, 200, L2gradient=True)


def gauss(img, kernel):
    """
    Determines the gauss blur of an image
    :param img: Image
    :param kernel: Kernel size for blurring effect
    :return: Gaussian blurred image
    """
    return cv2.GaussianBlur(img, (kernel, kernel), 0)


def draw_lines(img, lines, dest=None):
    """
    Draws lines on to image
    :param img: Image
    :param lines: Lines to be drawn
    :param dest: Image to be drawn on
    :return: Image with lines drawn on
    """
    if dest is None:
        dest = img

    if lines is None:
        print("No lines")
        return dest

    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(dest, (x1, y1), (x2, y2), thickness=7, color=(0, 255, 0))

    return dest


def roi(img):
    """
    Grabs the region of interest in an image
    :param img: Image
    :return: Region of interest
    """
    mask = np.zeros_like(img)

    pts = np.array([[(835, 330), (375, 330), (0, 720), (1280, 720)]], dtype=np.int32)

    if len(img.shape) > 2:
        channel_count = img.shape[2]
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    cv2.fillPoly(mask, pts, ignore_mask_color)

    masked_image = cv2.bitwise_and(img, mask)

    return masked_image


def process_image(img):
    """
    Grabs an image, determines the lanes, draws them and determiness the angle to turn to the middle of the lanes
    :param img: Image
    :return: Processed image
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    blur = gauss(gray, 5)

    edges = canny(blur)

    interest = roi(edges)

    lines = hough_lines_p(interest)

    left, right = determine_lanes(lines)

    m, b = determine_slope_intercept(left)

    n, c = determine_slope_intercept(right)

    if check_lanes(m, n):
        p1, p2 = determine_points(m, b, 270, 719)
        p3, p4 = determine_points(n, c, 270, 719)
        x, y = intersecting_lane_point(m, b, n, c)
        vanish_point = draw_vanishing_point(img, x, y)
        processed = draw_lane(vanish_point, p1, p2, p4, p3)
    else:
        print("THROW IMAGE")
        processed = img

    # print((determine_midway(p1, p4, width/2)))

    return processed


def determine_lanes(lines):
    """
    Determines the lanes from a set of lines
    :param lines: Lines
    :return: Two arrays of tuples containing the slope and y-intercept of the left lane and the right lane
    """
    left_lane = []
    right_lane = []

    if lines is None:
        return left_lane, right_lane

    for line in lines:
        for x1, y1, x2, y2 in line:
            if x1 == x2 or y1 == y2:
                continue
            slope = (y2-y1)/(x2-x1)
            intercept = y1 - slope * x1

            if slope < -1:
                left_lane.append([slope, intercept])
            if slope > 1:
                right_lane.append([slope, intercept])

    return left_lane, right_lane


def determine_slope_intercept(lines):
    """
    Determines the average slope and average y-intercept from an array of lines
    :param lines: Array of lines
    :return: Average slope, average y-intercept
    """
    total_slope = 0
    total_intercept = 0

    for slope, intercept in lines:
        total_slope += slope
        total_intercept += intercept

    if len(lines) > 0:
        avg_slope = total_slope / len(lines)
        avg_intercept = total_intercept / len(lines)
    else:
        avg_slope = 0
        avg_intercept = 0

    return avg_slope, avg_intercept


def determine_points(slope, intercept, cutoff=0, height=0):
    """
    Given a line, determines two points that are relevant to the region of interest
    :param slope: Slope of line
    :param intercept: Y-intercept of line
    :param cutoff: Top of the region of interest
    :param height: Bottom of the region of interest
    :return: Two points for a line
    """
    if slope is 0:
        x1 = x2 = 0
    else:
        x1 = (cutoff - intercept)/slope
        x2 = (height-intercept)/slope

    return (int(x1), cutoff), (int(x2), height)


def draw_lane(img, left_p1, left_p2, right_p1, right_p2, dest=None):
    """
    Draws lanes given 4 points, 2 for each line
    :param img: Image
    :param left_p1: First point for left lane
    :param left_p2: Second point for left lane
    :param right_p1: First point for right lane
    :param right_p2: Second point for right lane
    :param dest: Lanes to be drawn on
    :return: Image with lanes drawn on
    """
    if dest is None:
        dest = img

    cv2.line(dest, left_p1, left_p2, thickness=5, color=(255, 0, 0))
    cv2.line(dest, right_p1, right_p2, thickness=5, color=(0, 0, 255))

    return dest


def determine_midway(p1, p2, mid_pic):
    """
    Determines how off the middle of the image is from the middle of two points
    :param p1: Point 1
    :param p2: Point 2
    :param mid_pic: Middle position of image
    :return: Percent erro
    """
    mid_of_lane = (p1[0] + p2[0])/2
    error = round(percent_error(mid_pic, mid_of_lane), 4)

    return error


def percent_error(experimental, actual):
    """
    Returns percent error
    :param experimental: Experimental value
    :param actual: Actual value
    :return: Percent error
    """
    return ((experimental-actual)/actual) * 100


def check_lanes(left_slope, right_slope):
    """
    Checks if the lanes are valid
    :param left_slope: Slope of left lane
    :param right_slope: Slope of right lane
    :return: True if both lanes are valid
    """
    return check_left_lane(left_slope) and check_right_lane(right_slope)


def check_left_lane(slope):
    """
    Checks if left lane is valid
    :param slope: Slope of left lane
    :return: True if negative slope
    """
    return slope < 0


def check_right_lane(slope):
    """
    Checks if right lane is valid
    :param slope: Slope of right lane
    :return: True if positive slope
    """
    return slope > 0


def intersecting_lane_point(left_slope, left_intercept, right_slope, right_intercept):
    """
    Determines the intersectingpoint between two lines
    :param left_slope: Slope of left line
    :param left_intercept: Y-intercept of left line
    :param right_slope: Slope of right line
    :param right_intercept: Y-intercept of right line
    :return: X, Y coordinate
    """
    x = (right_intercept - left_intercept)/(left_slope - right_slope)
    y = (right_slope * x) + right_intercept

    return int(x), int(y)


def draw_vanishing_point(img, x, y, dest=None):
    """
    Draws the vanishing point in an image
    :param img: Image
    :param x: X coordinate
    :param y: Y coordinate
    :param dest: Image to be drawn on
    :return: Image with point drawn on
    """
    if dest is None:
        dest = img

    cv2.circle(dest, (x, y), 1, (0, 255, 0), thickness=10)

    return dest


def determine_angle_to_vp(midpoint, height, x, y):
    """
    Determines the angle from the middle-bottom of the image to the vanishing point of the image
    :param midpoint: Middle of image
    :param height: Height of image
    :param x: X coordinate of vanishing point
    :param y: Y coordinate of vanishing point
    :return: Degrees to vanish point
    """
    p1 = (midpoint, height)
    p2 = (midpoint, y)
    p3 = (x, y)

    opp= length_line(p2, p3)
    hyp = length_line(p1, p3)

    theta = np.arcsin(opp/hyp)

    degree = radians_to_degree(theta)

    return degree


def length_line(pt1, pt2):
    """
    Determines the length of a line
    :param pt1: First point
    :param pt2: Second point
    :return: Length of line
    """
    a = (pt2[0]-pt1[0])
    b = (pt2[1]-pt1[1])
    length = np.sqrt((a*a)+(b*b))

    return length


def radians_to_degree(radians):
    """
    Converts radians to degrees
    :param radians: Radians
    :return: Degrees
    """
    return (radians*180)/np.pi


def red(img, dest=None):
    """
    Grabs the red channel from an image
    :param img: Image
    :param dest: Destination
    :return: Red channel image
    """
    if dest is None:
        dest = img.copy()

    dest[:, :, 0] = 0
    dest[:, :, 1] = 0

    return dest


def blue(img, dest=None):
    """
    Grabs the blue channel from an image
    :param img: Image
    :param dest: Destination
    :return: Blue channel image
    """
    if dest is None:
        dest = img.copy()

    hsv = cv2.cvtColor(dest, cv2.COLOR_BGR2HSV)

    return hsv


def green(img, dest=None):
    """
    Grabs the green channel from an image
    :param img: Image
    :param dest: Destination
    :return: Green chanel image
    """
    if dest is None:
        dest = img.copy()

    dest[:, :, 0] = 0
    dest[:, :, 2] = 0

    return dest


def grab_color(img, lower, upper):
    """
    Grabs only certain color from an image
    :param img: Image
    :param lower: Lower bound
    :param upper:
    :return:
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    l_bound = np.array(lower, dtype=np.uint8)
    u_bound = np.array(upper, dtype=np.uint8)

    mask = cv2.inRange(hsv, l_bound, u_bound)

    return mask


def left_right_both(img, threshold=1000):
    """
    Determines whether image contains both lanes, left lane, right lane or neither
    :param img: Image
    :param threshold: Pixels to determine
    :return: Print
    """
    blue = grab_color(img, [100, 100, 100], [220, 255, 255])
    white = grab_color(img, [0, 0, 0], [10, 0, 255])

    n_blue = cv2.countNonZero(blue)     # Counts the number of blue pixels
    n_white = cv2.countNonZero(white)   # Counts the number of white pixels;

    cv2.imshow('blue', blue)
    cv2.imshow('white', white)

    # Checks if
    if n_blue > threshold and n_white > threshold:
        print("BOTH LANES")
    elif n_blue > threshold:
        print("RIGHT")
    elif n_white > threshold:
        print("LEFT")
    else:
        print("NONE")


def return_num(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    blur = gauss(gray, 5)

    edges = canny(blur)

    interest = roi(edges)

    lines = hough_lines_p(interest)

    left, right = determine_lanes(lines)

    m, b = determine_slope_intercept(left)

    n, c = determine_slope_intercept(right)

    if check_lanes(m, n):
        # p1, p2 = determine_points(m, b, 270, 719)
        # p3, p4 = determine_points(n, c, 270, 719)
        # x, y = intersecting_lane_point(m, b, n, c)
        # vanish_point = draw_vanishing_point(img, x, y)
        # processed = draw_lane(vanish_point, p1, p2, p4, p3)
        return 50
    else:
        return 40


if __name__ == '__main__':
    vid = cv2.VideoCapture(0)

    while vid.isOpened():
        ret, frame = vid.read()

        # p = process_image(frame)
        # image = frame[360:720, 0:1280]
        print(return_num(frame))

        cv2.imshow('frame', frame)
        # cv2.imshow('cut', image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # g = np.uint8([[[255, 0, 0]]])
    # w = np.uint8([[[255, 255, 255]]])
    #
    # hsv_green = cv2.cvtColor(g, cv2.COLOR_BGR2HSV)
    # hsv_white = cv2.cvtColor(w, cv2.COLOR_BGR2HSV)
    #
    # print(hsv_white)
    # print(hsv_green)

    # hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    #
    # lower_blue = np.array([100, 100, 100], dtype=np.uint8)
    # upper_blue = np.array([220, 255, 255], dtype=np.uint8)
    #
    # lower_white = np.array([0, 0, 0], dtype=np.uint8)
    # upper_white = np.array([10, 0, 255], dtype=np.uint8)
    #
    # mask_b = cv2.inRange(hsv, lower_blue, upper_blue)
    # mask_w = cv2.inRange(hsv, lower_white, upper_white)
    # mask = cv2.bitwise_or(mask_b, mask_w)
    # res = cv2.bitwise_and(frame, frame, mask=mask)
    #
    # n_blue = cv2.countNonZero(mask_b)
    # n_white = cv2.countNonZero(mask_w)
    #
    # left_right_both(n_blue, n_white)
