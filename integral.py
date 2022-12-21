import numpy as np
import cv2
import time

def my_padding(src, filter):
    (h, w) = src.shape
    if isinstance(filter, tuple):
        (h_pad, w_pad) = filter
    else:
        (h_pad, w_pad) = filter.shape
    h_pad = h_pad // 2
    w_pad = w_pad // 2
    padding_img = np.zeros((h+h_pad*2, w+w_pad*2))
    padding_img[h_pad:h+h_pad, w_pad:w+w_pad] = src

    # repetition padding
    # up
    padding_img[:h_pad, w_pad:w_pad + w] = src[0, :]
    # down
    padding_img[h_pad + h:, w_pad:w_pad + w] = src[h - 1, :]
    # left
    padding_img[:, :w_pad] = padding_img[:, w_pad:w_pad + 1]
    # right
    padding_img[:, w_pad + w:] = padding_img[:, w_pad + w - 1:w_pad + w]

    return padding_img

def my_filtering(src, filter):
    (h, w) = src.shape
    (f_h, f_w) = filter.shape

    #filter 확인
    #print('<filter>')
    #print(filter)

    # 직접 구현한 my_padding 함수를 이용
    pad_img = my_padding(src, filter)

    dst = np.zeros((h, w))
    for row in range(h):
        for col in range(w):
            dst[row, col] = np.sum(pad_img[row:row + f_h, col:col + f_w] * filter)

    return dst

def get_my_sobel():
    sobel_x = np.dot(np.array([[1], [2], [1]]), np.array([[-1, 0, 1]]))
    sobel_y = np.dot(np.array([[-1], [0], [1]]), np.array([[1, 2, 1]]))
    return sobel_x, sobel_y

def calc_derivatives(src):
    # calculate Ix, Iy
    sobel_x, sobel_y = get_my_sobel()
    Ix = my_filtering(src, sobel_x)
    Iy = my_filtering(src, sobel_y)
    return Ix, Iy

def get_integral_image(src):
    ##########################################################################
    # ToDo
    # 원본 이미지에서 integral image를 구하는 함수
    # 반복문 보다는 Numpy를 활용하여 쉽게 구현할 수 있으니 되도록이면 Numpy 사용을 지향
    # 실습 설명을 들으면 보다 쉽게 구현 가능
    ##########################################################################

    assert len(src.shape) == 2
    h, w = src.shape
    dst = np.zeros(src.shape)

    # 31 2 4            31  33  37
    for x in range(h):
        dst[x, 0] = np.sum(src[0:x+1, 0]) #누적합을 쭉 구한다.

    # 세로
    for y in range(w) :
        dst[0, y] = np.sum(src[0, 0:y + 1]) # 누적합을 쭉 구한다
    # 31 2 4            31  33  37
    # 12 26 9           43  71  84
    # 13 17 21          56 101 135
    #

    #135 = 21 + 84 + 101 - 71
    for x in range(1, h):
        for y in range(1, w):
            dst[x,y] = src[x,y] + dst[x -1, y] + dst[x, y - 1] - dst[x - 1, y - 1]

    return dst

def calc_local_integral_value(src, left_top, right_bottom):
    ##########################################################################
    # ToDo
    # integral image를 통해서 해당 영역의 필터의 합을 구하는 함수
    # 실습 설명을 들으면 보다 쉽게 구현 가능
    ##########################################################################

    #tuple 형식
    x, y = left_top
    x1,y1 = right_bottom

    #x -1 ,y -1   x1,y
    #x,y1 ,      x1,y1

    # 31   33   37    70    75   111
    # 43   71   84    127   161   222
    # 56   101  135   200   254   333
    # 80   148  197   278   346   444
    # 110  186  263   371   450   555
    # 111  222  333   444   555   666
    # 555 + 84 - 222 - 263 = 154

    x4 = src[x - 1, y - 1]
    x3 = src[x - 1, y1]
    x2 = src[x1, y - 1]
    x1 = src[x1, y1]
    if x == 0:
        x4 = 0
        x2 = 0
    if y == 0:
        x4 = 0
        x3 = 0
    result = x1 + x4 -x3 -x2
    return result

def find_local_maxima(src, ksize):
    (h, w) = src.shape
    pad_img = np.zeros((h+ksize, w+ksize))
    pad_img[ksize//2:h+ksize//2, ksize//2:w+ksize//2] = src
    dst = np.zeros((h, w))

    for row in range(h):
        for col in range(w):
            max_val = np.max(pad_img[row : row+ksize, col:col+ksize])
            if max_val == 0:
                continue
            if src[row, col] == max_val:
                dst[row, col] = src[row, col]

    return dst

def calc_M_harris(IxIx, IxIy, IyIy, fsize = 5):
    assert IxIx.shape == IxIy.shape and IxIx.shape == IyIy.shape
    h, w = IxIx.shape
    M = np.zeros((h, w, 2, 2))
    IxIx_pad = my_padding(IxIx, (fsize, fsize))
    IxIy_pad = my_padding(IxIy, (fsize, fsize))
    IyIy_pad = my_padding(IyIy, (fsize, fsize))

    """for row in range(h):
        for col in range(w):
            M[row, col, 0, 0] = np.sum(IxIx_pad[row:row+fsize, col:col+fsize])
            M[row, col, 0, 1] = np.sum(IxIy_pad[row:row+fsize, col:col+fsize])
            M[row, col, 1, 0] = M[row, col, 0, 1]
            M[row, col, 1, 1] = np.sum(IyIy_pad[row:row+fsize, col:col+fsize])"""
    ##########################################################################
    # ToDo
    # integral을 사용하지 않고 covariance matrix 구하기
    # 4중 for문을 채워서 완성하기
    # 실습 시간에 했던 내용을 생각하면 금방 해결할 수 있음
    # 위 주석을 활용하여 구현할 것 위에 코드가 정답이지만 그대로 쓰면 감점 예정.
    ##########################################################################
    for row in range(h):
        for col in range(w):

            for f_row in range(fsize):
                for f_col in range(fsize):
                    M[row, col, 0, 0] += IxIx_pad[row + f_row, col + f_col]
                    M[row, col, 0, 1] += IxIy_pad[row + f_row, col + f_col]
                    M[row, col, 1, 0]  = M[row, col, 0, 1]
                    M[row, col, 1, 1] += IyIy_pad[row + f_row, col + f_col]

    return M

def harris_detector(src, k = 0.04, threshold_rate = 0.01, fsize=5):
    harris_img = src.copy()
    h, w, c = src.shape
    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY) / 255.
    # calculate Ix, Iy
    Ix, Iy = calc_derivatives(gray)

    # Square of derivatives
    IxIx = Ix**2
    IyIy = Iy**2
    IxIy = Ix * Iy

    start = time.perf_counter()  # 시간 측정 시작
    M_harris = calc_M_harris(IxIx, IxIy, IyIy, fsize)
    end = time.perf_counter()  # 시간 측정 끝
    print('M_harris time : ', end-start)

    R = np.zeros((h, w))
    for row in range(h):
        for col in range(w):
            ##########################################################################
            # ToDo
            # det_M 계산
            # trace_M 계산
            # R 계산 Harris 방정식 구현
            ##########################################################################
            det_M = M_harris[row, col, 0,0] * M_harris[row, col, 1,1] - M_harris[row, col, 0,1] * M_harris[row, col, 1, 0]
            trace_M = M_harris[row, col, 0,0] + M_harris[row, col, 1,1]
            R[row, col] = det_M - k * (trace_M **2)

    # thresholding
    R[R < threshold_rate * np.max(R)] = 0

    R = find_local_maxima(R, 21)
    R = cv2.dilate(R, None)

    harris_img[R != 0]=[0, 0, 255]

    return harris_img

def calc_M_integral(IxIx_integral, IxIy_integral, IyIy_integral, fsize = 5):
    assert IxIx_integral.shape == IxIy_integral.shape and IxIx_integral.shape == IyIy_integral.shape
    h, w = IxIx_integral.shape
    M = np.zeros((h, w, 2, 2))

    IxIx_integral_pad = my_padding(IxIx_integral, (fsize, fsize))
    IxIy_integral_pad = my_padding(IxIy_integral, (fsize, fsize))
    IyIy_integral_pad = my_padding(IyIy_integral, (fsize, fsize))

    ##########################################################################
    # ToDo
    # integral 값을 이용하여 covariance matrix 구하기
    # 실습때 알려드린 integral의 지역 해당 영역 값을 구하는 함수를 완성하여 사용하면 쉽게 구할 수 있음
    ##########################################################################

    for row in range(h):
        for col in range(w):
            #인테그랄 이미지 ixix , iyiy 인테그랄 이미지, ix iy 인테그랄 이미지를 구해서 전체 합
            M[row,col,0,0] = calc_local_integral_value(IxIx_integral_pad, (row,col),(row + fsize - 1, col + fsize - 1))
            M[row,col,1,0] = calc_local_integral_value(IxIy_integral_pad, (row,col),(row + fsize - 1, col + fsize - 1))
            M[row,col,0,1] = calc_local_integral_value(IyIy_integral_pad, (row,col),(row + fsize - 1, col + fsize - 1))
            M[row,col,1,1] = calc_local_integral_value(IxIy_integral_pad, (row,col),(row + fsize - 1, col + fsize - 1))
    return M

def harris_detector_integral(src, k = 0.04, threshold_rate = 0.01, fsize=5):
    harris_img = src.copy()
    h, w, c = src.shape
    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY) / 255.
    # calculate Ix, Iy
    Ix, Iy = calc_derivatives(gray)

    # Square of derivatives
    IxIx = Ix**2
    IyIy = Iy**2
    IxIy = Ix * Iy

    start = time.perf_counter()  # 시간 측정 시작
    IxIx_integral = get_integral_image(IxIx)
    IxIy_integral = get_integral_image(IxIy)
    IyIy_integral = get_integral_image(IyIy)
    end = time.perf_counter()  # 시간 측정 끝
    print('make integral image time : ', end-start)

    start = time.perf_counter()  # 시간 측정 시작
    M_integral = calc_M_integral(IxIx_integral, IxIy_integral, IyIy_integral, fsize)
    end = time.perf_counter()  # 시간 측정 끝
    print('M_harris integral time : ', end-start)

    R = np.zeros((h, w))
    for row in range(h):
        for col in range(w):
            ##########################################################################
            # ToDo
            # det_M 계산
            # trace_M 계산
            # R 계산 Harris 방정식 구현
            ##########################################################################
            det_M = M_integral[row, col, 0,0] * M_integral[row,col, 1,1] - M_integral[row,col, 0, 1] * M_integral[row,col,1,0]
            trace_M = M_integral[row, col, 0, 0] + M_integral[row,col, 1,1]
            R[row, col] = det_M - k * trace_M**2

    # thresholding
    R[R < threshold_rate * np.max(R)] = 0

    R = find_local_maxima(R, 21)
    R = cv2.dilate(R, None)

    harris_img[R != 0]=[0, 0, 255]

    return harris_img

def integral_function_test():
    src = np.array([[31, 2, 4, 33, 5, 36],
                    [12, 26, 9, 10, 29, 25],
                    [13, 17, 21, 22, 20, 18],
                    [24, 23, 15, 16, 14, 19],
                    [30, 8, 28, 27, 11, 7],
                    [1, 35, 34, 3, 32, 6]])

    integral_src = get_integral_image(src)
    row, col = 2, 3
    b = 3

    sum = 0
    ## 22 + 20 + 18 + 16 + 14 + 19 + 27 + 11 + 7
    for i in range(row, row + b):
        for j in range(col, col + b):
            sum += src[i, j]

    ## 84 + 555 - 222 - 263
    integral_sum = calc_local_integral_value(integral_src, (row, col), (row + b - 1, col + b - 1))

    print("image: \n{}".format(src))
    print("integral image: \n{}".format(integral_src))

    print("sum [{}:{}, {}:{}]".format(row, row + b, col, col + b))
    print("image: {}".format(sum))
    print("integral image: {}".format(integral_sum))

def main():
    # Integral test 하려면 아래 함수 주석 해제
    integral_function_test()

    src = cv2.imread('zebra.png') # shape : (552, 435, 3)
    print('start!')
    print('src.shape : ', src.shape)
    fsize = 5
    print('fsize : ', fsize)
    harris_img = harris_detector(src, fsize = fsize)
    harris_integral_img = harris_detector_integral(src, fsize = fsize)
    cv2.imshow('harris_img ' + '202004183' , harris_img)
    cv2.imshow('harris_integral_img ' + '202004183' , harris_integral_img)
    cv2.waitKey()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()