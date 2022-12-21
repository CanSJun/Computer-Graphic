import numpy as np
import cv2
import time


def my_padding(src, filter):
    (h, w) = src.shape
    (h_pad, w_pad) = filter.shape
    h_pad = h_pad // 2
    w_pad = w_pad // 2
    padding_img = np.zeros((h + h_pad * 2, w + w_pad * 2))
    padding_img[h_pad:h + h_pad, w_pad:w + w_pad] = src

    """
    To do
    repetition padding
    :returns padding_img
    """
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

    # 직접 구현한 my_padding 함수를 이용
    pad_img = my_padding(src, filter)

    dst = np.zeros((h, w))
    for row in range(h):
        for col in range(w):
            dst[row, col] = np.sum(pad_img[row:row + f_h, col:col + f_w] * filter)

    return dst


def get_my_sobel():
    """
    To do
    Sobel_x, Sobel_y 구현
    :return: sobel_x, sobel_y
    """


    Blur1 = np.array([[-1,0,1]]) #2 * 2 array  1 2 1 , -1 0 1
    Der1 = np.array([[1],[2],[1]])

    Blur2 = np.array([[1,2,1]])
    Der2 = np.array([[-1],[0],[1]]) #2 * 2 array   -1 0 1 , 1 2 1

    sobel_x = np.dot(Der1, Blur1)
    sobel_y = np.dot(Der2, Blur2)

    return sobel_x, sobel_y


def my_get_Gaussian_filter(fshape, sigma=1):
    """
    To do
    Gaussian filter 구현
    :return: filter_gaus
    """
    (f_h, f_w) = fshape
    # 2차 Gaussian
    y, x = np.mgrid[-(f_h) // 2: (f_h // 2) + 1, -(f_w // 2): (f_w // 2) + 1]
    # 공식 그대로 대입
    filter_gaus = 1 / (2 * np.pi * sigma ** 2) * np.exp(-((x ** 2 + y ** 2 / (2 * sigma ** 2))))

    return filter_gaus


def GaussianFiltering(src, fshape=(3, 3), sigma=1):
    gaus = my_get_Gaussian_filter(fshape, sigma)
    dst = my_filtering(src, gaus)
    return dst


def find_local_maxima(src, ksize):
    (h, w) = src.shape
    pad_img = np.zeros((h + ksize, w + ksize))
    pad_img[ksize // 2:h + ksize // 2, ksize // 2:w + ksize // 2] = src
    dst = np.zeros((h, w))

    for row in range(h):
        for col in range(w):
            max_val = np.max(pad_img[row: row + ksize, col:col + ksize])
            if max_val == 0:
                continue
            if src[row, col] == max_val:
                dst[row, col] = src[row, col]

    return dst


def calc_derivatives(src):
    """
    #ToDo
    3x3 sobel 필터를 사용해서 Ix Iy 구하기
    :param src: 입력 이미지 (흑백)
    :return: Ix, Iy
    """
    # calculate Ix, Iy
    sobel_x, sobel_y = get_my_sobel()
    # 소벨 필터를 이용해서 x축 미분값 이미지 필터링값 y축 값의 이미지 필터링값을 구한다
    Ix = my_filtering(src, sobel_x)
    Iy = my_filtering(src, sobel_y)
    return Ix, Iy


def Faster_HarrisDetector(src, gaus_filter_size=3, gaus_sigma=1, alpha=0.04, threshold_rate=0.01):
    (h, w) = src.shape

    """
    # ToDo
    Ix 및 Iy 구하기
    """
    # calculate Ix, Iy
    Ix, Iy = calc_derivatives(src)

    # 0 ~ 1 사이의 값으로 변경 후 0 ~ 255로 변경 -> 결과가 잘 나왔는지 확인하기위해서
    dst_x_Norm = ((Ix - np.min(Ix)) / np.max(Ix - np.min(Ix)) * 255 + 0.5).astype(np.uint8)
    dst_y_Norm = ((Iy - np.min(Iy)) / np.max(Iy - np.min(Iy)) * 255 + 0.5).astype(np.uint8)
    cv2.imshow('dst_x_Norm', dst_x_Norm)
    cv2.imshow('dst_y_Norm', dst_y_Norm)
    cv2.waitKey()
    cv2.destroyWindow('dst_x_Norm')
    cv2.destroyWindow('dst_y_Norm')

    """
    # ToDo
    IxIx : Ix의 제곱
    IyIy : Iy의 제곱
    IxIy : Ix 곱하기 Iy

    # 구하기
    """
    # Square of derivatives
    IxIx = Ix ** 2
    IyIy = Iy ** 2
    IxIy = Ix * Iy

    # 0 ~ 1 사이의 값으로 변경 후 0 ~ 255로 변경 -> 결과가 잘 나왔는지 확인하기위해서
    dst_IxIy_Norm = ((IxIy - np.min(IxIy)) / np.max(IxIy - np.min(IxIy)) * 255 + 0.5).astype(np.uint8)
    cv2.imshow('IxIx', IxIx)
    cv2.imshow('IyIy', IyIy)
    # cv2.imshow('IxIy', IxIy)
    cv2.imshow('dst_IxIy_Norm', dst_IxIy_Norm)
    cv2.waitKey()
    cv2.destroyWindow('IxIx')
    cv2.destroyWindow('IyIy')
    # cv2.destroyWindow('IxIy')
    cv2.destroyWindow('dst_IxIy_Norm')

    # Gaussian filter
    """
    #ToDo
    #가우시안 필터 적용하기
    #G_IxIx = IxIx에 가우시안 필터 적용
    #G_IyIy = IyIy에 가우시안 필터 적용
    #G_IxIy = IxIy에 가우시안 필터 적용    
    """

    G_IxIx = GaussianFiltering(IxIx)
    G_IyIy = GaussianFiltering(IyIy)
    G_IxIy = GaussianFiltering(IxIy)

    # 0 ~ 1 사이의 값으로 변경 후 0 ~ 255로 변경 -> 결과가 잘 나왔는지 확인하기위해서
    G_dst_IxIy_Norm = ((G_IxIy - np.min(G_IxIy)) / np.max(G_IxIy - np.min(G_IxIy)) * 255 + 0.5).astype(np.uint8)
    cv2.imshow('G_IxIx', G_IxIx)
    cv2.imshow('G_IyIy', G_IyIy)
    # cv2.imshow('G_IxIy', G_IxIy)
    cv2.imshow('G_dst_IxIy_Norm', G_dst_IxIy_Norm)
    cv2.waitKey()
    cv2.destroyWindow('G_IxIx')
    cv2.destroyWindow('G_IyIy')
    # cv2.destroyWindow('G_IxIy')
    cv2.destroyWindow('G_dst_IxIy_Norm')

    # Normal Harris 시간 측정
    start = time.perf_counter()
    """
    # ToDo
    # har(Response) 구하기
    # 실습 PPT 27 page 참고
    """
    # R = det M - alpha * trace m^2
    har = G_IxIx * G_IyIy - (G_IxIy ** 2) - alpha * ((G_IxIx + G_IyIy) ** 2)

    print('Normal Harris Time check : ', time.perf_counter() - start)

    # 0 ~ 1 사이의 값으로 변경 후 0 ~ 255로 변경 -> 결과가 잘 나왔는지 확인하기위해서
    G_dst_har_Norm = ((har - np.min(har)) / np.max(har - np.min(har)) * 255 + 0.5).astype(np.uint8)
    cv2.imshow('har before threshold', G_dst_har_Norm)

    # thresholding
    har[har < threshold_rate * np.max(har)] = 0

    # 0 ~ 1 사이의 값으로 변경 후 0 ~ 255로 변경 -> 결과가 잘 나왔는지 확인하기위해서
    G_dst_har_thresh_Norm = ((har - np.min(har)) / np.max(har - np.min(har)) * 255 + 0.5).astype(np.uint8)
    cv2.imshow('har after threshold', G_dst_har_thresh_Norm)
    # 주변에서 가장 큰 값만 남기고 나머지 지우기
    dst = find_local_maxima(har, 21)

    return dst


def Sliding_Window_HarrisDetector(src, gaus_filter_size=3, gaus_sigma=1, window_size=3, alpha=0.04,
                                  threshold_rate=0.01):
    (h, w) = src.shape

    """
    # ToDo
    Ix 및 Iy 구하기
    """
    Ix, Iy = calc_derivatives(src)

    """
    # ToDo
    IxIx : Ix의 제곱
    IyIy : Iy의 제곱
    IxIy : Ix 곱하기 Iy

    # 구하기
    """
    IxIx = Ix ** 2
    IyIy = Iy ** 2
    IxIy = Ix * Iy

    """
    #ToDo
    #가우시안 필터 적용하기
    #G_IxIx = IxIx에 가우시안 필터 적용
    #G_IyIy = IyIy에 가우시안 필터 적용
    #G_IxIy = IxIy에 가우시안 필터 적용    
    """
    G_IxIx = GaussianFiltering(IxIx)
    G_IyIy = GaussianFiltering(IyIy)
    G_IxIy = GaussianFiltering(IxIy)

    # Summation methods
    dst = np.zeros((h, w), dtype=np.float32)
    pad_G_IxIx = my_padding(G_IxIx, np.zeros((window_size, window_size)))
    pad_G_IxIy = my_padding(G_IxIy, np.zeros((window_size, window_size)))
    pad_G_IyIy = my_padding(G_IyIy, np.zeros((window_size, window_size)))

    """
    #ToDo
    # Summation 구현
    """
    # Summation 방법 시간 체크
    M = np.zeros((h, w,2,2))
    start = time.perf_counter()
    for row in range(h):
        for col in range(w):
            # 각 점에 대한 R = det M - ktrace^2(M) 을 구해준다/
            for wx in range(window_size):
                for wy in range(window_size):
                    # 𝑔(𝐼_𝑥^2 )𝑔(𝐼_𝑦^2 )𝑔(𝐼_𝑥 𝐼_𝑦 ) 의 모든 좌표를 돌면서 window 영역
                    # 내의 모든 점들의 합을 각각 구함
                    M[row,col,0,0] += pad_G_IxIx[row + wx, col + wy]
                    M[row,col,1,1] += pad_G_IyIy[row + wx, col + wy]
                    M[row,col,1,0] += pad_G_IxIy[row + wx, col + wy]
                    M[row,col,0,1] += pad_G_IxIy[row + wx, col + wy]
            det = (M[row,col,0,0] * M[row,col,1,1]) - (M[row,col,0,1] * M[row,col,1,0])
            trace = M[row,col,0,0] + M[row,col,1,1]
            response = det - (alpha*(trace**2))
            if response > threshold_rate:
                 dst[row, col] = response

    print('Sliding_Window Time check : ', time.perf_counter() - start)

    # 주변에서 가장 큰 값만 남기고 나머지 지우기
    dst = find_local_maxima(dst, 21)

    return dst


def main():


    src = cv2.imread('zebra.png')
    harris_img = src.copy()
    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.
    cv2.imshow('original', src)
    dst = Sliding_Window_HarrisDetector(gray, gaus_filter_size=3, gaus_sigma=1, alpha=0.04)
    dst = cv2.dilate(dst, None)
    interest_points = np.zeros((dst.shape[0], dst.shape[1], 3))
    interest_points[dst != 0] = [0, 0, 255]
    harris_img[dst != 0] = [0, 0, 255]
    cv2.imshow('Summation_interest points', interest_points)
    cv2.imshow('Summation_harris_img', harris_img)
    cv2.waitKey()
    cv2.destroyAllWindows()
 
    src = cv2.imread('zebra.png')
    harris_img = src.copy()
    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.
    dst = Faster_HarrisDetector(gray, gaus_filter_size=3, gaus_sigma=1, alpha=0.04)
    dst = cv2.dilate(dst, None)
    interest_points = np.zeros((dst.shape[0], dst.shape[1], 3))
    interest_points[dst != 0] = [0, 0, 255]
    harris_img[dst != 0] = [0, 0, 255]
    cv2.imshow('normal_interest points', interest_points)
    cv2.imshow('normal_harris_img', harris_img)
    cv2.waitKey()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()