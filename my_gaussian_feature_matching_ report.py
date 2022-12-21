import numpy as np
import cv2
import random
from cv2 import DMatch

def my_feature_matching(des1, des2):
    ##########################################
    # TODO Brute-Force Feature Matching 구현
    # matches: cv2.DMatch의 객체를 저장하는 리스트
    # cv2.DMatch의 배열로 구성
    # cv2.DMatch:
    # trainIdx: img1의 kp1, des1에 매칭되는 index
    # queryIdx: img2의 kp2, des2에 매칭되는 index
    # kp1[queryIdx]와 kp2[trainIdx]는 매칭된 점
    # return 값 : matches
    ##########################################
    length1 = len(des1)
    length2 = len(des2)
    matches = list()
    for x in range(length1):
        min_L = 1024
        for y in range(length2):
            dist = L2_distance(des1[x], des2[y])
            
            #만약에 min_L가 현재 dist 계산결과 보다 크다면?
            if min_L> dist:
                qidx = x
                tidx = y
                min_L = dist

        #첫번쨰 인자 : 원본 이미지의 특징점
        #두번째 인자 : 목표 이미지의 특징점
        # 세번째 인자는 무조건 0
        # 4번쨰 인자는 L2 distance의 최솟값
        matches.append(DMatch(qidx, tidx, 0,  min_L))

    return matches

def get_matching_keypoints(img1, img2, keypoint_num):
    '''
    :param img1: 변환시킬 이미지
    :param img2: 변환 목표 이미지
    :param keypoint_num: 추출한 keypoint의 수
    :return: img1의 특징점인 kp1, img2의 특징점인 kp2, 두 특징점의 매칭 결과
    '''
    sift = cv2.SIFT_create(keypoint_num)

    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    ##########################################
    # TODO Brute-Force Feature Matching 구현
    ##########################################

    my_matches = my_feature_matching(des1, des2)

    ############################################################
    # TODO TEST 내장 함수를 사용한 것과 직접 구현 것의 결과 비교
    # 다음 3가지 중 하나라도 통과하지 못하면 잘못 구현한것으로 판단하여 프로그램 종료
    # 오류가 없다면 "매칭 오류 없음" 출력
    bf = cv2.BFMatcher_create(cv2.NORM_L2)
    matches = bf.match(des1, des2)
    # 1. 매칭 개수 비교
    assert len(matches) == len(my_matches)
    # 2. 매칭 점 비교
    for i in range(len(matches)):
        if (matches[i].trainIdx != my_matches[i].trainIdx) \
                or (matches[i].queryIdx !=
                                        my_matches[i].queryIdx):
            print("matching error")
            return

    # 3. distance 값 비교
    for i in range(len(matches)):
        if int(matches[i].distance) != int(my_matches[i].distance):
            print("distance calculation error")
            return

    print("매칭 오류 없음")
    ##########################################################

    # DMatch 객체에서 distance 속성 값만 가져와서 정렬
    my_matches = sorted(my_matches, key=lambda x: x.distance)
    # 매칭된 점들 중 20개만 가져와서 표시
    result = cv2.drawMatches(img1, kp1, img2, kp2, my_matches[:20], outImg=None, flags=2)

    cv2.imshow('My BF matching result', result)
    cv2.waitKey()
    cv2.destroyAllWindows()

    return kp1, kp2, my_matches

def L2_distance(vector1, vector2):
    return np.sqrt(np.sum((vector1 - vector2)**2))
def fit_coordinates(src, M):

    h, w, _ = src.shape
    cor_transform = []

    for row in range(h + 1):
        for col in range(w + 1):
            P = np.array([
                [col],
                [row],
                [1]
            ])

            P_dst = np.dot(M, P)
            dst_col = P_dst[0][0]
            dst_row = P_dst[1][0]
            cor_transform.append((dst_row, dst_col))

    cor_transform = list(set(cor_transform))  # 중복제거
    cor_transform = np.array(cor_transform)

    row_max = np.max(cor_transform[:, 0])
    row_min = np.min(cor_transform[:, 0])
    col_max = np.max(cor_transform[:, 1])
    col_min = np.min(cor_transform[:, 1])

    return row_max, row_min, col_max, col_min

def backward(src, M):

    row_max, row_min, col_max, col_min = fit_coordinates(src, M)
    h_ = round(row_max - row_min)
    w_ = round(col_max - col_min)

    h, w, c = src.shape
    M_inv = np.linalg.inv(M)
    dst = np.zeros((h_, w_, c))
    for row in range(h_):
        for col in range(w_):
            dst_p = np.array([
                    [col + col_min],
                    [row + row_min],
                    [1]
                ])
                # orig
            P = np.dot(M_inv, dst_p)
            x = P[0,0]
            y = P[1,0]

            src_col_right = int(np.ceil(x))
            src_col_left = int(x)

            src_row_bottom = int(np.ceil(y))
            src_row_top = int(y)

            if src_col_right >= w or src_row_bottom >= h or src_col_left < 0 or src_row_top < 0:
                continue
            s = x - src_col_left
            t = y - src_row_top

            intensity = (1 - s) * (1 - t) * src[src_row_top, src_col_left, :] \
                        + s * (1 - t) * src[src_row_top, src_col_right, :] \
                        + (1 - s) * t * src[src_row_bottom, src_col_left, :] \
                        + s * t * src[src_row_bottom, src_col_right, :]

            dst[row, col, :] = intensity


    dst = dst.astype(np.uint8)
    return dst

def G_backward(src,src2, M):

    #변환된 좌표의 주변 픽셀에 대해 bilinear interpolation수행

    #f  원본 이미지 픽셀값,  g 목표 이미지 픽셀값, G = 가우시안 필터값
    #가우시안은 3*3 size

    h, w, c = src.shape
    h2,w2,c2 = src2.shape
    filter_ = my_get_Gaussian_filter((3, 3), 3)
    padding = my_padding(src, filter_)
    M_inv = np.linalg.inv(M)

    dst = np.zeros((h2, w2, c2))

    for row in range(h2):
        for col in range(w2):
            dst_p = np.array([
                    [col],
                    [row],
                    [1]
                ])
            P = np.dot(M_inv, dst_p)
            x = P[0,0]
            y = P[1,0]
            result = 0
            src_col_right = int(np.ceil(x))
            src_col_left = int(x)

            src_row_bottom = int(np.ceil(y))
            src_row_top = int(y)

            if src_col_right >= w or src_row_bottom >= h or src_col_left < 0 or src_row_top < 0:
                continue
            for i in range(3):
                for j in range(3):


                    src_col_right = int(np.ceil(x + i))
                    src_col_left = int(x + i)

                    src_row_bottom = int(np.ceil(y + j))
                    src_row_top = int(y + j)

                    s = x + i - src_col_left
                    t = y + j - src_row_top

                    intensity = (1 - s) * (1 - t) * padding[src_row_top, src_col_left, :] \
                                + s * (1 - t) * padding[src_row_top, src_col_right, :] \
                                + (1 - s) * t * padding[src_row_bottom, src_col_left, :] \
                                + s * t * padding[src_row_bottom, src_col_right, :]
                    result += intensity * filter_[i,j]
            dst[row, col, :] = result


    dst = dst.astype(np.uint8)
    return dst

def my_ls(matches, kp1, kp2):

    A = []
    b = []
    for idx, match in enumerate(matches):
        trainInd = match.trainIdx
        queryInd = match.queryIdx

        x, y = kp1[queryInd].pt
        x_prime, y_prime = kp2[trainInd].pt
        #키포인트로 변환시켜주는 행렬을 찾아야함.

        A.append([x,y,1,0,0,0])
        A.append([0,0,0,x,y,1])

        b.append([x_prime])
        b.append([y_prime])

    A = np.array(A)
    b = np.array(b)

    try:
        AT_A = np.dot(A.T,A)
        AT_B = np.dot(A.T,b)
        #AT_A의 역행렬을 구해야함 => np.linalg.inv(x) => 역행렬
        X = np.dot(np.linalg.inv(AT_A), AT_B)
    except:
        print('can\'t calculate np.linalg.inv((np.dot(A.T, A)) !!!!!')
        X = None
    return X
def feature_matching_RANSAC(img1, img2, keypoint_num=None, iter_num=500, threshold_distance=5):
    '''
    :param img1: 변환시킬 이미지
    :param img2: 변환 목표 이미지
    :param keypoint_num: sift에서 추출할 keypoint의 수
    :param iter_num: RANSAC 반복횟수
    :param threshold_distance: RANSAC에서 inlier을 정할때의 거리 값
    :return: RANSAC을 이용하여 변환 된 결과
    '''
    kp1, kp2, matches = get_matching_keypoints(img1, img2, keypoint_num)

    matches_shuffle = matches.copy()
    #########################################################
    # TODO RANSAC 구현하기
    # inliers : inliers의 개수를 저장
    # M_list : 랜덤하게 뽑은 keypoint 3개의 좌표로 구한 변환 행렬
    # 절차
    # 1. 랜덤하게 3개의 matches point를 뽑아냄
    # 2. 1에서 뽑은 matches를 가지고 Least square를 사용한 affine matrix M을 구함
    # 3. 2에서 구한 M을 가지고 모든 matches point와 연산하여 inlier의 개수를 파악
    # 4. M을 사용하여 변환된 좌표와 SIFT를 통해 얻은 변환된 좌표와의 L2 Distance를 구함
    # 5. 거리 값이 threshold_distance보다 작으면 inlier로 판단
    # 6. iter_num만큼 반복하여 가장 많은 inlier를 보유한 M을 Best_M으로 설정
    ##########################################################
    inliers = []
    M_list = []
    for i in range(iter_num):
        print('\rcalculate RANSAC ... %d ' % (int((i + 1) / iter_num * 100)) + '%', end='\t')
        random.shuffle(matches_shuffle)
        three_points = matches_shuffle[:3]

        X = my_ls(three_points, kp1, kp2)
        if X is None:
            continue

        M = np.array([[X[0][0], X[1][0], X[2][0]],
                      [X[3][0], X[4][0], X[5][0]],
                      [0, 0, 1]])
        M_list.append(M)

        count_inliers = 0
        for idx, match in enumerate(matches):
            trainInd = match.trainIdx
            queryInd = match.queryIdx

            kp1_x, kp1_y = kp1[queryInd].pt
            kp2_x, kp2_y = kp2[trainInd].pt

            p = np.array([[kp1_x],
                          [kp1_y],
                          [1]]
                         )
            p_ = np.dot(M, p)
            x_p = p_[0][0]
            y_p = p_[1][0]
            distance = L2_distance(np.array([x_p, y_p]), np.array([kp2_x, kp2_y]))
            # check threshold_distance
            if (distance < threshold_distance):
                count_inliers += 1
        inliers.append(count_inliers)

    # iter_num만큼 반복하여 가장 많은 inlier를 보유한 M을 Best_M으로 설정
    best_M = M_list[np.argmax(inliers)]
    result = backward(img1, best_M)
    return result.astype(np.uint8)

#5주차떄 구현 하였던 구아시안 필터
def my_padding(src, filter):


    (h, w, c) = src.shape
    (h_pad, w_pad) = filter.shape
    h_pad = h_pad // 2
    w_pad = w_pad // 2
    padding_img = np.zeros((h + h_pad * 2, w + w_pad * 2,c))
    padding_img[h_pad:h + h_pad, w_pad:w + w_pad,:] = src
    # up
    padding_img[:h_pad, w_pad:w_pad + w,:] = src[0, :,:]
    # down
    padding_img[h_pad + h:, w_pad:w_pad + w,:] = src[h - 1, :,:]
    # left
    padding_img[:, :w_pad,:] = padding_img[:, w_pad:w_pad + 1,:]
    # right
    padding_img[:, w_pad + w:,:] = padding_img[:, w_pad + w - 1:w_pad + w,:]


    return padding_img

def my_get_Gaussian_filter(fshape, sigma=1):
    (f_h, f_w) = fshape
    # 2차 Gaussian
    y, x = np.mgrid[-(f_h // 2): (f_h // 2) + 1, -(f_w // 2): (f_w // 2) + 1]
    # 공식 그대로 대입
    filter_gaus = 1 / (2 * np.pi * sigma ** 2) * np.exp(-((x ** 2 + y ** 2 / (2 * sigma ** 2))))
    filter_gaus /= np.sum(filter_gaus)
    return filter_gaus

def feature_matching_gaussian(img1, img2, keypoint_num=None, iter_num=500, threshold_distance=5):
    '''
    :param img1: 변환시킬 이미지
    :param img2: 변환 목표 이미지
    :param keypoint_num: sift에서 추출할 keypoint의 수
    :param iter_num: RANSAC 반복횟수
    :param threshold_distance: RANSAC에서 inlier을 정할때의 거리 값
    :return: RANSAC을 이용하여 변환 된 결과
    '''


    kp1, kp2, matches = get_matching_keypoints(img1, img2, keypoint_num)

    matches_shuffle = matches.copy()
    inliers = []
    M_list = []
    for i in range(iter_num):
        print('\rcalculate gaussian ... %d ' % (int((i + 1) / iter_num * 100)) + '%', end='\t')
        random.shuffle(matches_shuffle)
        three_points = matches_shuffle[:3]

        X = my_ls(three_points, kp1, kp2)
        if X is None:
            continue

        M = np.array([[X[0][0], X[1][0], X[2][0]],
                      [X[3][0], X[4][0], X[5][0]],
                      [0, 0, 1]])
        M_list.append(M)

        count_inliers = 0
        for idx, match in enumerate(matches):
            trainInd = match.trainIdx
            queryInd = match.queryIdx

            kp1_x, kp1_y = kp1[queryInd].pt
            kp2_x, kp2_y = kp2[trainInd].pt

            p = np.array([[kp1_x],
                           [kp1_y],
                           [1]]
                          )
            p_ = np.dot(M, p)
            x_p = p_[0][0]
            y_p = p_[1][0]
            distance = L2_distance(np.array([x_p, y_p]), np.array([kp2_x, kp2_y]))
            # check threshold_distance
            if(distance < threshold_distance):
                count_inliers += 1
        inliers.append(count_inliers)

    #iter_num만큼 반복하여 가장 많은 inlier를 보유한 M을 Best_M으로 설정
    best_M =  M_list[np.argmax(inliers)]
    result = G_backward(img1, img2, best_M)
    return result.astype(np.uint8)

def main():
    src = cv2.imread('Lena.png')
    src = cv2.resize(src, None, fx=0.5, fy=0.5)
    src2 = cv2.imread('Lena_transforms.png')

    result = feature_matching_RANSAC(src, src2)
    gussian_result = feature_matching_gaussian(src, src2)

    cv2.imshow("input", src)
    cv2.imshow("goal", src2)
    cv2.imshow('result', result)
    cv2.imshow('gaussian result', gussian_result)
    cv2.waitKey()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
