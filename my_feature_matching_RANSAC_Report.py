import numpy as np
import cv2
import random
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

    ##############################
    # TODO Backward 방식 구현
    # 실습 참고
    ##############################
    row_max, row_min, col_max, col_min = fit_coordinates(src, M)
    h_ = round(row_max - row_min)
    w_ = round(col_max - col_min)

    h, w, c = src.shape
    M_inv = np.linalg.inv(M)
    dst = np.zeros((h_, w_, c))
    #보정된 좌표계에서 for로 쭉 돌아버린다. 쭉 돌고 col min 더해주고 row min을 더해준다.

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

def my_ls(matches, kp1, kp2):
    '''
    :param matches: keypoint matching 정보
    :param kp1: keypoint 정보.
    :param kp2: keypoint 정보2.
    :return: X : 위의 정보를 바탕으로 Least square 방식으로 구해진 Affine 변환 matrix의 요소 [a, b, c, d, e, f].T
    '''
    ##############################
    # TODO Least square 구현
    # A : 원본 이미지 좌표 행렬
    # b : 변환된 좌표 벡터
    # X : 구하고자 하는 Unknown transformation
    ##############################
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

    bf = cv2.BFMatcher(cv2.DIST_L2)

    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)

    """
    SIFT에서 특징점들에 대한 매칭 결과 확인하고 싶으면 주석 풀어서 확인
    result = cv2.drawMatches(img1, kp1, img2, kp2, matches[:20], outImg=None, flags=2)

    cv2.imshow('matching result', result)
    cv2.waitKey()
    cv2.destroyAllWindows()
    """

    """
    matches: List[cv2.DMatch]
    cv2.DMatch의 배열로 구성

    matches[i]는 distance, imgIdx, queryIdx, trainIdx로 구성됨
    trainIdx: 매칭된 img1에 해당하는 index
    queryIdx: 매칭된 img2에 해당하는 index

    kp1[queryIdx]와 kp2[trainIdx]는 매칭된 점
    """
    return kp1, kp2, matches

def feature_matching(img1, img2, keypoint_num=None):
    kp1, kp2, matches = get_matching_keypoints(img1, img2, keypoint_num)

    X = my_ls(matches, kp1, kp2)

    ##########################################
    # TODO Unknown transformation Matrix 구하기
    # M : [ [ a b c]     3 x 3 행렬
    #       [ d e f]
    #        [0 0 1 ]]
    ##########################################

    #6개 변수 예측을 하여야한다.. 변수가 6개 있으니깐.. 식 6개가 필요한데..
    #하나의 키포인트 좌표내에서 식이 두개가 생김.. 랜덤하게 샘플링을 하여야하니 최소로 식 6개를 만들어야 하니
    #매칭이 되는 포인트에 대해 식이 두개 나오니 총 3개 키포인트를 뽑아서 하면 변환 행렬을 구할 수 있다..
    # x'' = ax + by +c , y'' = dx + ey + f ...
    # [a b c]
    # [d e f]
    # 3*3 맞춰주기 [ 0 0 1 ]
    M = np.array([[X[0][0], X[1][0], X[2][0]],
                  [X[3][0], X[4][0], X[5][0]],
                  [0, 0, 1]])
    result = backward(img1, M)
    return result.astype(np.uint8)

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
            if(distance < threshold_distance):
                count_inliers += 1
        inliers.append(count_inliers)

    #iter_num만큼 반복하여 가장 많은 inlier를 보유한 M을 Best_M으로 설정
    best_M =  M_list[np.argmax(inliers)]
    result = backward(img1, best_M)
    return result

def L2_distance(vector1, vector2):
    ##########################################
    # TODO L2 Distance 구하기
    ##########################################
    return np.sqrt(np.sum((vector1 - vector2)**2))

def main():
    src = cv2.imread('./Lena.png')
    src2 = cv2.imread('Lena_transforms.png')

    result_RANSAC = feature_matching_RANSAC(src, src2)
    result_LS = feature_matching(src, src2)
    cv2.imshow('input', src)
    cv2.imshow('result RANSAC 202004183', result_RANSAC)
    cv2.imshow('result LS 202004183', result_LS)
    cv2.imshow('goal', src2)
    cv2.waitKey()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
