import cv2 as cv
import numpy as np
import os


def getHomography(query_kp,train_kp,good_match,img1,img2,min_match_count=10):
    # calculate homography matrix and show match result
    if len(good_match) > min_match_count:
        src_pts = np.float32([query_kp[m.queryIdx].pt for m in good_match]).reshape(-1, 1, 2)
        dst_pts = np.float32([train_kp[m.trainIdx].pt for m in good_match]).reshape(-1, 1, 2)
        M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)
        matchesMask = mask.ravel().tolist()
        h, w = img1.shape
        pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
        dst = cv.perspectiveTransform(pts, M)
        img2 = cv.polylines(img2, [np.int32(dst)], True, 255, 3, cv.LINE_AA)
    else:
        print("Not enough matches are found - {}/{}".format(len(good_match), min_match_count))
        matchesMask = None

    # print homography matrix and print number of match
    print('matrix',M)
    print('match points after homography',sum(matchesMask))

    # draw inliers matches after homography
    draw_params = dict(matchColor=(0, 255, 0),  # draw matches in green color
                       singlePointColor=None,
                       matchesMask=matchesMask,  # draw only inliers
                       flags=2)
    all_match = cv.drawMatches(img1, query_kp, img2, train_kp, good_match, None, **draw_params)

    # find top 10 match and show image
    project_src = cv.perspectiveTransform(src_pts,M)
    diff = project_src-dst_pts
    diff = diff.reshape(-1,2)
    distance = np.sum(diff*diff,axis=1)
    topindex = sorted(range(len(distance)), key= lambda i:distance[i])[0:10]
    topmask = np.zeros(len(matchesMask))
    for i in topindex:
        topmask[i] = 1
    # draw inliers
    draw_params = dict(matchColor=(0, 255, 0),  # draw matches in green color
                       singlePointColor=None,
                       matchesMask=topmask,  # draw only inliers
                       flags=2)
    top_match = cv.drawMatches(img1, query_kp, img2, train_kp, good_match, None, **draw_params)
    return all_match, top_match


def main():
    np.set_printoptions(precision=3)

    img_dir = './HW3_Data/'
    img_list = os.listdir(img_dir)
    dst_list = img_list[0:2]
    src_list = img_list[2:]
    sift = cv.SIFT_create()
    bf = cv.BFMatcher()
    d = 0
    for dst_name in dst_list:
        dst_path = img_dir + dst_name
        dst = cv.imread(dst_path)
        gray_dst = cv.cvtColor(dst, cv.COLOR_BGR2GRAY) # train image
        # find key point and descriptors with SIFT
        dst_kp, dst_des = sift.detectAndCompute(gray_dst, None)
        # show SIFT result
        dst = cv.drawKeypoints(gray_dst, dst_kp, dst,
                               flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        cv.imwrite('sift'+str(len(dst_kp))+dst_name, dst)
        for src_name in src_list:
            src_path = img_dir + src_name
            src = cv.imread(src_path)
            gray_src = cv.cvtColor(src, cv.COLOR_BGR2GRAY) # query image
            # find key point and descriptors with SIFT
            src_kp, src_des = sift.detectAndCompute(gray_src, None)
            # show SIFT result
            src = cv.drawKeypoints(gray_src, src_kp, src,
                                   flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            cv.imwrite('sift'+str(len(src_kp)) + src_name, src)
            # run matcher
            matches = bf.knnMatch(src_des, dst_des, k=2)
            # Apply ratio test save good matches with low ratio rate
            good = []
            for m, n in matches:
                if m.distance < 0.75 * n.distance:
                    good.append(m)
            print("# of match:",len(good))
            # sort match result wrt distance
            good = sorted(good, key=lambda x: x.distance)
            # Draw top 20 matches and show match result.
            match_result = cv.drawMatches(gray_src, src_kp, gray_dst, dst_kp, good[:20], None,
                                          flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
            cv.imwrite('top20_sift_match'+str(d)+src_name, match_result)

            gray_dst_inliers = cv.imread(dst_path, 0)
            all_m, top_m = getHomography(src_kp,dst_kp,good,gray_src,gray_dst_inliers)
            cv.imwrite('all_match'+str(d)+src_name, all_m)
            cv.imwrite('top10_match'+str(d)+src_name, top_m)
        d+=1

if __name__ == "__main__":
    main()
