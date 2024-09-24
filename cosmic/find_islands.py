import numpy as np
from scipy.ndimage import label
from scipy import ndimage


def find_islands(img: np.ndarray, island_size: int = 4):
    img_cutoff = (img > 0.0).astype(int)
    labels, num_features = label(img_cutoff)
    # print(num_features)
    areas = ndimage.sum(img_cutoff, labels, range(num_features + 1))
    mask = areas >= island_size
    # filter img
    filtered_img = mask[labels.ravel()].reshape(labels.shape)
    img[~mask[labels.ravel()].reshape(labels.shape)] = 0

    return img, img_cutoff


def find_outliers(img: np.ndarray):
    mean, std = img.mean(), img.std()
    if mean < 0:
        mean = 0

    img[img < mean + 3 * std] = 0
    img[img < 10] = 0
    return img, _


def get_cap(frame, ref, threshold):

    # t, raw_img = find_islands(
    #    frame - ref - threshold
    # )
    # t = np.clip(t, a_min=0, a_max=255)
    # return t + threshold * (t > 0.0).astype(int)
    t, _ = find_outliers(frame - ref)

    t, raw_img = find_islands(t)
    t = np.clip(t, a_min=0, a_max=255)
    return t


if __name__ == "__main__":

    from cosmic.calibration import find_threshold
    import cv2
    import matplotlib.pyplot as plt
    import time

    ref0 = np.loadtxt("data/test/testrun/reference_Cam0.npytxt")
    height, width = ref0.shape
    pixel_mask0 = ref0 > 5  # mask hot pixels
    ref0[pixel_mask0] = 0

    ref1 = np.loadtxt("data/test/testrun/reference_Cam2.npytxt")
    height, width = ref1.shape
    pixel_mask1 = ref1 > 5  # mask hot pixels
    ref1[pixel_mask1] = 0

    ref2 = np.loadtxt("data/test/testrun/reference_Cam4.npytxt")
    height, width = ref2.shape
    pixel_mask2 = ref2 > 5  # mask hot pixels
    ref2[pixel_mask2] = 0

    percentage = 0.10
    min_x, max_x = int(percentage * width), int((1 - percentage) * width)
    min_y, max_y = int(percentage * height), int((1 - percentage) * height)

    threshold = find_threshold(ref0[min_y:max_y, min_x:max_x])
    threshold0 = 10.0
    threshold1 = 20.0
    threshold2 = 10.0
    print(threshold)

    cap0 = cv2.VideoCapture(0)
    cap0.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap0.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap0.set(cv2.CAP_PROP_AUTO_EXPOSURE, 3)  # auto mode
    # cap0.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)  # manual mode
    # cap0.set(cv2.CAP_PROP_EXPOSURE, -7)

    cap1 = cv2.VideoCapture(2)
    cap1.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap1.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap1.set(cv2.CAP_PROP_AUTO_EXPOSURE, 3)  # auto mode
    # cap1.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)  # manual mode
    # cap1.set(cv2.CAP_PROP_EXPOSURE, -7)

    cap2 = cv2.VideoCapture(4)
    cap2.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap2.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap2.set(cv2.CAP_PROP_AUTO_EXPOSURE, 3)  # auto mode
    # cap2.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)  # manual mode
    # cap2.set(cv2.CAP_PROP_EXPOSURE, -7)

    start_time = time.time()
    integrated0 = np.zeros(ref0.shape)
    integrated1 = np.zeros(ref1.shape)
    integrated2 = np.zeros(ref2.shape)
    count = 0
    while time.time() - start_time <= 3600.0:
        cap0.grab()
        cap1.grab()
        cap2.grab()

        ret0, frame0 = cap0.retrieve()
        frame0 = cv2.cvtColor(frame0, cv2.COLOR_BGR2GRAY)
        frame0[pixel_mask0] = 0

        ret1, frame1 = cap1.retrieve()
        frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        frame1[pixel_mask1] = 0

        ret2, frame2 = cap2.retrieve()
        frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        frame2[pixel_mask2] = 0

        integrated0 += get_cap(frame0, ref0, threshold0)
        cv2.imshow("frame0", integrated0[min_y:max_y, min_x:max_x])
        cv2.imshow(
            "rawframe0",
            np.clip(frame0[min_y:max_y, min_x:max_x] * 100, a_min=0, a_max=255),
        )
        integrated1 += get_cap(frame1, ref1, threshold1)
        cv2.imshow("frame1", integrated1[min_y:max_y, min_x:max_x])

        integrated2 += get_cap(frame2, ref2, threshold2)
        cv2.imshow("frame2", integrated2[min_y:max_y, min_x:max_x])

        count += 1
        # temp = frame0 - ref0
        # temp[temp < 0] = 0
        # temp += np.amin(temp)
        # temp /= np.amax(temp)
        # temp = (temp * 255).astype("uint8")
        # tempCMP = cv2.applyColorMap(temp, cv2.COLORMAP_VIRIDIS)
        # cv2.imshow(
        #     "raw",
        #     temp[min_y:max_y, min_x:max_x],
        # )
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    #
    plt.imshow(integrated0[min_y:max_y, min_x:max_x])
    plt.figure()
    plt.imshow(integrated0[min_y:max_y, min_x:max_x] > 0)
    plt.savefig("data/test/islandtest.png")
    cv2.destroyAllWindows()
    cap0.release()
    cap1.release()
    cap2.release()
