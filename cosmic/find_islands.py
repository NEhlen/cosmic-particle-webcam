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


class Cam:
    def __init__(self, index):
        self.index = index
        self.ref = np.loadtxt(f"data/test/testrun/reference_Cam{index}.npytxt")
        self.height, self.width = self.ref.shape
        self.pixel_mask = self.ref > 6
        self.ref[self.pixel_mask] = 0

        self.percentage = 0.10
        self.min_x, self.max_x = int(self.percentage * self.width), int(
            (1 - self.percentage) * self.width
        )
        self.min_y, self.max_y = int(self.percentage * self.height), int(
            (1 - self.percentage) * self.height
        )

        self.threshold = find_threshold(
            self.ref[self.min_y : self.max_y, self.min_x : self.max_x]
        )
        print(self.threshold)

        self.cap = cv2.VideoCapture(cam_index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 3)  # auto mode

        self.integrated = np.zeros(self.ref.shape)


cam_indices = [
    0,
]

if __name__ == "__main__":

    from cosmic.calibration import find_threshold
    import cv2
    import matplotlib.pyplot as plt
    import time

    cams = []

    for cam_index in cam_indices:
        cams.append(Cam(cam_index))

    start_time = time.time()

    count = 0
    while time.time() - start_time <= 3600.0:
        for cam in cams:
            cam.cap.grab()

        for cur_count, cam in enumerate(cams):
            ret, frame = cam.cap.retrieve()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame[cam.pixel_mask] = 0
            cam.integrated += get_cap(frame, cam.ref, cam.threshold)
            cv2.imshow(
                f"frame{cam.index}",
                cam.integrated[cam.min_y : cam.max_y, cam.min_x : cam.max_x],
            )
            if cur_count == 0:
                cv2.imshow(
                    "rawframe0",
                    np.clip(
                        frame[cam.min_y : cam.max_y, cam.min_x : cam.max_x] * 20,
                        a_min=0,
                        a_max=255,
                    ),
                )

        count += 1

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    #
    cur_cam = cams[0]
    plt.imshow(
        cur_cam.integrated[cur_cam.min_y : cur_cam.max_y, cur_cam.min_x : cur_cam.max_x]
    )
    plt.figure()
    plt.imshow(
        cur_cam.integrated[cur_cam.min_y : cur_cam.max_y, cur_cam.min_x : cur_cam.max_x]
        > 0
    )
    plt.savefig("data/test/islandtest.png")
    cv2.destroyAllWindows()
    for cam in cams:
        cam.cap.release()
