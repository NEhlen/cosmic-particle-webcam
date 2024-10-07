import numpy as np
from scipy.ndimage import label
from scipy import ndimage
from datetime import datetime
import logging
import sys

run_folder = "data/test/logitechc270"

logging_file = "main.log"
logging.basicConfig(filename=logging_file, level=logging.INFO)
logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler(sys.stderr))


def find_islands(
    img: np.ndarray, island_size: int = 4
) -> tuple[np.ndarray, np.ndarray]:
    img_cutoff = (img > 0.0).astype(int)
    labels, num_features = label(img_cutoff)
    # print(num_features)
    areas = ndimage.sum(img_cutoff, labels, range(num_features + 1))
    mask = areas >= island_size
    # filter img
    # filtered_img = mask[labels.ravel()].reshape(labels.shape)
    img[~mask[labels.ravel()].reshape(labels.shape)] = 0

    return img, img_cutoff


def find_outliers(img: np.ndarray) -> np.ndarray:
    mean, std = img.mean(), img.std()
    if mean < 0:
        mean = 0

    img[img < mean + 5 * std] = 0
    img[img < 10] = 0
    return img


def get_cap(frame: np.ndarray, ref: np.ndarray):

    t = find_outliers(frame - ref)

    t, _ = find_islands(t)
    t = np.clip(t, a_min=0, a_max=255)
    return t


class Cam:
    def __init__(self, index: int, cutoff_percentage: float = 0.15):
        self.index = index
        self.ref = np.loadtxt(run_folder + f"/reference_Cam{index}.npytxt")
        self.frame = np.zeros(self.ref.shape)
        self.height, self.width = self.ref.shape
        self.pixel_mask = self.ref > 6
        self.ref[self.pixel_mask] = 0

        self.percentage = cutoff_percentage
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

        self.events = []

    def integrate_image(self):
        ret, frame = self.cap.retrieve()
        self.frame = frame
        self.frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.frame[self.pixel_mask] = 0
        processed = get_cap(
            self.frame[self.min_y : self.max_y, self.min_x : self.max_x],
            self.ref[self.min_y : self.max_y, self.min_x : self.max_x],
        )
        found_rays = False
        if processed.sum() > 0:
            event = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f"),
                "image": processed,
            }
            self.events.append(event)
            found_rays = True

        # pad processed back to original shape
        processed = np.pad(
            processed,
            [
                (self.min_y, self.ref.shape[0] - self.max_y),
                (self.min_x, self.ref.shape[1] - self.max_x),
            ],
            mode="constant",
            constant_values=0,
        )

        self.integrated += processed

        if found_rays:
            return 1
        else:
            return 0


cam_indices = [
    0,
]

if __name__ == "__main__":

    from cosmic.calibration import find_threshold
    import cv2
    import matplotlib.pyplot as plt
    import time

    logger.info("##### NEW RUN #####")
    logger.info(f"Saving to {run_folder}")
    cams: list[Cam] = []
    for cam_index in cam_indices:
        logger.info(f"initializing Cam {cam_index}")
        cams.append(Cam(cam_index))

    # warmup
    null_time = time.time()
    warmup = 60
    logger.info(f"Warming up for {warmup} seconds")
    while time.time() - null_time <= warmup:
        for cam in cams:
            cam.cap.grab()
        for count, cam in enumerate(cams):
            ret, frame = cam.cap.retrieve()

    # capture
    capture_time = 3600.0 * 3
    logger.info(f"Capturing for {capture_time} seconds")
    start_time = time.time()
    count = 0
    found_rays = 0
    while time.time() - start_time <= capture_time:
        for cam in cams:
            cam.cap.grab()

        for cur_count, cam in enumerate(cams):
            found_rays += cam.integrate_image()
            cv2.imshow(
                f"frame{cam.index}",
                cam.integrated[cam.min_y : cam.max_y, cam.min_x : cam.max_x],
            )
            if cur_count == 0:
                cv2.imshow(
                    f"rawframe{cam.index}",
                    np.clip(
                        cam.frame[cam.min_y : cam.max_y, cam.min_x : cam.max_x] * 20,
                        a_min=0,
                        a_max=255,
                    ),
                )

        count += 1

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    logger.info(f"Finished integration, found {found_rays} potential rays")
    cur_cam = cams[0]
    plt.imshow(
        cur_cam.integrated[cur_cam.min_y : cur_cam.max_y, cur_cam.min_x : cur_cam.max_x]
    )
    plt.figure()
    plt.imshow(cur_cam.integrated, vmin=0, vmax=30)
    plt.hlines(cur_cam.min_y, 0, cur_cam.integrated.shape[1])
    plt.hlines(cur_cam.max_y, 0, cur_cam.integrated.shape[1])
    plt.vlines(cur_cam.max_x, 0, cur_cam.integrated.shape[0])
    plt.vlines(cur_cam.min_x, 0, cur_cam.integrated.shape[0])
    plt.xlim([0, cur_cam.integrated.shape[1]])
    plt.ylim([0, cur_cam.integrated.shape[0]])
    plt.figure()
    plt.imshow(
        cur_cam.integrated[cur_cam.min_y : cur_cam.max_y, cur_cam.min_x : cur_cam.max_x]
        > 0
    )
    plt.savefig(run_folder + "/cosmic_rays.png")
    cv2.destroyAllWindows()
    for cam in cams:
        cam.cap.release()
