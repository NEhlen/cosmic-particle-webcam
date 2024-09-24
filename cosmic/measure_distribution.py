import numpy as np
from scipy.ndimage import label
from scipy import ndimage
import pandas as pd

CAM = 4

if __name__ == "__main__":

    from cosmic.calibration import find_threshold
    import cv2
    import matplotlib.pyplot as plt
    import time

    ref0 = np.loadtxt(f"data/test/testrun/reference_Cam{CAM}.npytxt")
    height, width = ref0.shape
    pixel_mask0 = ref0 > 5  # mask hot pixels
    ref0[pixel_mask0] = 0

    percentage = 0.10
    min_x, max_x = int(percentage * width), int((1 - percentage) * width)
    min_y, max_y = int(percentage * height), int((1 - percentage) * height)

    cap0 = cv2.VideoCapture(CAM)
    cap0.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap0.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap0.set(cv2.CAP_PROP_AUTO_EXPOSURE, 3)  # auto mode

    start_time = time.time()
    count = 0
    df = pd.DataFrame()
    cur_t = time.time()
    while cur_t - start_time <= 1200.0:
        cap0.grab()

        ret0, frame0 = cap0.retrieve()
        frame0 = cv2.cvtColor(frame0, cv2.COLOR_BGR2GRAY)
        frame0[pixel_mask0] = 0
        cv2.imshow("frame0", frame0)

        counts, bins = np.histogram(frame0.ravel(), bins=np.linspace(0, 10, 11))
        ds = pd.Series(counts, bins[:-1])

        df[cur_t - start_time] = ds
        df = df.copy()

        count += 1

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        time.sleep(0.995)
        cur_t = time.time()

    cv2.destroyAllWindows()
    cap0.release()

    df.plot()
    df.to_parquet(f"data/dist/disttest_cam{CAM}.parquet")
