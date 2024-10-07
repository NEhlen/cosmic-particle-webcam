import time

import cv2
import numpy as np
import matplotlib.pyplot as plt
import logging
import sys

import json

logging_file = "main.log"
logging.basicConfig(filename=logging_file, level=logging.INFO)
logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler(sys.stderr))


def calibrate(
    seconds: int = 60,
    run: str = "data/test/testrun",
    cam_id=0,
    warmup_time=0.0,
    **kwargs,
) -> np.ndarray:
    logger.info("-#" * 30)
    logger.info("Starting Calibration ...")
    cap = cv2.VideoCapture(cam_id)
    width, height = 1920, 1080
    if "width" in kwargs:
        width = kwargs["width"]
    if "height" in kwargs:
        height = kwargs["height"]

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    # cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 3)  # auto mode
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)  # manual mode
    cap.set(cv2.CAP_PROP_EXPOSURE, -2)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    start_time = time.time()
    frames = 0  # count integrated frames
    reference = np.zeros((height, width))

    logger.info(f"Starting Capture on Camera {cam_id}")
    logger.info(f"Dimensions: {width} x {height}")

    cur_t = time.time()
    while cur_t - start_time <= seconds:
        ret, frame = cap.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if cur_t - start_time > warmup_time:
            reference += frame
            frames += 1
        cv2.imshow("frame", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            logger.info("Capture Cancelled via Keyboard interrupt")
            break
        cur_t = time.time()
    logger.info(f"Captured {frames} frames in {seconds} seconds. Releasing Camera.")
    logger.info("-" * 30)
    cap.release()
    cv2.destroyAllWindows()

    # get average
    reference /= frames
    plt.imshow(reference, cmap="gray")
    plt.savefig(run + f"/reference_cam{cam_id}.png")
    np.savetxt(run + f"/reference_Cam{cam_id}.npytxt", reference)

    with open(run + f"/config_cam{cam_id}.json", "w+"):
        json.dumps(
            {
                "integration time [s]": seconds,
                "warmup time [s]": warmup_time,
                "width [pixels]": width,
                "height [pixels]": height,
            }
        )
    return reference


def find_threshold(ref: np.ndarray) -> float:
    return ref.mean() + 5.0 * ref.std()


if __name__ == "__main__":
    import os

    os.makedirs("data/test/logitechc270", exist_ok=True)

    ref0 = calibrate(
        180,
        width=1280,
        height=960,
        cam_id=0,
        warmup_time=60.0,
        run="data/test/logitechc270",
    )  # 1280,  # 720,
    # ref1 = calibrate(
    #     180,
    #     width=640,
    #     height=480,
    #     cam_id=2,
    #     warmup_time=60.0,
    # )  # 1280,  # 720,
    # ref2 = calibrate(
    #     180,
    #     width=640,
    #     height=480,
    #     cam_id=4,
    #     warmup_time=60.0,
    # )  # 1280,  # 720,
    threshold = find_threshold(ref0)
