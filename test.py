import os
import glob

import cv2
import wfdb
import numpy as np


def test_produce_single_image_mask():
    train_dataset = [
        header_file.replace(".hea", "") for header_file in glob.glob(
            os.path.join(
                "tmp", 
                "*", 
                "*", 
                "*",
                "*.hea"
            )
        ) 
    ]
    example = train_dataset[0]
    signals, fields = wfdb.rdsamp(example)
    signals = np.array(signals, dtype=np.float32).T

    signals = signals[:, :int(signals.shape[1]*0.25)]
    length_in_s = signals.shape[0] / fields["fs"]
    time = np.arange(signals.shape[-1]) / signals.shape[-1] * length_in_s
    time = time[:int(time.shape[0]*0.25)]

    image = cv2.imread("background.png", cv2.IMREAD_COLOR_RGB)
    image = image[:960, :1970, :]

    mask = np.zeros_like(image)

    offset = 0.25

    for idx_signal, signal in enumerate(signals[:4]):
        x_start = int(image.shape[1] * 0.31)
        x_end = int(image.shape[1] * 0.99)
        y_start = int(image.shape[0] * 0.01 + idx_signal*int(image.shape[0] * offset))
        y_end = int(image.shape[0] * 0.11 + idx_signal*int(image.shape[0] * offset))
        roi_width = x_end - x_start
        roi_height = y_end - y_start

        signal_min = np.min(signal)
        signal_max = np.max(signal)
        normalized_signal = signal * roi_height
        normalized_signal = normalized_signal.astype(int)

        signal_x_scale = roi_width / len(time)
        signal_y_scale = roi_height / (signal_max - signal_min)
        
        contour_points = []
        contour = []
        for i in range(1, len(time)):
            x = int(x_start + i * signal_x_scale)
            y = int(y_start + roi_height - normalized_signal[i])
            contour.append((x, y))

            if i == len(time) - 1 or x != int(x_start + (i + 1) * signal_x_scale):
                contour_points.append(np.array(contour, dtype=np.int32))
                contour = []
        
        cv2.drawContours(image, contour_points, -1, (0, 0, 0), 1)
        cv2.drawContours(mask, contour_points, -1, (255, 255, 255), 2)

        # cv2.putText(image, "I", (x_start-30, int((y_start+y_end)/2)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)

        # cv2.rectangle(image, (x_start, y_start), (x_end, y_end), (255, 0, 0), 1)

    cv2.imwrite(os.path.join("tmp", "sample.png"), image)
    cv2.imwrite(os.path.join("tmp", "sample_mask.png"), mask)