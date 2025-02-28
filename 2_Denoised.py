import wfdb
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import cv2
import os
import json

plt.close("all")

raw_ecgl_folder = "1_ecg_images"
mask_folder = "00_masks"
images_folder = "00_images"
noisy_folder = "2_noisy_ecg_images"
json_folder = "0_json_files"

os.makedirs(raw_ecgl_folder, exist_ok=True)
os.makedirs(mask_folder, exist_ok=True)
os.makedirs(images_folder, exist_ok=True)
os.makedirs(noisy_folder, exist_ok=True)
os.makedirs(json_folder, exist_ok=True)

base_path = Path("ptb-diagnostic-ecg-database-1.0.0")
patient_folders = [
    f for f in base_path.iterdir() if f.is_dir() and f.name.startswith("patient")
]


def add_salt_pepper_noise(image, salt_prob=0.01, pepper_prob=0.01):
    noisy_image = np.copy(image)
    salt_mask = np.random.rand(*image.shape) < salt_prob
    noisy_image[salt_mask] = 255
    pepper_mask = np.random.rand(*image.shape) < pepper_prob
    noisy_image[pepper_mask] = 0
    return noisy_image


def create_ecg_like_image(signals, lead_names, patient_name):
    bias = 0
    for i in range(len(lead_names)):
        if i > 0:
            max_amplitude = np.abs(np.min(signals[:, i])) + np.abs(
                np.max(signals[:, i - 1])
            )
            bias += max_amplitude * 1.1
        plt.plot(signals[:, i] + bias, label=f"Lead {lead_names[i]}", linewidth=0.5)

    image_path = f"{raw_ecgl_folder}/{patient_name}.png"
    plt.savefig(image_path, bbox_inches="tight", pad_inches=0.5, dpi=600)
    plt.close()

    A = plt.imread(image_path)
    image_path2 = f"{mask_folder}/{patient_name}.png"
    a1 = int(np.shape(A)[1] * 275 / 1925)
    a2 = int(np.shape(A)[1] * 1653 / 1925)
    b1 = int(np.shape(A)[0] * 210 / 1925)
    b2 = int(np.shape(A)[0] * 1240 / 1477)
    B = A[b1:b2, a1:a2, :]
    if len(B.shape) == 3:
        B = cv2.cvtColor(B, cv2.COLOR_BGR2GRAY)
    _, B_binarized = cv2.threshold(B, thresh=1, maxval=255, type=cv2.THRESH_BINARY)
    cv2.imwrite(image_path2, 255 - B * 255)

    pattern = plt.imread("pattern.jpg")[:, :, 0]
    ECG_noisy = (
        plt.imread(image_path)[:, :, 0]
        * plt.imread(image_path)[:, :, 1]
        * plt.imread(image_path)[:, :, 2]
    )
    new_size = np.shape(ECG_noisy)
    ECG_noisy = cv2.resize(ECG_noisy, [new_size[1], new_size[0]])
    p = cv2.resize(pattern, [new_size[1], new_size[0]])
    a = p * ECG_noisy + ECG_noisy
    plt.imshow(a, cmap="gray")
    ax = plt.gca()
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)
    plt.grid(False)
    plt.legend(loc="upper left", bbox_to_anchor=(1, 1), fontsize=10)
    plt.tight_layout()
    image_path_patt = f"{raw_ecgl_folder}/pattern_{patient_name}.png"
    plt.savefig(image_path_patt, bbox_inches="tight", pad_inches=0.5, dpi=600)
    plt.close()

    image_path2 = f"{images_folder}/pattern_{patient_name}.png"
    p = cv2.resize(pattern, [np.shape(B)[1], np.shape(B)[0]])
    B = B / np.max(B)
    p = p / np.max(p)
    B = 1 - ((1 - B) + (1 - p)) * 1
    plt.plot(B)
    if len(B.shape) == 3:
        B = cv2.cvtColor(B, cv2.COLOR_BGR2GRAY)
    _, B_binarized = cv2.threshold(B, thresh=1, maxval=255, type=cv2.THRESH_BINARY)
    cv2.imwrite(image_path2, B * 255)
    plt.close("all")

    return image_path, image_path_patt


def read_xyz_file(xyz_path):
    try:
        with open(xyz_path, "r", encoding="latin-1") as file:
            xyz_data = file.read()
        return xyz_data
    except Exception as e:
        print(f"Error reading XYZ file: {e}")
        return None


def process_patient_folder(patient_folder):
    record_files = list(patient_folder.glob("*.hea"))
    if not record_files:
        print(f"No records found in {patient_folder.name}")
        return

    record_name = record_files[0].stem
    signals, fields = wfdb.rdsamp(str(patient_folder / record_name))

    hea_info = {
        "n_sig": fields["n_sig"],
        "fs": fields["fs"],
        "sig_name": fields["sig_name"],
        "units": fields["units"],
        "comments": fields["comments"],
    }

    xyz_file = list(patient_folder.glob("*.xyz"))
    xyz_info = None
    if xyz_file:
        xyz_info = read_xyz_file(xyz_file[0])

    json_data = {"hea_info": hea_info}

    json_path = f"{json_folder}/{patient_folder.name}.json"
    with open(json_path, "w") as json_file:
        json.dump(json_data, json_file, indent=4)

    lead_names = fields["sig_name"]
    patient_name = patient_folder.name
    ecg_image_path, ecg_image_path_patt = create_ecg_like_image(
        signals, lead_names, patient_name
    )

    image = cv2.imread(ecg_image_path)
    noisy_image = add_salt_pepper_noise(image)
    noisy_image_path = f"{noisy_folder}/{patient_name}_noisy.png"
    cv2.imwrite(noisy_image_path, noisy_image)

    image_patt = cv2.imread(ecg_image_path_patt)
    noisy_image_patt = add_salt_pepper_noise(image_patt)
    noisy_image_path_patt = f"{noisy_folder}/pattern_{patient_name}_noisy.png"
    cv2.imwrite(noisy_image_path_patt, noisy_image_patt)


for patient_folder in patient_folders:
    print(f"Processing {patient_folder.name}...")
    process_patient_folder(patient_folder)

print("Done!")
