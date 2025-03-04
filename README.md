# ECG LLM: A Novel Approach for Converting Scanned ECG Images to Digital Signals Using Large Language Models

This repository contains the code and resources for the research article titled "ECG LLM: A Novel Approach for Converting Scanned ECG Images to Digital Signals Using Large Language Models". The study introduces a novel framework that leverages large language models (LLMs) to convert scanned ECG images into digital signals, addressing challenges such as noise and artifacts in ECG signals, particularly in non-invasive fetal monitoring.
![ECG LLM Overview](https://github.com/Hamed-Aghapanah/ECG_LLM/blob/main/FIGs.png)
## Repository Structure

- **pattern_generator.py**: This script generates a pattern image by repeating a base ECG image multiple times horizontally and vertically. The generated pattern is used to simulate noise in ECG images, which is essential for testing denoising algorithms.
  
- **test.py**: This script tests the process of producing a single image mask from ECG signals. It reads ECG signals, processes them, and generates a visual representation of the ECG waveform. The script also creates a mask to isolate the ECG signal from the background.

- **1_ECG_reader.py**: This script reads raw ECG signals from the PTB Diagnostic ECG Database and converts them into visual representations. It also adds salt-and-pepper noise to simulate real-world conditions and saves the noisy images for further processing.

- **2_Denoised.py**: This script applies a median filter to denoise the noisy ECG images. It compares the denoised images with the original images and calculates the difference between them. The results are saved for further analysis.

- **3_image2signal.py**: This script converts the denoised ECG images back into digital signals. It uses advanced image processing techniques to extract meaningful waveforms from the images and reconstructs the original ECG signals. The script also evaluates the error between the original and reconstructed signals using metrics such as Dynamic Time Warping (DTW) and Structural Similarity Index (SSIM).

- **4_LLM.py**: This script integrates large language models (LLMs) for further analysis of the extracted ECG features. It uses models like GPT-4 and LLaMA3 10B to generate diagnostic reports and interpret the ECG signals.

## Key Components of the Research

### Dataset 
Downloading the Dataset
To download the dataset, follow these steps:

Visit the PTB Diagnostic ECG Database page on PhysioNet.

Click the Download ZIP button to download the entire dataset as a compressed file.

Extract the contents of the ZIP file to a directory of your choice (e.g., data/ptbdb/).
```bash
wget -r -N -c -np https://physionet.org/files/ptbdb/1.0.0/
# or 
import wfdb

# Load a sample ECG record
record = wfdb.rdrecord('data/ptbdb/patient001/s0010_re')
signals = record.p_signal  # ECG signals
fields = record.__dict__   # Metadata



data/

└── ptbdb/

    ├── patient001/

    │   ├── s0010_re.dat

    │   ├── s0010_re.hea

    │   └── s0010_re.atr

    ├── patient002/

    │   ├── s0014lre.dat

    │   ├── s0014lre.hea

    │   └── s0014lre.atr

    └── ...
```

```bibtex
@article{bousseljot2005nutzung,  title={Nutzung der EKG-Signaldatenbank CARDIODAT der PTB über das Internet},  author={Bousseljot, R and Kreiseler, D and Schnabel, A},  journal={Biomedizinische Technik/Biomedical Engineering},
  volume={50},  number={s1},  pages={317--318},  year={2005},  publisher={De Gruyter}}
```
### Signal-to-Image Conversion
The raw ECG signals are converted into visual representations using the `1_ECG_reader.py` script. This step involves plotting the ECG signals for each lead and adding noise to simulate real-world conditions.

### Image Denoising
The `2_Denoised.py` script applies a median filter to remove noise from the ECG images. The denoised images are compared with the original images to evaluate the effectiveness of the denoising process.

### Image-to-Signal Conversion
The `3_image2signal.py` script reconstructs the digital ECG signals from the denoised images. It uses advanced image processing techniques to extract the ECG waveforms and evaluates the accuracy of the reconstructed signals using metrics such as DTW, SSIM, and Cross-Correlation.
![mage-to-Signal Conversion](https://github.com/Hamed-Aghapanah/ECG_LLM/blob/main/Picture1.png)
![mage-to-Signal Conversion](https://github.com/Hamed-Aghapanah/ECG_LLM/blob/main/7_result2/patient001%20lead%20i.png)


### Feature Extraction and LLM Integration
The `4_LLM.py` script extracts key features from the reconstructed ECG signals, such as the P, Q, R, S, and T waves. These features are then fed into large language models (LLMs) for further analysis and diagnostic report generation.

## How to Use the Repository

### Clone the Repository
```bash
git clone https://github.com/your-username/ECG-LLM.git
cd ECG-LLM
```

### Install Dependencies
Ensure you have the required Python libraries installed:
```bash
pip install numpy matplotlib opencv-python scipy neurokit2 pandas tqdm wfdb
```

### Run the Scripts
Generate ECG images with noise:
```bash
python 1_ECG_reader.py
```

Denoise the ECG images:
```bash
python 2_Denoised.py
```

Convert images back to signals:
```bash
python 3_image2signal.py
```

Analyze ECG features using LLMs:
```bash
python 4_LLM.py
```

## Contribution
Contributions to this repository are welcome! If you have any suggestions, improvements, or bug fixes, please open an issue or submit a pull request.

## Citation
If you use this code or the research findings in your work, please cite the original article:
```bibtex
@article{aghapanah2024ecgllm,
  title={ECG LLM: A Novel Approach for Converting Scanned ECG Images to Digital Signals Using Large Language Models},
  author={Aghapanah, Hamed},
  journal={Journal of Advanced Medical Technologies},
  year={2024},
  publisher={Isfahan University of Medical Sciences}
}
```

## Contact
For any questions or inquiries, please contact:
Hamed Aghapanah  
Email: h.aghapanah@amt.mui.ac.ir  
School of Advanced Technologies in Medicine, Isfahan University of Medical Sciences, Isfahan, Iran.

This repository provides a comprehensive framework for converting scanned ECG images into digital signals using large language models. The code and datasets are made available to facilitate reproducibility and further research in the field of ECG signal processing.
