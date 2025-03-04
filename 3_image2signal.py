import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import scipy.signal
import neurokit2 as nk
import pandas as pd
from tqdm import tqdm
import copy

enable_plot = False 
enable_plot = True 






plt.rcParams['font.family'] = 'Times New Roman'

plt.close('all')

noisy_folder = '3_denoised_ecg_images'
ocr_folder = '6_ocr'
result2_folder = '7_result2'
csv_folder = '8_csv'
os.makedirs(result2_folder, exist_ok=True)
os.makedirs(ocr_folder, exist_ok=True)
os.makedirs(csv_folder, exist_ok=True)

segments_all = []

for filename in os.listdir(noisy_folder):
    if filename.endswith('.png') and not 'pattern' in filename.lower():
        patteint_no = filename.replace('denoised_', '').replace('.png', '')
        print('patteint_no =', patteint_no)
        
        image_path = os.path.join(noisy_folder, filename)
        image = cv2.imread(image_path)
        
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if enable_plot:
            plt.figure(100)
            plt.subplot(2, 2, 1)
            plt.title('Original Image')
            plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            plt.axis('off')
        
        B = max(int(np.shape(image)[0] / 100), 80)
        _, binary_image = cv2.threshold(gray_image, 200, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        cropped_image = image[y+B:y+h-B, x+B:x+w-B]
        
        cropped_image001=copy.deepcopy(cropped_image)
        
        gray_image_cropped = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
        _, binary_image_cropped = cv2.threshold(gray_image_cropped, 200, 255, cv2.THRESH_BINARY_INV)
        contours_cropped, _ = cv2.findContours(binary_image_cropped, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        segments = [cnt for cnt in contours_cropped if cv2.contourArea(cnt) > 50]
        
        for _ in range(5):
            merged_contours = []
            processed = [False] * len(segments)
            
            for i in range(len(segments)):
                if processed[i]:
                    continue
                
                base_contour = segments[i]
                overlapping = [base_contour]
                base_rect = cv2.boundingRect(base_contour)
                
                for j in range(i + 1, len(segments)):
                    if processed[j]:
                        continue
                    
                    cnt = segments[j]
                    cnt_rect = cv2.boundingRect(cnt)
                    
                    if (base_rect[1] < cnt_rect[1] + cnt_rect[3] and
                        base_rect[1] + base_rect[3] > cnt_rect[1]):
                        overlapping.append(cnt)
                        processed[j] = True
                
                merged_contour = np.vstack(overlapping)
                merged_contour = cv2.convexHull(merged_contour)
                merged_contours.append(merged_contour)
                processed[i] = True
            
            segments = merged_contours
        if enable_plot:
            plt.figure(100)
            plt.subplot(2, 2, 1)
            plt.title('Original Image')
            plt.imshow(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))
            plt.axis('off')
        
        max_x = []; max_y = []
        min_y = []; min_x = []
        
        for merged_segment in segments:
            random_color = tuple(np.random.randint(0, 256, 3).tolist())
            cv2.drawContours(cropped_image, [merged_segment], -1, random_color, 2)
            
            x = [merged_segment[i, 0, 0] for i in range(len(merged_segment))]
            y = [merged_segment[i, 0, 1] for i in range(len(merged_segment))]
            
            max_x.append(np.max(x))
            max_y.append(np.max(y))
            min_y.append(np.min(y))
            min_x.append(np.min(x))
        
        if len(min_x) == 1:
            min_y = int(min_y[0])
            max_y = int(max_y[0])
            min_x = int(min_x[0])
            max_x = int(max_x[0])
             
            
            gray_image_cropped001 = cv2.cvtColor(cropped_image001, cv2.COLOR_BGR2GRAY)
            _, binary_image_cropped001 = cv2.threshold(gray_image_cropped001, 200, 255, cv2.THRESH_BINARY_INV)
            contours_cropped001, _ = cv2.findContours(binary_image_cropped001, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            segments = [cnt for cnt in contours_cropped001 if cv2.contourArea(cnt) > 50]
            
            for _ in range(5):
                merged_contours = []
                processed = [False] * len(segments)
                
                for i in range(len(segments)):
                    if processed[i]:
                        continue
                    
                    base_contour = segments[i]
                    overlapping = [base_contour]
                    base_rect = cv2.boundingRect(base_contour)
                    
                    for j in range(i + 1, len(segments)):
                        if processed[j]:
                            continue
                        
                        cnt = segments[j]
                        cnt_rect = cv2.boundingRect(cnt)
                        
                        if (base_rect[1] < cnt_rect[1] + cnt_rect[3] and
                            base_rect[1] + base_rect[3] > cnt_rect[1]):
                            overlapping.append(cnt)
                            processed[j] = True
                    
                    merged_contour = np.vstack(overlapping)
                    merged_contour = cv2.convexHull(merged_contour)
                    merged_contours.append(merged_contour)
                    processed[i] = True
                
                segments = merged_contours
            if enable_plot:
                plt.figure(100)
                plt.subplot(2, 2, 1)
                plt.title('Original Image')
                plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                plt.axis('off')
            
            
             
            max_x = []; max_y = []
            min_y = []; min_x = []
            
            for merged_segment in segments:
                random_color = tuple(np.random.randint(0, 256, 3).tolist())
                cv2.drawContours(cropped_image, [merged_segment], -1, random_color, 2)
                
                x = [merged_segment[i, 0, 0] for i in range(len(merged_segment))]
                y = [merged_segment[i, 0, 1] for i in range(len(merged_segment))]
                
                max_x.append(np.max(x))
                max_y.append(np.max(y))
                min_y.append(np.min(y))
                min_x.append(np.min(x))
        if enable_plot:
            plt.figure(100)
            plt.subplot(2, 2, 3)
            plt.title('Cropped Image (Merged Contours)')
            plt.imshow(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))
            plt.axis('off')
        
        S = 0
        signal_all = []
        all_results = []
        
        LEAD_name = [
            "i", "ii", "iii", 
            "avr", "avl", "avf", 
            "v1", "v2", "v3", 
            "v4", "v5", "v6", 
            "vx", "vy", "vz"
        ]
        
        
        for i in tqdm(range(len(segments)), desc="Processing segments"):
            merged_segment = segments[i]
            S = S + 5
            signal = []
            Lead_name = 'lead ' + LEAD_name[i]
            if enable_plot:
                plt.figure(100)
                plt.subplot(len(segments), 5, S-2)
                # if i == 0:
                plt.title('Cropped Image lead '+Lead_name)
                # cropped_image001 = image
                plt.imshow(cv2.cvtColor(cropped_image001[min_y[i]:max_y[i], :], cv2.COLOR_BGR2RGB))
                plt.axis('off')
            
            image_Signal_i = cv2.cvtColor(cropped_image001[min_y[i]:max_y[i], :], cv2.COLOR_BGR2RGB)
            temp = image_Signal_i.copy()
            
            
            
            for y in range(np.shape(image_Signal_i)[1]):
                temp[:, y, 0] = 0
                image_Signal_iy = image_Signal_i[:, y, 0]
                index = np.argmin(image_Signal_iy)
                min_value = min(image_Signal_iy)
                
                if not min_value == image_Signal_i[0, 0, 0]:
                    signal.append(-1 * index)
            
            lowpass = scipy.signal.butter(1, 0.5, btype='lowpass', fs=200, output='sos')
            lowpassed = scipy.signal.sosfilt(lowpass, signal)
            filtered_signal = signal - lowpassed
            removed = int(len(filtered_signal) / 10)
            filtered_signal = filtered_signal[removed:-1 * removed]
            
            np.savez('ecg.npz', signal=filtered_signal)
            
            ecg_signal = filtered_signal
            from ECG_EX2 import ECG_extract_qrs2
            
            
            # print('Lead_name =', Lead_name)
            
            out_path_image001 = result2_folder + '/' + patteint_no + Lead_name + '.png'
            # out_path_image='temp.png', lead='Lead II',dpi=100
            
            results,Result2 = ECG_extract_qrs2(ecg_signal,  enable_plot )
            # Close all figures except figure 1
            
            results['Lead'] = Lead_name
            all_results.append(results)
            signal_all.append(filtered_signal)
            
          
            if enable_plot:
                plt.figure(100)
                plt.subplot(len(segments), 5, S-1)
                plt.plot(filtered_signal)
                plt.axis('off')
                if i == 0:
                    plt.title('OCR Image to signal')
                
                
                
            
            
                    
            t, ecg_signal, r_peaks, t_peaks, p_peaks, q_peaks, s_peaks,rpeaks = Result2.values()
            
            
            # plt.figure(figsize=(12, 6))
            if enable_plot:
                plt.figure(200)
                plt.subplot(221)
                plt.plot(t, ecg_signal, label='ECG Signal', color='blue')
                plt.scatter(t[r_peaks], ecg_signal[r_peaks], color='red', label='R-Peaks')
                plt.scatter(t[t_peaks], ecg_signal[t_peaks], color='orange', label='T-Peaks')
                plt.scatter(t[p_peaks], ecg_signal[p_peaks], color='green', label='P-Peaks')
                plt.scatter(t[q_peaks], ecg_signal[q_peaks], color='purple', label='Q-Peaks')
                plt.scatter(t[s_peaks], ecg_signal[s_peaks], color='brown', label='S-Peaks')
                plt.title('ECG Signal with Peaks' + Lead_name)
                plt.xlabel('Time (s)')
                plt.ylabel('Amplitude')
                plt.grid()
                plt.legend()
                plt.subplot(122)
                plt.imshow(plt.imread('temp.png'))
                plt.axis('off')
                plt.subplot(223)
                if len(t_peaks) > 2:  # Ensure there are at least 3 T-peaks
                    index = np.where(t == t[t_peaks[2]])[0][0]
                    plt.plot(t[0:index], ecg_signal[0:index], label='ECG Signal', color='blue')
                    plt.scatter(t[r_peaks[:3]], ecg_signal[r_peaks[:3]], color='red', label='R-Peaks')
                    plt.scatter(t[t_peaks[:3]], ecg_signal[t_peaks[:3]], color='orange', label='T-Peaks')
                    plt.scatter(t[p_peaks[:3]], ecg_signal[p_peaks[:3]], color='green', label='P-Peaks')
                    plt.scatter(t[q_peaks[:3]], ecg_signal[q_peaks[:3]], color='purple', label='Q-Peaks')
                    plt.scatter(t[s_peaks[:3]], ecg_signal[s_peaks[:3]], color='brown', label='S-Peaks')
                    plt.title('Zoomed-In ECG Signal with Peaks' + Lead_name)
                    plt.xlabel('Time (s)')
                    plt.ylabel('Amplitude')
                    plt.grid()
                    plt.legend()
                 
                plt.figure(100)
                plt.subplot(len(segments), 5, S-1)
                plt.plot(ecg_signal, label='ECG Signal', color='blue')
                plt.scatter(rpeaks['ECG_R_Peaks'], ecg_signal[rpeaks['ECG_R_Peaks']], color='red', label='R-Peaks')
                if i == 0:
                    plt.title('ECG Signal with R-Peaks')
                plt.xlabel('Time (samples)')
                plt.ylabel('Amplitude')
                # plt.legend()
                
                plt.subplot(len(segments), 5, S)
                plt.imshow(plt.imread('temp.png'))
                plt.axis('off') 
                
                
                plt.figure(200)
                manager = plt.get_current_fig_manager()
                manager.full_screen_toggle()
                ocr_folder_path = f"{ocr_folder}/{patteint_no} Lead {Lead_name}.png"
                plt.savefig(ocr_folder_path, bbox_inches='tight', pad_inches=0.5, dpi=300)
                plt.close(200)
            


        if enable_plot:
            plt.show()
        segments_all.append(segments)
        
        if enable_plot:
            print(f'Number of merged segments for {filename}: {len(segments)}')
            ocr_folder_path_all = f"{ocr_folder}/{patteint_no}.png"
            try:
                plt.figure(100)
                manager = plt.get_current_fig_manager()
                manager.full_screen_toggle()
                plt.savefig(ocr_folder_path_all, bbox_inches='tight', pad_inches=0.5, dpi=600)
            except:
                print('error', ocr_folder_path)
            
            plt.close('all')
        
        df = pd.DataFrame(all_results)
        columns = ['Lead'] + [col for col in df.columns if col != 'Lead']
        df = df[columns]
        
        filename_csv = csv_folder + '/ecg_results ' + str(patteint_no) + '.csv'
        filename_xlsx = csv_folder + '/ecg_results ' + str(patteint_no) + '.xlsx'
        df.to_csv(filename_csv, index=False)
        df.to_excel(filename_xlsx, index=False)

print(f'Number of merged segments: {len(segments_all)}')