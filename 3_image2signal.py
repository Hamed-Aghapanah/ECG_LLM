import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, butter, filtfilt, resample
import neurokit2 as nk
import pandas as pd
from tqdm import tqdm

plt.rcParams['font.family'] = 'Times New Roman'  # تنظیم فونت به Times New Roman
# plt.rcParams['font.size'] = 16  # تنظیم سایز فونت به 16
# plt.rcParams['font.weight'] = 'bold'  # تنظیم وزن فونت به bold (اختیاری)

# Close any existing plots
plt.close('all')

# Path to the noisy images folder
noisy_folder = '3_denoised_ecg_images'
ocr_folder = '6_ocr'
result2_folder='7_result2'
csv_folder='8_csv'
os.makedirs(result2_folder, exist_ok=True)
os.makedirs(csv_folder, exist_ok=True)




import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
import scipy # import signal, datasets

def remove_baseline_drift(signal,fs=20000 ):
    # Butterworth, first order, 0.5 Hz cutoff
    lowpass = scipy.signal.butter(1, 0.5, btype='lowpass', fs=fs, output='sos')
    lowpassed = scipy.signal.sosfilt(lowpass, signal)
    filtered_signal = signal - lowpassed
    removed = int (len(filtered_signal)/10)
    filtered_signal=filtered_signal[removed:-1*removed]
     
    
    return filtered_signal

# Function to identify segments
def identify_segments(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary_image = cv2.threshold(gray_image, 200, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    segments = [cnt for cnt in contours if cv2.contourArea(cnt) > 50]  # Area threshold
    return segments

# Function to remove border and crop the image
def remove_border_and_crop(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    B = max(int(np.shape(image)[0] / 100), 5)
    _, binary_image = cv2.threshold(gray_image, 200, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find the largest contour (border)
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Get bounding rectangle coordinates
    x, y, w, h = cv2.boundingRect(largest_contour)
    
    # Crop the image
    cropped_image = image[y+B:y+h-B, x+B:x+w-B]
    
    return cropped_image

# Function to merge overlapping contours based on vertical overlap
def merge_vertical_contours(segments):
    if not segments:
        return []

    merged_contours = []
    processed = [False] * len(segments)  # Track processed contours
    
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
            
            # Check for vertical overlap
            if (base_rect[1] < cnt_rect[1] + cnt_rect[3] and
                base_rect[1] + base_rect[3] > cnt_rect[1]):
                overlapping.append(cnt)
                processed[j] = True  # Mark this contour as processed

        # Merge contours using convex hull
        merged_contour = np.vstack(overlapping)
        merged_contour = cv2.convexHull(merged_contour)
        merged_contours.append(merged_contour)

        processed[i] = True  # Mark the base contour as processed

    return merged_contours

# Function to perform multiple merges and display results
def process_images(noisy_folder,ocr_folder= '6_ocr' , iterations=5):
    segments_all = []
    for filename in os.listdir(noisy_folder) :
        if filename.endswith('.png') and  not 'pattern' in filename.lower():
            patteint_no=filename.replace('denoised_', '')
            patteint_no=patteint_no.replace('.png', '')
            print('patteint_no =',patteint_no)
            # print('filename ',filename)
            image_path = os.path.join(noisy_folder, filename)
            image = cv2.imread(image_path)
            # Remove border and crop the image
            cropped_image = remove_border_and_crop(image)
            import copy
            cropped_image001 = copy.deepcopy(cropped_image)
            segments = identify_segments(cropped_image)
            
            for _ in range(iterations):
                segments = merge_vertical_contours(segments)
            print(np.shape(segments))
            
            plt.figure(1)
            plt.subplot(2, 2, 1)
            plt.title('Original Image')
            plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            plt.axis('off')
            
            max_x=[]; max_y=[]
            min_y=[]; min_x=[]
            

            # Draw merged contours on the cropped image with random colors
            for merged_segment in segments:
                random_color = tuple(np.random.randint(0, 256, 3).tolist())
                cv2.drawContours(cropped_image, [merged_segment], -1, random_color, 2)  # Draw contours with random color

                x=[]
                y=[]
                for i in range(len(merged_segment)):
                    x.append(  merged_segment[i,0,0])
                    y.append(  merged_segment[i][0,1])
                
                max_x.append(np.max(np.max(x)))
                max_y.append(np.max(np.max(y)))
                min_y.append(np.min(np.min(y)))
                min_x.append(np.min(np.min(x)))
            
            print(np.shape(min_x))
            if np.shape(min_x)[0] ==1:
                cropped_image = remove_border_and_crop(image)
                print(np.shape(cropped_image ))
                print(min_y)
                
                
                min_y=int(min_y)
                max_y=int(max_y)
                min_x=int(min_x)
                max_x=int(max_x)
                
                x=cropped_image [ min_y:max_y , min_x:max_x  , :]
                cropped_image001 = copy.deepcopy(x)
                segments = identify_segments(cropped_image)
                
                for _ in range(iterations):
                    segments = merge_vertical_contours(segments)
                print(np.shape(segments))
                
                plt.figure(1)
                plt.subplot(2, 2, 1)
                plt.title('Original Image')
                plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                plt.axis('off')
                
                max_x=[]; max_y=[]
                min_y=[]; min_x=[]
                

                # Draw merged contours on the cropped image with random colors
                for merged_segment in segments:
                    random_color = tuple(np.random.randint(0, 256, 3).tolist())
                    cv2.drawContours(cropped_image, [merged_segment], -1, random_color, 2)  # Draw contours with random color
                    x=[]
                    y=[]
                    for i in range(len(merged_segment)):
                        x.append(  merged_segment[i,0,0])
                        y.append(  merged_segment[i][0,1])
                    max_x.append(np.max(np.max(x)))
                    max_y.append(np.max(np.max(y)))
                    min_y.append(np.min(np.min(y)))
                    min_x.append(np.min(np.min(x)))
                    
                
            print( ' np.shape(merged_segment) ' ,np.shape(merged_segment))
            # print(max_x,max_y)    
            plt.subplot(2, 2, 3)
            plt.title('Cropped Image (Merged Contours)')
            plt.imshow(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))
            plt.axis('off')
            
            
            
            S=0
            signal_all=[]
            
            all_results = []

            LEAD_name = [
            "i", "ii", "iii", 
            "avr", "avl", "avf", 
            "v1", "v2", "v3", 
            "v4", "v5", "v6", 
            "vx", "vy", "vz"                ]
            
            # for i in range (len(segments)):
            for i in tqdm(range(len(segments)), desc="Processing segments"):
                merged_segment = segments[i]  
                S=S+4
                signal=[]
                plt.figure(1)
                plt.subplot(len(segments), 4, S-1)
                if i==0:
                    plt.title('Cropped Image ' )
                
                plt.imshow(cv2.cvtColor(cropped_image001[min_y[i]:max_y[i] , : ], cv2.COLOR_BGR2RGB))
                plt.axis('off')
                sss
                image_Signal_i = cv2.cvtColor(cropped_image001[min_y[i]:max_y[i] , : ], cv2.COLOR_BGR2RGB)
                temp = copy.deepcopy( image_Signal_i)
                for y in range(np.shape(image_Signal_i)[1]):                      
                    temp[:,y ,0]=0
                    

                    image_Signal_iy=image_Signal_i[:,y ,0]
                    image_Signal_iy=image_Signal_iy[: ]
                    index = np.argmin(image_Signal_iy)
                    min_value = min (image_Signal_iy)
                    
                    if not min_value == image_Signal_i  [0,0,0]:
                        signal.append(-1*index)
               
                signal = remove_baseline_drift(signal, fs=200)
                np.savez('ecg.npz', signal=signal)
                
                
                ecg_signal =  signal
                from ECG_EX import ECG_extract_qrs
                
                Lead_name =' lead '+LEAD_name[i]
                
                
                
                out_path_image001  = result2_folder +'/'+patteint_no+Lead_name+'.png'
                
                results = ECG_extract_qrs(ecg_signal,out_path_image=out_path_image001 ,lead =Lead_name,dpi=600)
                
                results['Lead'] = Lead_name
                
                all_results.append(results)
                
                signal_all.append(signal)    
                
                plt.figure(1)
                plt.subplot(len(segments), 4, S )
                plt.plot(signal )
                plt.axis('off')
                if i==0:
                    plt.title('OCR Image  to signal' )
                    
                print(i)
                
                 
            
            plt.show()
            segments_all.append(segments)
            
            print(f'Number of merged segments for {filename}: {len(segments)}')
            # print(maxx ,minn)
            ocr_folder_path = f"{ocr_folder}/{patteint_no}.png"
            try:
                plt.savefig(ocr_folder_path, bbox_inches='tight', pad_inches=0.5, dpi=300)  # افزایش رزولوشن به 600 dpi
            except:
                print('eerroorr' , ocr_folder_path)
                pass
            plt.close('all')
            
            # Convert the list of dictionaries to a DataFrame
            df = pd.DataFrame(all_results)
            
            # Reorder columns to have 'Lead' as the first column
            columns = ['Lead'] + [col for col in df.columns if col != 'Lead']
            df = df[columns]
            
            # Save the DataFrame to a CSV file
            filename_csv = csv_folder+'/ecg_results '+ str(patteint_no)+ '.csv'
            filename_xlsx =csv_folder+ '/ecg_results '+ str(patteint_no)+ '.xlsx'
            df.to_csv( filename_csv, index=False)
            
            df.to_excel(filename_xlsx, index=False)
            


            # 
        
    return segments_all

# Process images with multiple merges
segments_all = process_images(noisy_folder,ocr_folder, iterations=5)
print(f'Number of merged segments: {len(segments_all)}')
