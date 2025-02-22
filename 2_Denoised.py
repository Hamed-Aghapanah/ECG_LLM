import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

plt.close('all')
# مسیر پوشه‌ها
original_folder = '1_ecg_images'
noisy_folder = '2_noisy_ecg_images'
denoised_folder = '3_denoised_ecg_images'
difference_folder = '4_difference_images'
result_folder = '5_result'

# ایجاد پوشه‌های خروجی اگر وجود نداشته باشند
os.makedirs(denoised_folder, exist_ok=True)
os.makedirs(difference_folder, exist_ok=True)
os.makedirs(result_folder, exist_ok=True)

# لیست فایل‌ها در پوشه‌ها
original_images = sorted(os.listdir(original_folder))
noisy_images = sorted(os.listdir(noisy_folder))

# تابع برای اعمال فیلتر میانه‌ای (دینویزی)
def denoise_image(image):
    return cv2.medianBlur(image, 3)  # اندازه کرنل فیلتر میانه‌ای را می‌توانید تغییر دهید


# پردازش هر تصویر
for original_img, noisy_img in zip(original_images, noisy_images):
    # خواندن تصاویر
    original = cv2.imread(os.path.join(original_folder, original_img), cv2.IMREAD_GRAYSCALE)
    noisy = cv2.imread(os.path.join(noisy_folder, noisy_img), cv2.IMREAD_GRAYSCALE)
    A,B=np.shape(noisy)
    A1,A2= int (A/8),int(A/4)
    B1,B2= int (B/8),int(B/4)
    
    # دینویزی کردن تصویر نویزی
    denoised = denoise_image(noisy)
    
    try:
    # محاسبه تفاضل بین تصویر دینویزی شده و تصویر اصلی
        difference = cv2.absdiff(denoised, original)
    except: pass
    # ذخیره تصاویر دینویزی شده و تفاضل
    denoised_filename = os.path.join(denoised_folder, f"denoised_{original_img}")
    difference_filename = os.path.join(difference_folder, f"difference_{original_img}")
    cv2.imwrite(denoised_filename, denoised)
    try:
        cv2.imwrite(difference_filename, difference)
    except: pass
    # نمایش تصاویر
    plt.figure(figsize=(20, 5))

    plt.subplot(2, 2, 1)
    plt.title('Original Image')
    plt.imshow(original[A1:A2 ,B1:B2], cmap='gray')
    plt.axis('off')

    plt.subplot(2, 2, 2)
    plt.title('Noisy Image')
    plt.imshow(noisy[A1:A2 ,B1:B2], cmap='gray')
    plt.axis('off')

    plt.subplot(2, 2, 3)
    plt.title('Denoised Image')
    plt.imshow(denoised[A1:A2 ,B1:B2], cmap='gray')
    plt.axis('off')

    plt.subplot(2, 2, 4)
    plt.title('Difference (Denoised - Original)')
    plt.imshow(difference[A1:A2 ,B1:B2], cmap='gray')
    plt.axis('off')

    plt.show()
    
    
    plt.savefig (result_folder+'\\croped_'+original_img, bbox_inches='tight', pad_inches=0.5, dpi=600)
    
    plt.close()
    
    
    # نمایش تصاویر
    plt.figure(figsize=(20, 5))

    plt.subplot(2, 2, 1)
    plt.title('Original Image')
    plt.imshow(original, cmap='gray')
    plt.axis('off')

    plt.subplot(2, 2, 2)
    plt.title('Noisy Image')
    plt.imshow(noisy, cmap='gray')
    plt.axis('off')

    plt.subplot(2, 2, 3)
    plt.title('Denoised Image')
    plt.imshow(denoised, cmap='gray')
    plt.axis('off')

    plt.subplot(2, 2, 4)
    plt.title('Difference (Denoised - Original)')
    plt.imshow(difference, cmap='gray')
    plt.axis('off')

    plt.show()
    
    
    plt.savefig (result_folder+'\\'+original_img, bbox_inches='tight', pad_inches=0.5, dpi=600)
    
    plt.close()