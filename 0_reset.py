import os
import shutil

# لیست پوشه‌هایی که می‌خواهید حذف کنید
folders_to_delete = [
    '1_ecg_images',
    '00_masks',
    '00_images',
    '2_noisy_ecg_images',
    '0_json_files',
    '3_denoised_ecg_images',
    '4_difference_images',
    '5_result',
    
]

# حذف هر یک از پوشه‌ها
for folder in folders_to_delete:
    if os.path.exists(folder):
        shutil.rmtree(folder)
        print(f'پوشه {folder} با موفقیت حذف شد.')
    else:
        print(f'پوشه {folder} وجود ندارد.')