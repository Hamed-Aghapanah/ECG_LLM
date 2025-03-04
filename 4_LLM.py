import os
import pandas as pd
import requests

# تابع برای دریافت لیست مدل‌های نصب‌شده در ollama
def get_ollama_models():
    try:
        # ارسال درخواست به API ollama برای دریافت لیست مدل‌ها
        response = requests.get('http://localhost:11434/api/tags')
        
        # بررسی موفقیت‌آمیز بودن درخواست
        if response.status_code == 200:
            # دریافت لیست مدل‌ها از پاسخ JSON
            models = [model['name'] for model in response.json()['models']]
            return models
        else:
            print(f"خطا در دریافت مدل‌ها: کد وضعیت {response.status_code}")
            return []
    except Exception as e:
        print(f"خطا در اتصال به ollama: {e}")
        return []

# تابع برای ارسال درخواست به ollama و دریافت پاسخ
def ask_ollama(model_name, prompt):
    try:
        data = {
            "model": model_name,
            "prompt": prompt,
            "stream": False
        }
        response = requests.post('http://localhost:11434/api/generate', json=data)
        if response.status_code == 200:
            return response.json().get('response', 'بدون پاسخ')
        else:
            print(f"خطا در دریافت پاسخ از مدل {model_name}: کد وضعیت {response.status_code}")
            return None
    except Exception as e:
        print(f"خطا در ارسال درخواست به ollama: {e}")
        return None

# تابع برای پردازش یک فایل اکسل
def process_excel_file(file_path, output_folder, models):
    # خواندن فایل اکسل
    df = pd.read_excel(file_path)
    
    # بررسی وجود ستون‌های مورد نیاز
    required_columns = ['P_dur (ms)', 'QRS_dur (ms)', 'T_dur (ms)', 'ST_dur (ms)']
    if not all(col in df.columns for col in required_columns):
        print(f"فایل {file_path} شامل ستون‌های مورد نیاز نیست. ستون‌های مورد نیاز: {required_columns}")
        return
    
    # ایجاد پرسش‌ها بر اساس ستون‌های دیگر
    for model_name in models:
        df[f'Diagnosis_{model_name}'] = df.apply(
            lambda row: ask_ollama(
                model_name,
                f"تشخیص بیماری مرتبط با P_dur={row['P_dur (ms)']} ms، QRS_dur={row['QRS_dur (ms)']} ms، T_dur={row['T_dur (ms)']} ms، و ST_dur={row['ST_dur (ms)']} ms."
            ),
            axis=1
        )
    
    # ذخیره‌سازی فایل پردازش‌شده
    file_name = os.path.basename(file_path)
    output_path = os.path.join(output_folder, f"processed_{file_name}")
    df.to_excel(output_path, index=False)
    print(f"فایل {file_name} پردازش و در {output_path} ذخیره شد.")

# تابع اصلی برای پردازش تمام فایل‌های اکسل در یک پوشه
def process_all_excel_files(input_folder, output_folder, models):
    # ایجاد پوشه خروجی اگر وجود نداشته باشد
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # پردازش هر فایل اکسل در پوشه ورودی
    for file_name in os.listdir(input_folder):
        if file_name.endswith('.xlsx') or file_name.endswith('.xls'):
            file_path = os.path.join(input_folder, file_name)
            process_excel_file(file_path, output_folder, models)

# مسیر پوشه‌های ورودی و خروجی
input_folder = '8_csv'  # پوشه حاوی فایل‌های اکسل
output_folder = 'processed_results'  # پوشه برای ذخیره فایل‌های پردازش‌شده

# دریافت لیست مدل‌های نصب‌شده در ollama
models = get_ollama_models()
if models:
    print("مدل‌های نصب‌شده در ollama:")
    for model in models:
        print(f"- {model}")
    
    # پردازش تمام فایل‌های اکسل
    process_all_excel_files(input_folder, output_folder, models)
else:
    print("هیچ مدلی در ollama یافت نشد.")