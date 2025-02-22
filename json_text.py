import os
import pandas as pd
import subprocess

# تابع برای تبدیل یک فایل Excel به متن
def convert_excel_to_text(file_path, output_folder):
    # خواندن فایل Excel با استفاده از pandas
    excel_data = pd.read_excel(file_path, sheet_name=None)  # خواندن همه صفحات (sheets)
    
    # نام فایل خروجی
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    output_file_path = os.path.join(output_folder, f"{base_name}.txt")
    
    # نوشتن داده‌ها به فایل متنی
    with open(output_file_path, "w", encoding="utf-8") as txt_file:
        for sheet_name, df in excel_data.items():
            txt_file.write(f"### Sheet: {sheet_name}\n")
            txt_file.write(df.to_string(index=False) + "\n\n")  # نوشتن داده‌ها بدون شاخص
            
            # اضافه کردن توضیحات به متن (اختیاری)
            if sheet_name == "Sheet1":
                txt_file.write("\n### Explanation:\n")
                for _, row in df.iterrows():
                    lead = row["Lead"]
                    p_dur = row["P_dur (ms)"]
                    qrs_dur = row["QRS_dur (ms)"]
                    t_dur = row["T_dur (ms)"]
                    txt_file.write(
                        f"The ECG result for {lead} shows a P wave duration of {p_dur} ms, "
                        f"QRS complex duration of {qrs_dur} ms, and T wave duration of {t_dur} ms.\n"
                    )
        print(f"Converted: {file_path} -> {output_file_path}")

# تابع اصلی برای تبدیل تمام فایل‌های Excel در یک پوشه به متن
def convert_all_excel_to_text(input_folder, output_folder):
    # ایجاد پوشه خروجی اگر وجود ندارد
    os.makedirs(output_folder, exist_ok=True)
    
    # مرور فایل‌های Excel در پوشه ورودی
    for filename in os.listdir(input_folder):
        if filename.endswith((".xlsx", ".xls")):  # فقط فایل‌های Excel
            file_path = os.path.join(input_folder, filename)
            try:
                convert_excel_to_text(file_path, output_folder)
            except Exception as e:
                print(f"Error converting file {filename}: {e}")

# تابع برای ارسال متن به olama و دریافت پاسخ
def get_ollama_response(prompt, model="llama2"):
    process = subprocess.Popen(
        ["ollama", "chat", "--model", model, "--prompt", prompt],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    output, error = process.communicate()
    if error:
        return f"Error: {error}"
    return output.strip()

# # تنظیم پوشه‌های ورودی و خروجی
# input_folder = "path/to/excel_files"  # جایگزین کنید با مسیر واقعی فایل‌های Excel
# output_folder = "path/to/text_files"  # جایگزین کنید با مسیر خروجی مورد نظر

input_folder = "8_csv"  # جایگزین کنید با مسیر واقعی فایل‌های Excel
output_folder = "9_text"  # جایگزین کنید با مسیر خروجی مورد نظر


# اجرای تابع تبدیل
convert_all_excel_to_text(input_folder, output_folder)

# مثال: ارسال داده‌ها به olama و دریافت پاسخ
if __name__ == "__main__":
    # خواندن یکی از فایل‌های متنی تولید شده
    text_file = os.path.join(output_folder, "ecg_results patient001.txt")
    if os.path.exists(text_file):
        with open(text_file, "r", encoding="utf-8") as file:
            prompt = file.read()
        
        # ارسال متن به olama
        response = get_ollama_response(prompt)
        print("Ollama Response:")
        print(response)
    else:
        print(f"Text file not found: {text_file}")
# # تنظیم پوشه‌های ورودی و خروجی
# input_folder = "8_csv"  # جایگزین کنید با مسیر واقعی فایل‌های Excel
# output_folder = "9_text"  # جایگزین کنید با مسیر خروجی مورد نظر

# اجرای تابع تبدیل
# convert_all_excel_to_text(input_folder, output_folder)

 