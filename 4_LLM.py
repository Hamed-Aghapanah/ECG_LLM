# import ollama

# # بررسی مدل‌های موجود
# models = ollama.list()
# print(models)

# # بارگذاری یک مدل (اگر قبلاً نصب نشده باشد)
# ollama.pull("llama2")  # به جای "llama2" می‌توانید از مدل‌های دیگر استفاده کنید

# # تولید متن با استفاده از مدل
# response = ollama.generate(model="llama2", prompt="Hello, how are you?")
# print(response['response'])


import openai

# تنظیم کلید API
openai.api_key = "your-api-key"  # کلید API خود را اینجا وارد کنید

# ارسال درخواست به OpenAI
response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",  # یا "gpt-4" اگر دسترسی دارید
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "اطلاعات استخراج شده از ECG را تحلیل کن."}
    ],
    max_tokens=500  # حداکثر تعداد توکن‌های پاسخ
)

# دریافت پاسخ
report = response['choices'][0]['message']['content']
print(report)