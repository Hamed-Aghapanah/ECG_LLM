import numpy as np
import matplotlib.pyplot as plt

# خواندن تصویر
image = plt.imread('p1.tif')

# بررسی ابعاد تصویر
print("ابعاد تصویر اصلی:", image.shape)  # باید (ارتفاع, عرض, کانال‌ها) باشد


a = 160
b = 40
# ایجاد یک آرایه خالی برای ذخیره تصویر نهایی
# ابعاد آرایه باید (ارتفاع تصویر * 40, عرض تصویر * 20, کانال‌های رنگی) باشد
image20x40 = np.zeros((image.shape[0] * b, image.shape[1] * a, image.shape[2]), dtype=image.dtype)

# چسباندن تصویر 20 بار افقی و 40 بار عمودی
for i in range(a):  # تکرار افقی
    for j in range(b):  # تکرار عمودی
        # محاسبه موقعیت برای قرار دادن تصویر
        start_row = j * image.shape[0]
        end_row = (j + 1) * image.shape[0]
        start_col = i * image.shape[1]
        end_col = (i + 1) * image.shape[1]
        
        # قرار دادن تصویر در موقعیت مناسب
        image20x40[start_row:end_row, start_col:end_col, :] = image

# نمایش تصویر نهایی
plt.imshow(image20x40)
plt.axis('off')  # غیرفعال کردن محورها
plt.show()
plt.imsave('pattern.jpg', image20x40,dpi = 600)