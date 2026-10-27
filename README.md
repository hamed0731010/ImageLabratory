# Image Laboratory

یک نرم‌افزار دسکتاپ مبتنی بر `PyQt6` برای آماده‌سازی دیتاست‌های تصویری، برچسب‌گذاری دستی و خودکار، آموزش مدل و ارزیابی نتایج.

## Run

```powershell
python -m pip install -r requirements.txt
python main.py
```

پس از نصب، برنامه به‌صورت محلی روی ویندوز اجرا می‌شود. یک پوشه تصویر و کلاس‌ها را انتخاب کنید؛ سپس تصاویر با انتخاب کلاس به پوشه مربوطه کپی یا منتقل می‌شوند.

## قابلیت‌ها

### 1. برچسب‌گذاری دستی — Manual Labeling

نمایش تصویر، تصویر قبلی و بعدی، ردکردن، Undo، زوم، جابه‌جایی، چرخش، آینه‌کردن، برش و انتخاب Copy یا Move.

### 2. برچسب‌گذاری با مدل بینایی — VLM / JEV Labeling

هر تصویر به API سازگار با OpenAI ارسال می‌شود و بر اساس پاسخ مدل در پوشه برچسب قرار می‌گیرد.

ورودی‌ها: API Endpoint، API Token، نام مدل، حداکثر توکن خروجی، Prompt، JSON Schema، Dataset و پوشه خروجی. پاسخ باید یکی از فیلدهای `label`، `class`، `category` یا `name` را داشته باشد.

برای هر برچسب یک پوشه ساخته می‌شود. نتایج در JSONL و خلاصه اجرا در `summary.json` ذخیره می‌شوند. پردازش قابل ادامه است.

> مدل آنلاین به اینترنت نیاز دارد؛ مدل محلی با Endpoint محلی قابل استفاده است.

### 3. دسته‌بندی متنی — OCR Keyword Classifier

استخراج متن با Tesseract و دسته‌بندی تصاویر بر اساس کلمات کلیدی. گزارش `ocr_keyword_report.csv` نیز تولید می‌شود.

### 4. طبقه‌بندی با PyTorch — PyTorch Classifier

پشتیبانی از Complete Model، `TorchScript` و `State Dictionary` برای پیش‌بینی و انتقال تصاویر به پوشه کلاس.

### 5. آموزش مدل — Train Classifier

آموزش `EfficientNet-B1`، `ResNet-18` و `MobileNet-V3` با تنظیمات Epoch، Batch، اندازه تصویر و Resume. خروجی‌ها شامل `best.pt`، `last.pt`، `model.onnx`، `history.json`، `classes.json` و `summary.json` هستند.

### 6. ارزیابی مدل — Model Evaluation

محاسبه Accuracy، Precision، Recall، F1، Confusion Matrix و نمایش تصاویر اشتباه.

### 7. تبدیل مدل — PyTorch to ONNX

تبدیل مدل‌های PyTorch، TorchScript و State Dictionary به فرمت ONNX.

### 8. خوشه‌بندی تصاویر — Image Clustering

گروه‌بندی تصاویر با embeddingهای ResNet-18 و روش‌های `K-Means` یا `Agglomerative Clustering`.

### 9. تولید دیتاست مصنوعی — Template Generator

تولید نمونه با تغییر متن، فونت، اندازه، موقعیت، چرخش، رنگ و کنتراست.

### 10. مدیریت پروژه — Project Management

مسیر دیتاست، کلاس‌ها و آخرین موقعیت کار در `image_label_studio.db` ذخیره می‌شود.

## ساختار خروجی VLM

```text
labeled/
├── cat/
├── dog/
└── car/
```

## تکنولوژی‌ها

`Python`، `PyQt6`، `PyTorch`، `TorchVision`، `Pillow`، `ONNX`، `Tesseract` و `scikit-learn`
