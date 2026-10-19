# Image Label Studio

A simple PyQt6 desktop app for whole-image classification.

## Run

```powershell
python -m pip install -r requirements.txt
python main.py
```

Choose a folder containing images, enter classes separated by commas (for example `cat, dog, car`), and click **Set classes**. Select a class button to copy or move the current image into a matching subfolder and advance automatically.

The viewer loads the original image and only scales the on-screen view to fit; use the mouse wheel to zoom and drag to pan. **Rotate 90°** and **Mirror** apply to the current image and are saved when you label it. **Rotate all source images 90°** permanently rewrites every source image after confirmation.

Use **Save setup** to store the folder and classes in the local `image_label_studio.db` SQLite database. The most recently saved setup is restored automatically the next time the app starts, and **Load from database** lets you choose another saved setup.

The current image position is saved too, so restarting the program resumes from the last image you were reviewing.

The sidebar also includes **PyTorch → ONNX**. Complete models and TorchScript files can be exported directly. For a state dictionary, select **State dictionary**, then provide the Python architecture file and model class name.

The **ONNX classifier** feature accepts an ONNX model, labels file (`.json`, `.txt`, or `.csv`), and image directory. It predicts every image and copies or moves each image into a subfolder named for its predicted class. Input size can be detected automatically or entered manually as `WIDTH,HEIGHT`.

The **PyTorch classifier** provides the same workflow for complete PyTorch models, TorchScript models, and state dictionaries. State dictionaries require the matching architecture Python file and model class.

The **OCR keyword classifier** uses Tesseract and a JSON keyword map. It counts each keyword for every class, assigns the image to the class with the highest count, optionally handles unmatched images, and writes `ocr_keyword_report.csv` in the source folder.

The **Train classifier** feature accepts an ImageFolder-style dataset, supports EfficientNet-B1, ResNet-18, and MobileNet-V3, and creates training artifacts: `best.pt`, `last.pt`, `model.onnx`, `history.json`, `classes.json`, and `summary.json`. A dataset may contain `train_data` and `val_data` folders, or class folders directly for an automatic train/validation split.

The **Image clustering** feature creates visual embeddings with pretrained ResNet-18, then groups images with K-Means or Agglomerative clustering. It creates `cluster_###` folders and writes `clusters.csv` and `summary.json`.
