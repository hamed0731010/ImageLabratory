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
