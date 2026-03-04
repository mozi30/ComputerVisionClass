# ComputerVisionClass

Projects in Computer Vision for university.

## Project Structure

```
ComputerVisionClass/
├── .gitignore                   # Python & CV-specific ignore rules
├── requirements.txt             # Shared Python dependencies
├── utils/                       # Shared utilities (image loading, display, saving)
│   ├── __init__.py
│   └── image_utils.py
├── 01_image_processing/         # Basic image operations & filtering
│   ├── README.md
│   └── main.py
├── 02_image_classification/     # HOG + SVM image classification
│   ├── README.md
│   └── main.py
├── 03_object_detection/         # Face/object detection with Haar cascades
│   ├── README.md
│   └── main.py
└── 04_image_segmentation/       # Thresholding & segmentation methods
    ├── README.md
    └── main.py
```

## Getting Started

1. **Create and activate a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate   # Linux/macOS
   venv\Scripts\activate      # Windows
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run a project**
   ```bash
   python 01_image_processing/main.py --image path/to/image.jpg
   ```

## Dependencies

- [NumPy](https://numpy.org/)
- [OpenCV](https://opencv.org/)
- [Pillow](https://python-pillow.org/)
- [Matplotlib](https://matplotlib.org/)
- [scikit-image](https://scikit-image.org/)
- [scikit-learn](https://scikit-learn.org/)
