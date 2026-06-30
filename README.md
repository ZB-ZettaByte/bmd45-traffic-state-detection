# BMD-45 Traffic State Detection

A small traffic-camera computer vision project that detects vehicles and assigns a simple traffic density state: `unclear`, `low`, `medium`, or `high`.

The project uses a fine-tuned YOLO26n model, a small streamed subset of the BMD-45 Bengaluru Mobility Dataset, a command-line inference script, and a Streamlit demo app for uploading traffic images.

## Demo

[VIDEO_LINK_HERE]

## Dataset

This project uses the BMD-45 subset from Hugging Face:

- Dataset: `iisc-aim/BMD-45`
- Training subset: 150 images
- Demo subset: 30 images
- Source split for training subset: first 150 samples from `train`
- Source split for demo subset: first 30 samples from `val`

The frames were selected with the Hugging Face `datasets` library using `streaming=True`. This avoids downloading the full 45k-image dataset locally. The subset selection is implemented in `src/load_data.py`:

```python
ds = load_dataset("iisc-aim/BMD-45", streaming=True)
train_subset = ds["train"].take(150)
demo_subset = ds["val"].take(30)
```

The selected frames are converted from COCO-style annotations to YOLO label files and saved under:

```text
data/yolo/images/
data/yolo/labels/
```

Generated dataset files are ignored by Git.

## Model Setup

The detection model is YOLO26n fine-tuned on the BMD-45 subset using Google Colab with a T4 GPU.

Training configuration:

- Model: YOLO26n
- Dataset: BMD-45 streamed subset
- Training images: 150
- Epochs: 80
- Image size: 640
- Batch size: 16
- Augmentation: enabled with `augment=True`
- Training environment: Google Colab, T4 GPU

Training notebook: https://colab.research.google.com/drive/1lakC27oJayAtklOCSkJTtABzhzZApsY7?usp=sharing

The trained model weights should be saved as:

```text
src/models/best.pt
```

Place `best.pt` inside the `src/models/` folder before running inference or launching the Streamlit app. The deployment expects this file to be available at `src/models/best.pt`.

## Training in Google Colab

Use the Google Colab notebook to train the YOLO26n model with GPU support:

https://colab.research.google.com/drive/1lakC27oJayAtklOCSkJTtABzhzZApsY7?usp=sharing

Training steps:

1. Open the Colab notebook link.
2. In Colab, go to `Runtime` > `Change runtime type`.
3. Select `T4 GPU` as the hardware accelerator.
4. Run the notebook cells to install dependencies and prepare the training environment.
5. Load the BMD-45 subset from Hugging Face using streaming.
6. Convert the selected BMD-45 annotations to YOLO format.
7. Train YOLO26n using `epochs=80`, `imgsz=640`, `batch=16`, and `augment=True`.
8. After training finishes, download the best model weights.
9. Save the downloaded weights locally as `src/models/best.pt`.

## Density Label Logic

Traffic density is assigned from the detected vehicle count:

| Vehicle count | Density label |
| --- | --- |
| 0 | `unclear` |
| 1-5 | `low` |
| 6-15 | `medium` |
| 16+ | `high` |

This logic is implemented in `src/inference.py`.

## Project Structure

```text
.
├── app/
│   └── main.py              # Streamlit upload and visualization app
├── src/
│   ├── convert.py           # COCO bbox to YOLO label conversion
│   ├── inference.py         # YOLO inference, counting, CSV output, density labels
│   ├── load_data.py         # Streams BMD-45 subset and converts it to YOLO format
│   ├── models/
│   │   └── best.pt          # Trained YOLO model weights
│   └── visualization.py     # Draws sample bounding boxes with OpenCV
├── outputs/
│   ├── sample_predictions.csv or sample_predictions.json
│   └── sample_detection_images/  # Not more than 10
├── pyproject.toml           # Python project metadata and dependencies
├── uv.lock                  # Locked dependency versions
├── README.md
└── LICENSE
```

Local generated files may include:

```text
data/
outputs/
yolo_data.zip
```

These are ignored by Git.

## Application Setup

### 1. Clone the repository

```bash
git clone https://github.com/ZB-ZettaByte/bmd45-traffic-state-detection.git
cd bmd45-traffic-state-detection
```

### 2. Install uv

If `uv` is not installed, install it based on your operating system. See the official Astral installation guide for more options:

https://docs.astral.sh/uv/getting-started/installation/#installation-methods

For macOS and Linux:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

For Windows PowerShell:

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Restart the terminal if needed so the `uv` command is available.

### 3. Install dependencies

This project targets Python 3.12 and uses `uv.lock` for reproducible installs.

```bash
uv sync
```

Core dependencies include:

- `datasets`
- `opencv-python`
- `streamlit`
- `ultralytics`

### 4. Add model weights

After training in Google Colab, download the trained YOLO weights into the `src/models/` folder as shown in [Training in Google Colab](#training-in-google-colab):

```text
src/models/best.pt
```

Both `src/inference.py` and `app/main.py` currently load this file from `src/models/best.pt`.

Beginner checklist before running the app:

- Confirm `src/models/best.pt` exists.
- Run `uv sync` after cloning the project.
- Use `uv run streamlit run app/main.py` if `streamlit` is not available globally.
- Open `http://localhost:8501` after Streamlit starts.

### 5. Prepare the dataset subset

Run the data loader to stream the subset from Hugging Face and convert annotations to YOLO format:

```bash
uv run python src/load_data.py
```

Expected generated folders:

```text
data/yolo/images/
data/yolo/labels/
```

### 6. Optional: Generate sample annotated images

To draw OpenCV bounding boxes on sample training images:

```bash
uv run python src/visualization.py
```

Annotated images are saved to:

```text
outputs/sample_detection_images/
```

### 7. Run batch inference

Run inference over images in `data/yolo/images` and write vehicle counts plus density labels to a CSV:

```bash
uv run python src/inference.py
```

Output:

```text
outputs/sample_predictions.csv
```

### 8. Start the Streamlit App

Launch the app locally using:

```bash
uv run streamlit run app/main.py
```

By default, the Streamlit app starts at:

```text
http://localhost:8501
```

Then upload one or more traffic images. The app displays the original image, YOLO detections, total vehicle count, per-class counts, and the density label.

## Inference Workflow

The inference pipeline is implemented by `VehicleDetector` in `src/inference.py`:

1. Load YOLO weights from `src/models/best.pt`.
2. Run prediction on one image or a folder of images.
3. Count detected bounding boxes.
4. Convert the count into a density label.
5. Save results to CSV when running the script directly.

Primary command:

```bash
uv run python src/inference.py
```

## Streamlit App

The Streamlit app in `app/main.py` provides an upload-based demo:

- Upload one or more `.jpg`, `.jpeg`, or `.png` images.
- Run YOLO detection with `src/models/best.pt`.
- Show original and annotated images side by side.
- Display total vehicle count.
- Display density label.
- Show a table of detected vehicle classes and counts.

Start command:

```bash
streamlit run app/main.py
```

Default local app URL:

```text
http://localhost:8501
```

## Deployment

The app can be deployed to Streamlit Cloud for free.

### Streamlit Cloud Setup

1. Push the repository to GitHub.
2. Go to [share.streamlit.io](https://share.streamlit.io).
3. Log in to Streamlit Cloud.
4. Connect your GitHub account.
5. Click `Create app`.
6. Choose `Deploy a public app from GitHub`.
7. In `Repository`, select the GitHub repository, for example:

   `ZB-ZettaByte/bmd45-traffic-state-detection`

8. In `Branch`, select:

   `main`

9. In `Main file path`, enter:

   `app/main.py`

10. Click `Deploy`.

The model weights are included in the repository at:

`src/models/best.pt`

Because the model file is already included, no extra model download or setup is needed on Streamlit Cloud.

---

### Files Needed for Deployment

Before deploying, make sure these files are committed to GitHub:

- `app/main.py`
- `src/inference.py`
- `src/models/best.pt`
- `pyproject.toml`
- `uv.lock`
- `packages.txt`
- `README.md`

---

### Why `packages.txt` Was Added

The app uses Ultralytics YOLO, which imports OpenCV through `cv2`. On Streamlit Cloud, OpenCV may require additional Linux system libraries that are not installed by default.

During deployment, the app showed missing library errors such as:

- `ImportError: libGL.so.1: cannot open shared object file`
- `ImportError: libgthread-2.0.so.0: cannot open shared object file`

To fix these deployment errors, a `packages.txt` file was added.

The final `packages.txt` contains:

```txt
libgl1
libglib2.0-0t64
```

## Development Environment

Local development was done in VS Code. Model training was done in Google Colab using a T4 GPU.

## Known Limitations

- The training subset is very small: only 150 images. This limits model accuracy and generalization.
- The density estimate is count-based, so it does not consider road area, lane count, camera perspective, speed, or vehicle spacing.
- Some BMD-45 annotations appear incorrect, which can affect training quality and evaluation.
- Uploaded images that are blurry, dark, occluded, far from the camera, or taken from unusual angles may produce unreliable detections.

## AI Assistance

Claude from Anthropic was used for assistance with:

- OpenCV bounding box drawing logic in `src/visualization.py`
- COCO to YOLO conversion formula in `src/convert.py`

## Next Steps

Future improvements for this project include:

- **More training data**: Only 150 images were used. Training on a larger BMD-45 subset would significantly improve detection accuracy.
- **Better density logic**: Weight by vehicle size or use bounding box area coverage instead of raw count.
- **Video support**: Extend the app to handle video files or live CCTV streams.
- **Multi-camera support**: Monitor multiple camera feeds simultaneously for city-wide traffic estimation.

## License

This project is licensed under the MIT License. See `LICENSE` for details.
