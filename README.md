# BMD-45 Traffic State Detection

A small traffic-camera computer vision project that detects vehicles and assigns a simple traffic density state: **unclear**, **low**, **medium**, or **high**.

The project uses a fine-tuned YOLO26n model, a small streamed subset of the BMD-45 Bengaluru mobility dataset, a command-line inference script, and a Streamlit demo app for uploading traffic images.

## Demo

A short demo video showing the Streamlit app workflow is available here:

- **Demo Video:** [https://youtu.be/bCPGBkK0z5g](https://youtu.be/bCPGBkK0z5g)
- **Live Demo:** [https://bmd45-traffic-state-detection.streamlit.app/](https://bmd45-traffic-state-detection.streamlit.app/)

## Dataset

This project uses the [BMD-45 dataset](https://huggingface.co/datasets/iisc-aim/BMD-45) from Hugging Face:

- Dataset: [https://huggingface.co/datasets/iisc-aim/BMD-45](https://huggingface.co/datasets/iisc-aim/BMD-45)
- Training subset: 150 images
- Demo subset: 30 images
- Source split for training subset: first 150 samples from training split
- Source split for demo subset: first 30 samples from validation split

The frames were selected with the Hugging Face **datasets** library using **streaming=True** to avoid downloading the full 45k-image dataset. The subset selection is implemented in **src/load_data.py**:

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

Generated dataset files are not committed to Git. They can be recreated by running the dataset conversion and preparation scripts

## Model Setup

The detection model is YOLO26n fine-tuned on the BMD-45 subset using Google Colab with a T4 GPU

Training configuration:

- Model: YOLO26n
- Dataset: BMD-45 streamed subset
- Training images: 150
- Epochs: 80
- Image size: 640
- Batch size: 16
- Augmentation: enabled with **augment=True**
- Training environment: Google Colab, T4 GPU

Training notebook: https://colab.research.google.com/drive/1lakC27oJayAtklOCSkJTtABzhzZApsY7?usp=sharing

The trained model weights should be saved as:

```text
src/models/best.pt
```

Place **best.pt** inside the **src/models/** folder before running inference or launching the Streamlit app. The deployment expects this file to be under **src/models/**

<a id="data-preparation-for-colab"></a>

## Data Preparation for Colab

Before using the Colab notebook for training, zip the **sample_detection_images/** folder created by running the visualization script:

```bash
uv run python src/visualization.py
zip -r sample_detection_images.zip sample_detection_images/
```
<a id="training-in-google-colab"></a>

## Training in Google Colab

Use the Google Colab notebook to train the YOLO26n model with GPU support:

https://colab.research.google.com/drive/1lakC27oJayAtklOCSkJTtABzhzZApsY7?usp=sharing

The training steps are included inside the Google Colab notebook.

After training finishes, download the best model weights and save them inside the **src/models/** folder as **best.pt**

## Density Label Logic

Traffic density is assigned from the detected vehicle count:

| Vehicle count | Density label |
| --- | --- |
| 0 | `unclear` |
| 1-5 | `low` |
| 6-15 | `medium` |
| 16+ | `high` |

This logic is implemented in **src/inference.py**

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
│   ├── sample_predictions.csv
│   └── sample_detection_images/  # Not more than 10
├── pyproject.toml           # Python project metadata and dependencies
├── uv.lock                  # Locked dependency versions
├── README.md
└── LICENSE
```

Generated files may include:

```text
data/
outputs/
yolo_data.zip
```

These files are not added to Git

## Environment & Tech Stack

- **VS Code**: Development
- **Google Colab**: YOLO26n model training with a T4 GPU

![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)
![YOLO](https://img.shields.io/badge/YOLO-Ultralytics-111F68)
![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?logo=opencv&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white)
![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Datasets-FFD21E?logo=huggingface&logoColor=black)
![uv](https://img.shields.io/badge/uv-Python%20Manager-DE5FE9)
![Google Colab](https://img.shields.io/badge/Google%20Colab-T4%20GPU-F9AB00?logo=googlecolab&logoColor=black)
![VS Code](https://img.shields.io/badge/VS%20Code-007ACC?logo=visualstudiocode&logoColor=white)

## Application Setup

### 1. Clone the repository

```bash
git clone https://github.com/ZB-ZettaByte/bmd45-traffic-state-detection.git

cd bmd45-traffic-state-detection
```

### 2. Install uv

If **uv** is not installed, install it based on your operating system. See the official Astral installation guide for more options:

https://docs.astral.sh/uv/getting-started/installation/#installation-methods

For macOS and Linux:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

For Windows PowerShell:

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Restart the terminal if needed so the **uv** command is available

### 3. Install dependencies

This project targets Python 3.12 and uses **uv.lock** for reproducible installs

```bash
uv sync
```

Core dependencies include:

- datasets
- opencv-python
- streamlit
- ultralytics

### 4. Data Collection

Run the data loader to download the selected BMD-45 subset from Hugging Face

```bash
uv run python src/load_data.py
```

This step collects the traffic-camera images and annotation data needed for the project

### 5. Data Preparation

Run the conversion script to convert the selected BMD-45 annotations into YOLO format

```bash
uv run python src/convert.py
```

Expected generated folders:

```text
data/yolo/images/
data/yolo/labels/
```

Generated dataset files are not committed to Git. They can be recreated by running the data collection and conversion scripts

### 6. Verify Data Annotations

This step draws the converted YOLO bounding boxes on a few sample images to visually verify the COCO to YOLO conversion worked:

```bash
uv run python src/visualization.py
```

Annotated images are saved to **outputs/sample_detection_images/**

### 7. Model Training

Model training is done in Google Colab. Before training, first create the Colab training zip file using the [Data Preparation for Colab](#data-preparation-for-colab) section. Then train the model using the [Training in Google Colab](#training-in-google-colab) section

After training finishes, download the best model weights and save them as:

```text
src/models/best.pt
```

### 8. Run Inference

Then run batch inference over images in **data/yolo/images/** and write vehicle counts plus density labels to a CSV file

```bash
uv run python src/inference.py
```
Output saved to **outputs/sample_predictions.csv**

### 9. Start the Streamlit App

Run the Streamlit app locally:

```bash
uv run streamlit run app/main.py
```

After Streamlit starts, open **http://localhost:8501**

The app lets a user upload traffic images and view the original image, image with detections, vehicle counts, density label, and notes/limitations.

## Inference Workflow

The inference pipeline is implemented by VehicleDetector in **src/inference.py:**

1. Load YOLO weights from **src/models/best.pt**
2. Run prediction on one image or a folder of images
3. Count detected bounding boxes
4. Convert the count into a density label
5. Save results to CSV when running the script directly

Primary command:

```bash
uv run python src/inference.py
```

## Streamlit App

The Streamlit app in **app/main.py **provides an upload-based demo:

- Upload one or more **.jpg, .jpeg, or .png** images
- Run YOLO detection with **src/models/best.pt**
- Show original and annotated images side by side
- Display total vehicle count
- Display density label
- Show a table of detected vehicle classes and counts

Start command:

```bash
uv run streamlit run app/main.py
```

Default local app URL:

```text
http://localhost:8501
```

## Deployment

The app can be deployed to Streamlit Cloud for free.

### Streamlit Cloud Setup

1. Push the repository to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Log in to Streamlit Cloud
4. Connect your GitHub account
5. Click **Create app**
6. Choose **Deploy** a public app from GitHub.
7. In **Repository**, select the GitHub repository, for example: **ZB-ZettaByte/bmd45-traffic-state-detection**
8. In **Branch**, select: **main**
9. In Main file path, enter: **app/main.py**
10. Click **Deploy**

The model weights are included in the repository at: `src/models/best.pt`

Because the model file is already included, no extra model download or setup is needed on Streamlit Cloud.

---

### Why `packages.txt` Was Added

The app uses Ultralytics YOLO, which imports OpenCV through **cv2**. On Streamlit Cloud, OpenCV may require additional Linux system libraries that are not installed by default

During deployment, the app showed missing library errors such as:

- **ImportError: libGL.so.1: cannot open shared object file**
- **ImportError: libgthread-2.0.so.0: cannot open shared object file**

To fix these deployment errors, a **packages.txt** file was added.

The final **packages.txt** contains:

```txt
libgl1
libglib2.0-0t64
```

## Known Limitations

- The training subset is very small: only 150 images. This limits model accuracy and generalization
- The density estimate is count-based, so it does not consider road area, lane count, camera perspective, speed, or vehicle spacing
- Some BMD-45 annotations appear incorrect, which can affect training quality and evaluation
- Uploaded images that are blurry, dark, occluded, far from the camera, or taken from unusual angles may produce unreliable detections

## Next Steps

Future improvements for this project include:

- **More training data**: Only 150 images were used. Training on a larger BMD-45 subset would significantly improve detection accuracy
- **Better density logic**: Weight by vehicle size or use bounding box area coverage instead of raw count
- **Video support**: Extend the app to handle video files or live CCTV streams
- **Multi-camera support**: Monitor multiple camera feeds simultaneously for city-wide traffic estimation

## AI Assistance

AI tools were used to support code implementation, Streamlit UI improvements, and README formatting. All AI-generated suggestions were reviewed, edited, and tested before being included in the final project.

No private data, API keys, credentials, or restricted datasets were used.

### Tools Used

- [Claude from Anthropic](https://www.anthropic.com/claude)
- [ChatGPT from OpenAI](https://chatgpt.com/)

These tools were used only for chat-based assistance.

### Prompts Used

<details>
<summary><strong>OpenCV Bounding Box Drawing</strong></summary>

Prompt used:

> Explain how to visualize YOLO-format label files on traffic images using OpenCV. I want to understand how to read a YOLO label file, convert normalized center-based coordinates back to pixel coordinates, draw bounding boxes, add class labels, and return the annotated image. Please keep the explanation beginner, simpler way and show a small example function without changing the original dataset annotations.

Used for guidance in the `draw_boxes(self, image_path, label_path)` method in `src/visualization.py`.

</details>

<details>
<summary><strong>Streamlit GitHub Link and Footer Improvements</strong></summary>

Prompt used:

> Please help me fix small UI issues in only these two Streamlit methods: `display_github_link(self)` and `show_notes_and_footer(self)`.
>
> Keep the current OOP structure.
>
> 1. For `display_github_link(self)`, make the GitHub link look cleaner by replacing the badge-style link with a small GitHub button using an inline SVG icon. Link it to the project GitHub repository.
>
> 2. For `show_notes_and_footer(self)`, remove the repeated notes sentence, keep the Notes and Limitations section simple, fix the footer HTML, and show only the copyright text.
>
> Do not make a big card header. Do not change existing method names. Do not remove comments or docstrings. Please provide replacement snippets only for `display_github_link(self)` and `show_notes_and_footer(self)`.

Used for guidance in the `display_github_link(self)` and `show_notes_and_footer(self)` methods in `app/main.py`.

</details>

<details>
<summary><strong>COCO to YOLO Bounding Box Conversion</strong></summary>

Prompt used:

> Explain the formula for converting a bounding box from COCO format to YOLO format. COCO uses [x, y, width, height] with x and y as the top-left corner. YOLO needs normalized center coordinates: x_center, y_center, width, and height. Please explain the calculation step by step in a beginner way so I can apply it in my existing conversion method.

Used for guidance in the `convert_bbox(self, bbox, img_width, img_height)` method in `src/convert.py`.

</details>

<details>
<summary><strong>README Density Table and Project Structure Formatting</strong></summary>

Prompt used:

> Can you help me format two small README sections? First format the traffic density logic as a Markdown table using vehicle count and density label:
>
> - 0 vehicles = unclear
> - 1–5 vehicles = low
> - 6–15 vehicles = medium
> - 16+ vehicles = high
>
> Also help me format the Project Structure section in a README style. Keep it simple

Used for guidance in formatting the Density Label Logic and Project Structure sections in `README.md`.

</details>

<details>
<summary><strong>README Technology Badges Formatting</strong></summary>

Prompt used:

> Can you help me format simple README technology badges for my project? I want badges for the main tools used, such as Python 3.12, YOLO, OpenCV, Streamlit, Hugging Face Datasets, uv, Google Colab, and VS Code. Please keep the badges clean for a GitHub README.

Used for guidance in formatting the technology badges section in `README.md`.

</details>

## License

This project is licensed under the MIT License. See `LICENSE` for details.
