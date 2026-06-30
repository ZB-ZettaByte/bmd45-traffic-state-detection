import sys
from pathlib import Path
import streamlit as st
from PIL import Image

# Required to import the VehicleDetector class from the src folder
sys.path.append("src")
from inference import VehicleDetector

# TrafficDensityApp class for the Streamlit application to detect traffic density from uploaded images
class TrafficDensityApp:
    def __init__(self, model_path: str):
        """
        __init__: 

        Args:
            model_path: str
                path to the YOLO model weights file

        """

        self.detector = VehicleDetector(model_path)
        self.output_dir = Path("outputs")

    def setup_page(self):
        """
        setup_page: Sets up the Streamlit page config and title for the application
        """
        st.set_page_config(page_title="Traffic State Detection", layout="wide")

        st.title("Traffic State Detection")
        
        self.display_github_link()
        st.write(
            "This app shows traffic-camera images, vehicle boxes, vehicle counts, "
            "and a simple density label."
        )
    
    def display_github_link(self):
        """
        display_github_link: Displays a GitHub link in the Streamlit app
        """
        st.markdown(
            """
            <a href="https://github.com/ZB-ZettaByte/bmd45-traffic-state-detection/"
               target="_blank"
               style="display: inline-flex; align-items: center; gap: 6px; padding: 6px 10px; border: 1px solid #d0d7de; border-radius: 6px; color: #24292f; text-decoration: none; font-size: 14px; font-weight: 500;">
                <svg height="16" width="16" viewBox="0 0 16 16" fill="currentColor" aria-hidden="true">
                    <path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82A7.65 7.65 0 0 1 8 3.86c.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0 0 16 8c0-4.42-3.58-8-8-8Z"></path>
                </svg>
                GitHub
            </a> 
            """,
            unsafe_allow_html=True
        )

    def upload_images(self):
        """
        upload_images: Creates a file uploader in the Streamlit app for users to upload an image
        """
        return st.file_uploader(
            "Upload a traffic image to see the output",
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=True
        )

    def save_uploaded_image(self, uploaded_file, image_number):
        """
        save_uploaded_image: Saves the uploaded image to a temporary path for processing

        Args:
            uploaded_file: UploadedFile
                the uploaded image file from Streamlit's file uploader
       
        Returns:
            image: PIL.Image.Image
                the uploaded image opened as a PIL Image
        """
        self.output_dir.mkdir(exist_ok=True)

        image = Image.open(uploaded_file).convert("RGB")

        image_path = self.output_dir / f"image_{image_number}.jpg"
        image.save(image_path)
        
        return image, image_path

    def detect_vehicles(self, image_path):
        """
        detect_vehicles: Detects vehicles in the uploaded image using the YOLO model

        Returns:
            result: ultralytics.yolo.engine.results.Results
                YOLO prediction results for the image
            
            vehicle_counts: dict
                dictionary with vehicle classes as keys and their counts as values
            
            count: int
                total number of vehicles detected in the image
            
            label: str
                density label based on the vehicle count: "unclear", "low", "medium", or "high"
        """
        result = self.detector.predict_image(str(image_path))
        vehicle_counts = self.get_vehicle_counts(result)

        count = sum(vehicle_counts.values())
        label = self.detector.get_density_label(count)

        return result, vehicle_counts, count, label

    def get_vehicle_counts(self, result):
        """ 
        get_vehicle_counts: Counts the number of vehicles detected in a YOLO result and returns a
        dictionary with vehicle classes as keys and their counts as values

        Args:
            result: ultralytics.yolo.engine.results.Results
                YOLO prediction results for an image
            
        Returns:
            dict: vehicle_counts
                dictionary with vehicle classes as keys and their counts as values  
        """
        vehicle_counts = {}

        for box in result.boxes:
            class_number = int(box.cls[0])
            class_name = result.names[class_number]

            if class_name in vehicle_counts:
                vehicle_counts[class_name] += 1
            else:
                vehicle_counts[class_name] = 1

        return vehicle_counts

    def show_results(self, image, result, count, label):
        """ 
        show_results: Displays the original image, annotated image with detections, vehicle count, and density label in the Streamlit app

        Args:
            image: PIL.Image.Image
                the original uploaded image
            result: ultralytics.yolo.engine.results.Results
                YOLO prediction results for the image
            count: int
                total number of vehicles detected in the image
            label: str
                density label based on the vehicle count: "unclear", "low", "medium", or "high"
        """
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Original image")
            st.image(image, use_container_width=True)

            st.write("Total vehicles")
            st.header(count)

        with col2:
            st.subheader("Image with detections")
            annotated = result.plot()
            st.image(annotated, use_container_width=True)

            st.write("Density label")
            st.header(label)

    def show_vehicle_table(self, vehicle_counts):
        """
        show_vehicle_table: Displays a table of vehicle classes and their counts in the Streamlit app

        Args:
            vehicle_counts: dict
                dictionary with vehicle classes as keys and their counts as values
        
        """
        st.subheader("Vehicle counts")

        if len(vehicle_counts) == 0:
            st.write("No vehicles found in this image.")
            return

        table_data = []

        for vehicle_class, vehicle_count in vehicle_counts.items():
            table_data.append({
                "Vehicle class": vehicle_class,
                "Count": vehicle_count
            })

        st.table(table_data)

    def show_warning(self, label):
        """
        show_warning: Displays a warning message in the Streamlit app if the density label is "unclear"

        Args:
            label: str
                density label based on the vehicle count: "unclear", "low", "medium", or "high"
        """
        if label == "unclear":
            st.warning("Not enough detections to estimate density reliably.")
    
    def show_notes_and_footer(self):
        """
        show_notes_and_footer: Displays basic notes, limitations, GitHub link, and copyright footer
        """

        st.markdown("---")
        st.subheader("Notes and Limitations")
        st.write(
            "This app uses vehicle detections to estimate a simple traffic-density label."
        )

        st.write(
            "Results may be less accurate for blurry images, dark scenes, "
            "blocked vehicles, far-away vehicles, or images with unusual camera angles."
        )

        st.markdown(
            """
            <div style="text-align: center; color: #8c8c8c; font-size: 14px; border-top: 1px solid #e6e6e6; margin-top: 28px; padding-top: 16px;">
              © 2026 Sai Rithwik Kukunuri. All rights reserved.
            </div>
            """, 
            unsafe_allow_html=True
        )

    def run(self):
        """
        run: Runs the Streamlit application, handling image upload, vehicle detection, and displaying results
        """
        self.setup_page()

        uploaded_files = self.upload_images()

        if uploaded_files is None:
            st.info("Please upload one or more traffic images to display the output.")
            return
        
        for image_number, uploaded_file in enumerate(uploaded_files, start=1):
            st.divider()
            st.subheader(f"Image {image_number}: {uploaded_file.name}")

            image, image_path = self.save_uploaded_image(uploaded_file, image_number)
            result, vehicle_counts, count, label = self.detect_vehicles(image_path)
            
            self.show_results(image, result, count, label)
            self.show_vehicle_table(vehicle_counts)
            self.show_warning(label)
        self.show_notes_and_footer()

def main():
    """
    main: Entry point for the Streamlit application. Initializes the TrafficDensityApp with the YOLO model weights and runs the app.
    """
    app = TrafficDensityApp(model_path="src/models/best.pt")
    app.run()



if __name__ == "__main__":
    main()