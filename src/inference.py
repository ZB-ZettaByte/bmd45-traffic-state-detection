from ultralytics import YOLO
from pathlib import Path
import csv

# VehicleDetector class for detecting vehicles in images using a YOLO model
class VehicleDetector:
    def __init__(self, weights_path):
        """
        __init__: Init Function to load the YOLO model with the given weights

        Args:
            weights_path: str
                path to the YOLO model weights file
    
        """
        self.model = YOLO(weights_path)

    def predict_image(self, image_path):
        """
        predict_image: Predicts vehicles in a single image using the YOLO model

        Args:
            image_path: str
                path to the image file
        
        Returns:
            result: ultralytics.yolo.engine.results.Results
                YOLO prediction results for the image
        """
        results = self.model(image_path)
        return results[0]
    
    def predict_folder(self, folder_path):
        """
        predict_folder: Predicts vehicles in all images in a folder using the YOLO model

        Args:
            folder_path: str
                path to the folder containing images
        
        Returns:
            results: dict
                dictionary with image paths as keys and YOLO results as values
        """


        folder = Path(folder_path)
        # Get all image files in the folder that supports .jpg and .png
        image_paths = list(folder.glob("*.jpg")) + list(folder.glob("*.png"))

        # Create empty dict results and loop through each image to predict and store the results
        results = {}

        for img_path in image_paths:
            # Predict vehicles in the image and store the result in the dictionary
            results[str(img_path)] = self.predict_image(img_path)
        return results

    def count_vehicles(self, result):
        """
        count_vehicles: Counts the number of vehicles detected in a YOLO result

        Args:
            result: ultralytics.yolo.engine.results.Results
                YOLO prediction results for an image
        
        Returns:
            int: number of vehicles detected in the image

        """
        return len(result.boxes)

    def save_to_csv(self, results, output_path):
        """
        save_to_csv: Saves the vehicle count and density label for each image to a CSV file

        Args:
            results: dict
                dictionary with image paths as keys and YOLO results as values
            output_path: str
                path to save the CSV file
        
        """

        with open(output_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["image", "vehicle_count", "density_label"])
            for image_path, result in results.items():
                count = self.count_vehicles(result)
                label = self.get_density_label(count)
                writer.writerow([image_path, count, label])

    def get_density_label(self, count):
        """
        Returns a density label based on the vehicle count.

        Args:
            count: int
                number of vehicles detected
        
        Returns:
            str: density label ("unclear", "low", "medium", or "high")
        """

        # Define density labels based on vehicle count: 0-5: low, 6-15: medium, >15: high
        if count == 0:
            return "unclear"
        elif count <= 5:
            return "low"
        elif count <= 15:
            return "medium"
        else:
            return "high"


if __name__ == "__main__":
    detector = VehicleDetector("src/models/best.pt")
    results = detector.predict_folder("data/yolo/images")
    detector.save_to_csv(results, "outputs/sample_predictions.csv")
    print("inference done")