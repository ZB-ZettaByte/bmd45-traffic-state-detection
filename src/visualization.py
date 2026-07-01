import cv2
from pathlib import Path

# draws bounding boxes on sample images and saves them to sample_detection_images/
class SampleVisualizer:

    def __init__(self, images_dir, labels_dir, output_dir):
        """
        __init__: Init Function sets up the paths for images, labels, and output folder

        Args:
            images_dir: str
                path to folder containing images
            labels_dir: str 
                path to folder containing YOLO label files
            output_dir: str
                path to save annotated images

        """
        # Set up paths for images, labels, and output folder
        self.images_dir = Path(images_dir)
        self.labels_dir = Path(labels_dir)
        self.output_dir = Path(output_dir)

        # Create the output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True) 
        
    # class names for the BMD-45 dataset
    CLASS_NAMES = {
        0: "Hatchback", 1: "Sedan", 2: "SUV", 3: "MUV", 4: "Bus",
        5: "Truck", 6: "Three-wheeler", 7: "Two-wheeler", 8: "LCV",
        9: "Mini-bus", 10: "Tempo", 11: "Bicycle", 12: "Van", 13: "Other"
    }
    
    def draw_boxes(self, image_path, label_path):
        """
        draw_boxes: Draws bounding boxes on an image based on YOLO label file.

        Args:
            image_path: Path
                path to the image file
            label_path: Path 
                path to the corresponding YOLO label file
        
        Returns:
            image: numpy.ndarray
                annotated image with bounding boxes and labels

        """

        image = cv2.imread(str(image_path))
        img_height, img_width = image.shape[:2]
        
        # Read the YOLO label file and draw boxes
        with open(label_path, "r") as f:
            for line in f:
                # Split the line into parts: class_id, cx, cy, w, h
                parts = line.strip().split()

                # Convert YOLO format back to pixel coordinates
                category = int(parts[0])
                cx = float(parts[1])
                cy = float(parts[2])
                w = float(parts[3])
                h = float(parts[4])
                
                # Convert normalized coordinates to pixel values
                x1 = int((cx - w / 2) * img_width)
                y1 = int((cy - h / 2) * img_height)
                x2 = int((cx + w / 2) * img_width)
                y2 = int((cy + h / 2) * img_height)
                
                # Draw the rectangle and label on the image
                label = self.CLASS_NAMES.get(category, str(category))

                # Draw the bounding box and label on the image
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 255), 2)

                # Draw the label above the bounding box
                cv2.putText(image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        return image

    def save_samples(self, num_samples=10):
        """
        save_samples: Saves annotated sample images to the output folder.

        Args:
            num_samples: int
                number of sample images to save
        
        """
        
        # Get a list of image files in the images folder sorted by name
        image_files = sorted(self.images_dir.glob("train_*.jpg"))[:num_samples]

        # Annotate and save each sample image
        for image_path in image_files:
            # Get the label file
            label_path = self.labels_dir / (image_path.stem + ".txt")
            
            # Check if the label file exists
            if not label_path.exists():
                continue

            # Draw bounding boxes on the image
            annotated = self.draw_boxes(image_path, label_path)
            # Save the annotated image to the output directory
            output_path = self.output_dir / image_path.name
            
            # Save the annotated image
            cv2.imwrite(str(output_path), annotated)
            print(f"Saved {output_path}")


if __name__ == "__main__":
    visualizer = SampleVisualizer(
        images_dir="data/yolo/images",
        labels_dir="data/yolo/labels",
        output_dir="outputs/sample_detection_images"
    )
    visualizer.save_samples(num_samples=10)
