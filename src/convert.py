from pathlib import Path
from PIL import Image

# COCOToYOLOConverter class for converting COCO format annotations to YOLO format
class COCOToYOLOConverter:

    def __init__(self, output_dir):
        """
        __init__: Init Function to set up output folders for images and labels

        """
        self.images_dir = Path(output_dir) / "images"
        self.labels_dir = Path(output_dir) / "labels"
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.labels_dir.mkdir(parents=True, exist_ok=True)

    def convert_bbox(self, bbox, img_width, img_height):
        """
        convert_bbox: Converts a single bounding box from COCO format to YOLO format.
        COCO: x, y, w, h
        YOLO: cx, cy, w, h
        
        Args:
            bbox: list
                bounding box in COCO format [x, y, w, h]
            img_width: int
                image width in pixels
            img_height: int
                image height in pixels

        Returns:
            tuple: (cx, cy, w_norm, h_norm)
        """
        
        # Convert COCO bbox to YOLO format
        x = bbox[0]
        y = bbox[1]
        w = bbox[2]
        h = bbox[3]

        # Convert to YOLO format
        cx = (x + w / 2) / img_width
        cy = (y + h / 2) / img_height
        w_norm = w / img_width
        h_norm = h / img_height


        return cx, cy, w_norm, h_norm

    def convert_sample(self, sample, image_name):
        """
        convert_sample: Saves one image and its annotation .txt file

        Args:
            sample: dict
                a single dataset sample containing 'image' and 'objects'
            image_name: str
                name to save the image and label file as
        """

        # Convert the image to RGB and get its dimensions
        image = sample["image"].convert("RGB")
        img_width, img_height = image.size

        # save image
        image_path = self.images_dir / f"{image_name}.jpg"
        image.save(image_path)

        # save label file
        label_path = self.labels_dir / f"{image_name}.txt"
        bboxes = sample["objects"]["bbox"]
        categories = sample["objects"]["categories"]

        # Save the bounding boxes and categories in YOLO format
        with open(label_path, "w") as f:
            for bbox, category in zip(bboxes, categories):
                cx, cy, w_norm, h_norm = self.convert_bbox(
                    bbox, 
                    img_width, 
                    img_height
                )
                
                # Write the category and bounding box to the label file
                f.write(
                    f"{category} "
                    f"{cx:.6f} "
                    f"{cy:.6f} "
                    f"{w_norm:.6f} "
                    f"{h_norm:.6f}\n"
                )
    
    def convert_dataset(self, subset, split_name):
        """
        convert_dataset: Converts all samples in a subset and saves them to disk.

        Args:
            subset: iterable
                an iterable of dataset samples
            split_name: str
                name of the dataset split (e.g., "train", "val")
        """

        # Loop through each sample in the subset and convert it
        for i, sample in enumerate(subset):
            # Format the index with leading zeros to make it 4 digits
            image_name = f"{split_name}_{i:04d}"
            
            # Convert and save the sample
            self.convert_sample(sample, image_name)
        print(f"Done. Saved to {self.images_dir} and {self.labels_dir}")