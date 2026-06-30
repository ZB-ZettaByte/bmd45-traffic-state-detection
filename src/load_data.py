from datasets import load_dataset
from convert import COCOToYOLOConverter

# Main function to load the BMD-45 dataset and convert it to YOLO format
def main():
    """
    Main data loading function that:
    1. Loads the BMD-45 dataset from Hugging Face:
       https://huggingface.co/datasets/iisc-aim/BMD-45
    """

    # Load the BMD-45 dataset from Hugging Face
    ds = load_dataset("iisc-aim/BMD-45", streaming=True)

    # 150 training images and 30 demo images for testing
    train_subset = ds["train"].take(150)
    demo_subset = ds["val"].take(30)

    train_list = list(train_subset)
    demo_list = list(demo_subset)

    # Checkpoint: Print the number of images in each subset
    print(f"Train images: {len(train_list)}")
    print(f"Demo images: {len(demo_list)}")

    # Print the first sample from the train subsset to see the format of the data
    sample = train_list[0]
    print(sample["objects"])

    # Convert the dataset to YOLO format and save both splits to disk
    converter = COCOToYOLOConverter("data/yolo")
    converter.convert_dataset(train_list, "train")
    converter.convert_dataset(demo_list, "demo")

if __name__ == "__main__":
    main()