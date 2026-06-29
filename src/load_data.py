from datasets import load_dataset

def main():
    """
    Main data loading function that:
    1. Loads the BMD-45 dataset from Hugging Face:
       https://huggingface.co/datasets/iisc-aim/BMD-45
    2. Uses streaming to avoid downloading the full dataset during subset selection.
    3. Takes the first 150 images from the train split for fine-tuning/adaptation.
    4. Takes the first 30 images from the val split for demo inference
    """
    ds = load_dataset("iisc-aim/BMD-45", streaming=True)

    train_subset = ds["train"].take(150)
    demo_subset = ds["val"].take(30)

    train_list = list(train_subset)
    demo_list = list(demo_subset)

    print(f"Train images: {len(train_list)}")
    print(f"Demo images: {len(demo_list)}")

    # Print the first sample from the train subsset to see the format of the data
    sample = train_list[0]
    print(sample["objects"])

if __name__ == "__main__":
    main()