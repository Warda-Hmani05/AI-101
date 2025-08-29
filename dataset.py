import os
import random
import shutil
import datetime

# Global variables
STARTING_FOLDER = r"C:\Users\Hp\Desktop\AI pt3"
NUM_TO_PROCESS =50
TRAIN_RATIO = 0.5
TEST_RATIO = 0.3
VAL_RATIO = 0.2

# Folders containing images and labels
IMAGES_FOLDER = os.path.join(STARTING_FOLDER, "images")
LABELS_FOLDER = os.path.join(STARTING_FOLDER, "labels")

def find_matching_pairs(images_folder, labels_folder):
    """Find pairs of .txt and .jpg/.png files with the same name."""
    image_files = [f for f in os.listdir(images_folder) if f.endswith(('.jpg', '.png'))]
    txt_files = [f for f in os.listdir(labels_folder) if f.endswith('.txt')]
    
    matching_pairs = []
    for img in image_files:
        base_name = os.path.splitext(img)[0]
        if f"{base_name}.txt" in txt_files:
            matching_pairs.append((
                os.path.join(labels_folder, f"{base_name}.txt"),
                os.path.join(images_folder, img)
            ))
    return matching_pairs

def prepare_yolo_data(matching_pairs, num_to_process, output_folder, train_ratio=0.5, test_ratio=0.3, val_ratio=0.2):
    """Prepares YOLO dataset folders and copies selected image-label pairs."""
    selected_pairs = random.sample(matching_pairs, num_to_process)
    
    train_count = int(num_to_process * train_ratio)
    test_count = int(num_to_process * test_ratio)
    val_count = num_to_process - train_count - test_count

    folders = {
        "train": {"images": os.path.join(output_folder, "images/train"), "labels": os.path.join(output_folder, "labels/train")},
        "test": {"images": os.path.join(output_folder, "images/test"), "labels": os.path.join(output_folder, "labels/test")},
        "val": {"images": os.path.join(output_folder, "images/val"), "labels": os.path.join(output_folder, "labels/val")},
    }

    # Create folders
    for ftype in folders.values():
        os.makedirs(ftype["images"], exist_ok=True)
        os.makedirs(ftype["labels"], exist_ok=True)

    for i, (label_path, image_path) in enumerate(selected_pairs):
        if i < train_count:
            output_image_folder = folders["train"]["images"]
            output_label_folder = folders["train"]["labels"]
        elif i < train_count + test_count:
            output_image_folder = folders["test"]["images"]
            output_label_folder = folders["test"]["labels"]
        else:
            output_image_folder = folders["val"]["images"]
            output_label_folder = folders["val"]["labels"]

        shutil.copy(image_path, os.path.join(output_image_folder, os.path.basename(image_path)))
        shutil.copy(label_path, os.path.join(output_label_folder, os.path.basename(label_path)))

        print(f"{i+1}/{num_to_process} Copied {os.path.basename(image_path)} and {os.path.basename(label_path)}")

def main():
    matching_pairs = find_matching_pairs(IMAGES_FOLDER, LABELS_FOLDER)
    print(f"Found {len(matching_pairs)} matching pairs.")

    num_pairs_to_process = int(input("How many pairs do you want to process? "))
    num_pairs_to_process = max(1, min(num_pairs_to_process, len(matching_pairs)))

    current_date = datetime.date.today().strftime('%Y-%m-%d')
    output_folder = os.path.join(STARTING_FOLDER, f"pt2_dataset_{num_pairs_to_process}_{current_date}")

    prepare_yolo_data(
        matching_pairs=matching_pairs,
        num_to_process=num_pairs_to_process,
        output_folder=output_folder,
        train_ratio=TRAIN_RATIO,
        test_ratio=TEST_RATIO,
        val_ratio=VAL_RATIO,
    )

    print("\n✅ Finished.")

if __name__ == "__main__":
    main()
