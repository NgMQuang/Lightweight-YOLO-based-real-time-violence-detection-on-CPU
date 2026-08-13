import os
import glob
from pathlib import Path
from collections import defaultdict

def check_and_fix_dataset():
    """
    Deep check of the dataset:
    1. Find unlabeled images (images without corresponding labels)
    2. Find orphaned labels (labels without corresponding images)
    3. Create empty labels for unlabeled images
    """
    
    base_path = Path(r"c:\Users\Dell\Desktop\Violence_attention_system\FightAttention")
    
    # Check train dataset
    train_images_dir = base_path / "images" / "train"
    train_labels_dir = base_path / "labels" / "train"
    
    # Check val dataset
    val_images_dir = base_path / "images" / "val"
    val_labels_dir = base_path / "labels" / "val"
    
    results = {
        'train': {'unlabeled': [], 'orphaned_labels': [], 'created_labels': 0},
        'val': {'unlabeled': [], 'orphaned_labels': [], 'created_labels': 0}
    }
    
    # Process train dataset
    print("=" * 80)
    print("CHECKING TRAIN DATASET")
    print("=" * 80)
    process_dataset(train_images_dir, train_labels_dir, 'train', results)
    
    # Process val dataset
    print("\n" + "=" * 80)
    print("CHECKING VAL DATASET")
    print("=" * 80)
    process_dataset(val_images_dir, val_labels_dir, 'val', results)
    
    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY REPORT")
    print("=" * 80)
    
    for dataset_type in ['train', 'val']:
        print(f"\n{dataset_type.upper()} Dataset:")
        print(f"  Unlabeled images found: {len(results[dataset_type]['unlabeled'])}")
        if results[dataset_type]['unlabeled']:
            print(f"    - Created empty labels for: {results[dataset_type]['created_labels']} images")
        
        print(f"  Orphaned labels found: {len(results[dataset_type]['orphaned_labels'])}")
        if results[dataset_type]['orphaned_labels']:
            print(f"    - First few orphaned labels:")
            for label in results[dataset_type]['orphaned_labels'][:5]:
                print(f"      • {label}")
    
    print("\n" + "=" * 80)
    print("Dataset check complete!")
    print("=" * 80)

def process_dataset(images_dir, labels_dir, dataset_name, results):
    """Process a single dataset (train or val)"""
    
    images_dir = Path(images_dir)
    labels_dir = Path(labels_dir)
    
    # Ensure directories exist
    if not images_dir.exists():
        print(f"Error: {images_dir} does not exist")
        return
    
    if not labels_dir.exists():
        labels_dir.mkdir(parents=True, exist_ok=True)
        print(f"Created missing labels directory: {labels_dir}")
    
    # Get all image files
    image_extensions = ['*.png', '*.jpg', '*.jpeg', '*.bmp']
    all_images = []
    for ext in image_extensions:
        all_images.extend(images_dir.glob(ext))
    
    # Get all label files
    all_labels = set(f.stem for f in labels_dir.glob('*.txt'))
    
    print(f"\nFound {len(all_images)} images and {len(all_labels)} labels")
    
    # Find unlabeled images (images without corresponding labels)
    unlabeled_count = 0
    for img_path in all_images:
        img_stem = img_path.stem
        if img_stem not in all_labels:
            unlabeled_count += 1
            results[dataset_name]['unlabeled'].append(img_stem)
            
            # Create an empty label file for this image
            label_path = labels_dir / f"{img_stem}.txt"
            try:
                label_path.touch()  # Create empty file
                results[dataset_name]['created_labels'] += 1
            except Exception as e:
                print(f"  Error creating label for {img_stem}: {e}")
    
    if unlabeled_count > 0:
        print(f"✓ Found {unlabeled_count} unlabeled images")
        print(f"✓ Created {results[dataset_name]['created_labels']} empty label files")
        print(f"  Sample unlabeled images:")
        for img_name in results[dataset_name]['unlabeled'][:5]:
            print(f"    • {img_name}.png")
        if len(results[dataset_name]['unlabeled']) > 5:
            print(f"    ... and {len(results[dataset_name]['unlabeled']) - 5} more")
    else:
        print(f"✓ All images have corresponding labels")
    
    # Find orphaned labels (labels without corresponding images)
    orphaned_count = 0
    for label_stem in all_labels:
        # Check if image exists with any common extension
        found_image = False
        for ext in image_extensions:
            if (images_dir / f"{label_stem}{ext[1:]}").exists():
                found_image = True
                break
        
        if not found_image:
            orphaned_count += 1
            results[dataset_name]['orphaned_labels'].append(label_stem)
    
    if orphaned_count > 0:
        print(f"⚠ Found {orphaned_count} orphaned labels (no corresponding image)")
        print(f"  Sample orphaned labels:")
        for label_name in results[dataset_name]['orphaned_labels'][:5]:
            print(f"    • {label_name}.txt")
        if len(results[dataset_name]['orphaned_labels']) > 5:
            print(f"    ... and {len(results[dataset_name]['orphaned_labels']) - 5} more")
    else:
        print(f"✓ No orphaned labels found")
    
    # Print statistics
    print(f"\nDataset Statistics for {dataset_name.upper()}:")
    print(f"  Total images: {len(all_images)}")
    print(f"  Total labels: {len(all_labels)}")
    print(f"  Unlabeled images: {unlabeled_count}")
    print(f"  Orphaned labels: {orphaned_count}")
    print(f"  Matched pairs: {len(all_images) - unlabeled_count}")

if __name__ == "__main__":
    check_and_fix_dataset()
