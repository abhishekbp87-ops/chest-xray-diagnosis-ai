"""
Advanced data preparation and organization script.
Handles various data formats and creates comprehensive datasets.
"""

import argparse
import csv
import json
import logging
import shutil
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pydicom
import seaborn as sns
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DatasetOrganizer:
    """Advanced dataset organization and preparation."""

    def __init__(self, data_dir: str, output_dir: str = "processed_data"):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Supported formats
        self.image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
        self.dicom_extensions = {".dcm", ".dicom"}

        self.stats = {
            "total_files": 0,
            "images_found": 0,
            "dicom_found": 0,
            "invalid_files": 0,
            "classes": {},
            "splits": {},
        }

    def organize_chest_xray_dataset(
        self,
        test_size: float = 0.2,
        val_size: float = 0.2,
        random_state: int = 42,
    ) -> str:
        """Organize chest X-ray dataset with proper splits."""
        logger.info("Starting dataset organization...")

        # Scan directory structure
        image_data = self._scan_directory_structure()

        if not image_data:
            raise ValueError("No valid images found in the dataset directory")

        # Create splits
        train_data, temp_data = train_test_split(
            image_data,
            test_size=(test_size + val_size),
            stratify=[item["label"] for item in image_data],
            random_state=random_state,
        )

        # Split temp into validation and test
        val_data, test_data = train_test_split(
            temp_data,
            test_size=test_size / (test_size + val_size),
            stratify=[item["label"] for item in temp_data],
            random_state=random_state,
        )

        # Create organized structure
        self._create_organized_structure(train_data, val_data, test_data)

        # Create CSV index
        csv_path = self._create_csv_index(train_data, val_data, test_data)

        # Generate statistics
        self._generate_statistics(train_data, val_data, test_data)

        logger.info(f"Dataset organization complete! CSV saved to: {csv_path}")
        return csv_path

    def _scan_directory_structure(self) -> List[Dict]:
        """Scan directory for images and extract metadata."""
        logger.info("Scanning directory structure...")

        image_data = []

        for file_path in tqdm(self.data_dir.rglob("*")):
            if file_path.is_file():
                self.stats["total_files"] += 1

                # Check file extension
                if file_path.suffix.lower() in self.image_extensions:
                    try:
                        # Validate image
                        with Image.open(file_path) as img:
                            width, height = img.size
                            mode = img.mode

                        # Extract label from directory structure
                        label = self._extract_label_from_path(file_path)

                        if label:
                            image_data.append(
                                {
                                    "path": str(file_path.relative_to(self.data_dir)),
                                    "absolute_path": str(file_path),
                                    "filename": file_path.name,
                                    "label": label,
                                    "width": width,
                                    "height": height,
                                    "mode": mode,
                                    "size_bytes": file_path.stat().st_size,
                                    "format": file_path.suffix.lower()[1:],
                                }
                            )

                            self.stats["images_found"] += 1
                            self.stats["classes"][label] = (
                                self.stats["classes"].get(label, 0) + 1
                            )

                    except Exception as e:
                        logger.warning(f"Invalid image file: {file_path} - {e}")
                        self.stats["invalid_files"] += 1

                elif file_path.suffix.lower() in self.dicom_extensions:
                    try:
                        # Process DICOM file
                        dicom_data = self._process_dicom_file(file_path)
                        if dicom_data:
                            image_data.append(dicom_data)
                            self.stats["dicom_found"] += 1

                    except Exception as e:
                        logger.warning(f"Invalid DICOM file: {file_path} - {e}")
                        self.stats["invalid_files"] += 1

        logger.info(f"Found {len(image_data)} valid images")
        return image_data

    def _extract_label_from_path(self, file_path: Path) -> Optional[str]:
        """Extract label from file path structure."""
        # Common patterns for chest X-ray datasets
        path_parts = file_path.parts

        # Look for common class names in path
        common_classes = {
            "normal": "Normal",
            "pneumonia": "Pneumonia",
            "covid": "COVID-19",
            "tuberculosis": "Tuberculosis",
            "tb": "Tuberculosis",
            "bacterial": "Bacterial Pneumonia",
            "viral": "Viral Pneumonia",
        }

        for part in path_parts:
            part_lower = part.lower()
            for key, label in common_classes.items():
                if key in part_lower:
                    return label

        # Check parent directory
        if file_path.parent.name.lower() in common_classes:
            return common_classes[file_path.parent.name.lower()]

        # Check filename
        filename_lower = file_path.stem.lower()
        for key, label in common_classes.items():
            if key in filename_lower:
                return label

        return None

    def _process_dicom_file(self, file_path: Path) -> Optional[Dict]:
        """Process DICOM file and extract metadata."""
        try:
            dicom_data = pydicom.dcmread(file_path)

            # Convert DICOM to PIL Image
            pixel_array = dicom_data.pixel_array
            image = Image.fromarray(pixel_array)

            # Save as PNG
            png_path = self.output_dir / "converted_dicom" / f"{file_path.stem}.png"
            png_path.parent.mkdir(parents=True, exist_ok=True)
            image.save(png_path)

            # Extract label
            label = self._extract_label_from_path(file_path)

            if label:
                return {
                    "path": str(png_path.relative_to(self.data_dir)),
                    "absolute_path": str(png_path),
                    "filename": png_path.name,
                    "label": label,
                    "width": image.size[0],
                    "height": image.size[1],
                    "mode": image.mode,
                    "size_bytes": png_path.stat().st_size,
                    "format": "png",
                    "original_dicom": str(file_path),
                    "patient_id": getattr(dicom_data, "PatientID", "Unknown"),
                    "study_date": getattr(dicom_data, "StudyDate", "Unknown"),
                    "modality": getattr(dicom_data, "Modality", "Unknown"),
                }

        except Exception as e:
            logger.error(f"Error processing DICOM file {file_path}: {e}")

        return None

    def _create_organized_structure(
        self, train_data: List, val_data: List, test_data: List
    ):
        """Create organized directory structure."""
        logger.info("Creating organized directory structure...")

        organized_dir = self.output_dir / "organized"

        for split_name, data in [
            ("train", train_data),
            ("val", val_data),
            ("test", test_data),
        ]:
            split_dir = organized_dir / split_name

            for item in tqdm(data, desc=f"Organizing {split_name}"):
                class_dir = split_dir / item["label"]
                class_dir.mkdir(parents=True, exist_ok=True)

                # Copy file to organized structure
                src_path = Path(item["absolute_path"])
                dst_path = class_dir / item["filename"]

                if not dst_path.exists():
                    shutil.copy2(src_path, dst_path)

                # Update path in data
                item["organized_path"] = str(dst_path.relative_to(organized_dir))

    def _create_csv_index(self, train_data: List, val_data: List, test_data: List) -> str:
        """Create CSV index file."""
        logger.info("Creating CSV index...")

        csv_path = self.output_dir / "chest_xray_dataset.csv"

        with open(csv_path, "w", newline="", encoding="utf-8") as csvfile:
            fieldnames = [
                "path",
                "filename",
                "label",
                "split",
                "width",
                "height",
                "mode",
                "size_bytes",
                "format",
                "organized_path",
            ]

            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            # Write train data
            for item in train_data:
                writer.writerow(
                    {
                        "path": item["path"],
                        "filename": item["filename"],
                        "label": item["label"],
                        "split": "train",
                        "width": item["width"],
                        "height": item["height"],
                        "mode": item["mode"],
                        "size_bytes": item["size_bytes"],
                        "format": item["format"],
                        "organized_path": item.get("organized_path", ""),
                    }
                )

            # Write validation data
            for item in val_data:
                writer.writerow(
                    {
                        "path": item["path"],
                        "filename": item["filename"],
                        "label": item["label"],
                        "split": "val",
                        "width": item["width"],
                        "height": item["height"],
                        "mode": item["mode"],
                        "size_bytes": item["size_bytes"],
                        "format": item["format"],
                        "organized_path": item.get("organized_path", ""),
                    }
                )

            # Write test data
            for item in test_data:
                writer.writerow(
                    {
                        "path": item["path"],
                        "filename": item["filename"],
                        "label": item["label"],
                        "split": "test",
                        "width": item["width"],
                        "height": item["height"],
                        "mode": item["mode"],
                        "size_bytes": item["size_bytes"],
                        "format": item["format"],
                        "organized_path": item.get("organized_path", ""),
                    }
                )

        return str(csv_path)

    def _generate_statistics(self, train_data: List, val_data: List, test_data: List):
        """Generate and save dataset statistics."""
        logger.info("Generating dataset statistics...")

        # Update statistics
        self.stats["splits"] = {
            "train": len(train_data),
            "val": len(val_data),
            "test": len(test_data),
        }

        # Create detailed statistics
        all_data = train_data + val_data + test_data

        # Class distribution
        class_dist = {}
        for item in all_data:
            label = item["label"]
            class_dist[label] = class_dist.get(label, 0) + 1

        # Image statistics
        widths = [item["width"] for item in all_data]
        heights = [item["height"] for item in all_data]
        sizes = [item["size_bytes"] for item in all_data]

        stats_dict = {
            "total_images": len(all_data),
            "num_classes": len(class_dist),
            "class_distribution": class_dist,
            "split_distribution": self.stats["splits"],
            "image_dimensions": {
                "width_stats": {
                    "min": min(widths),
                    "max": max(widths),
                    "mean": np.mean(widths),
                    "std": np.std(widths),
                },
                "height_stats": {
                    "min": min(heights),
                    "max": max(heights),
                    "mean": np.mean(heights),
                    "std": np.std(heights),
                },
            },
            "file_size_stats": {
                "min_bytes": min(sizes),
                "max_bytes": max(sizes),
                "mean_bytes": np.mean(sizes),
                "total_size_mb": sum(sizes) / (1024 * 1024),
            },
        }

        # Save statistics
        stats_path = self.output_dir / "dataset_statistics.json"
        with open(stats_path, "w") as f:
            json.dump(stats_dict, f, indent=2, default=str)

        # Create visualizations
        self._create_visualizations(stats_dict, all_data)

        # Print summary
        self._print_summary(stats_dict)

    def _create_visualizations(self, stats_dict: Dict, all_data: List):
        """Create visualization plots."""
        logger.info("Creating visualizations...")

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # Class distribution
        classes = list(stats_dict["class_distribution"].keys())
        counts = list(stats_dict["class_distribution"].values())

        axes[0, 0].bar(classes, counts, color="skyblue")
        axes[0, 0].set_title("Class Distribution")
        axes[0, 0].set_xlabel("Class")
        axes[0, 0].set_ylabel("Count")
        plt.setp(axes[0, 0].xaxis.get_majorticklabels(), rotation=45, ha="right")

        # Split distribution
        splits = list(stats_dict["split_distribution"].keys())
        split_counts = list(stats_dict["split_distribution"].values())

        axes[0, 1].pie(split_counts, labels=splits, autopct="%1.1f%%", startangle=90)
        axes[0, 1].set_title("Split Distribution")

        # Image dimensions
        widths = [item["width"] for item in all_data]
        heights = [item["height"] for item in all_data]

        axes[1, 0].scatter(widths, heights, alpha=0.5, s=10)
        axes[1, 0].set_title("Image Dimensions Distribution")
        axes[1, 0].set_xlabel("Width")
        axes[1, 0].set_ylabel("Height")

        # File size distribution
        sizes_mb = [item["size_bytes"] / (1024 * 1024) for item in all_data]

        axes[1, 1].hist(sizes_mb, bins=50, alpha=0.7, color="lightgreen")
        axes[1, 1].set_title("File Size Distribution")
        axes[1, 1].set_xlabel("File Size (MB)")
        axes[1, 1].set_ylabel("Frequency")

        plt.tight_layout()
        plt.savefig(self.output_dir / "dataset_statistics.png", dpi=300, bbox_inches="tight")
        plt.close()

        logger.info("Visualizations saved to dataset_statistics.png")

    def _print_summary(self, stats_dict: Dict):
        """Print dataset summary."""
        print("\n" + "=" * 50)
        print("DATASET SUMMARY")
        print("=" * 50)
        print(f"Total Images: {stats_dict['total_images']:,}")
        print(f"Number of Classes: {stats_dict['num_classes']}")
        print(f"Total Size: {stats_dict['file_size_stats']['total_size_mb']:.2f} MB")

        print("\nClass Distribution:")
        for class_name, count in stats_dict["class_distribution"].items():
            percentage = (count / stats_dict["total_images"]) * 100
            print(f"  {class_name}: {count:,} ({percentage:.1f}%)")

        print("\nSplit Distribution:")
        for split_name, count in stats_dict["split_distribution"].items():
            percentage = (count / stats_dict["total_images"]) * 100
            print(f"  {split_name}: {count:,} ({percentage:.1f}%)")

        print("\nImage Dimensions:")
        width_stats = stats_dict["image_dimensions"]["width_stats"]
        height_stats = stats_dict["image_dimensions"]["height_stats"]
        print(
            f"  Width: {width_stats['min']} - {width_stats['max']} "
            f"(avg: {width_stats['mean']:.0f})"
        )
        print(
            f"  Height: {height_stats['min']} - {height_stats['max']} "
            f"(avg: {height_stats['mean']:.0f})"
        )


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Organize and prepare chest X-ray dataset")

    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Directory containing the raw dataset",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="processed_data",
        help="Output directory for processed dataset",
    )
    parser.add_argument(
        "--test_size",
        type=float,
        default=0.2,
        help="Proportion of dataset for testing",
    )
    parser.add_argument(
        "--val_size",
        type=float,
        default=0.2,
        help="Proportion of dataset for validation",
    )
    parser.add_argument(
        "--random_state",
        type=int,
        default=42,
        help="Random seed for reproducible splits",
    )

    return parser.parse_args()


def main():
    """Main function."""
    args = parse_args()

    # Validate arguments
    if not Path(args.data_dir).exists():
        raise ValueError(f"Data directory does not exist: {args.data_dir}")

    if args.test_size + args.val_size >= 1.0:
        raise ValueError("test_size + val_size must be less than 1.0")

    # Create organizer
    organizer = DatasetOrganizer(args.data_dir, args.output_dir)

    # Organize dataset
    try:
        csv_path = organizer.organize_chest_xray_dataset(
            test_size=args.test_size,
            val_size=args.val_size,
            random_state=args.random_state,
        )

        logger.info("Dataset organization completed successfully!")
        logger.info(f"Dataset CSV: {csv_path}")
        logger.info(f"Statistics saved to: {args.output_dir}")

    except Exception as e:
        logger.error(f"Dataset organization failed: {e}")
        raise


if __name__ == "__main__":
    main()
