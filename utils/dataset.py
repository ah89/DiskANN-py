import os
import numpy as np
import struct
import argparse


def read_fvecs(file_path):
    """
    Reads a .fvecs file and returns a numpy array of shape (n, d).
    """
    with open(file_path, "rb") as f:
        data = f.read()
        offset = 0
        vectors = []
        while offset < len(data):
            d = struct.unpack('i', data[offset:offset + 4])[0]
            offset += 4
            vector = struct.unpack('f' * d, data[offset:offset + d * 4])
            offset += d * 4
            vectors.append(vector)
        return np.array(vectors, dtype=np.float32)


def read_bvecs(file_path):
    """
    Reads a .bvecs file and returns a numpy array of shape (n, d).
    """
    with open(file_path, "rb") as f:
        data = f.read()
        offset = 0
        vectors = []
        while offset < len(data):
            d = struct.unpack('i', data[offset:offset + 4])[0]
            offset += 4
            vector = struct.unpack('B' * d, data[offset:offset + d])
            offset += d
            vectors.append(vector)
        return np.array(vectors, dtype=np.uint8)


def read_ivecs(file_path):
    """
    Reads a .ivecs file and returns a numpy array of shape (n, d).
    """
    with open(file_path, "rb") as f:
        data = f.read()
        offset = 0
        vectors = []
        while offset < len(data):
            d = struct.unpack('i', data[offset:offset + 4])[0]
            offset += 4
            vector = struct.unpack('i' * d, data[offset:offset + d * 4])
            offset += d * 4
            vectors.append(vector)
        return np.array(vectors, dtype=np.int32)


def save_as_npy(array, output_path):
    """
    Saves a numpy array to a .npy file.
    """
    np.save(output_path, array)
    print(f"[INFO] Saved {output_path}")


def process_dataset(main_dir, dataset_name):
    """
    Processes the dataset with files prefixed by its name and transforms them into .npy format.
    
    :param main_dir: Path to the main dataset directory.
    :param dataset_name: Name of the dataset (e.g., "siftsmall").
    """
    dataset_dir = os.path.join(main_dir, dataset_name)
    if not os.path.exists(dataset_dir):
        raise FileNotFoundError(f"[ERROR] Dataset directory not found: {dataset_dir}")

    print(f"[INFO] Processing dataset: {dataset_name} in {dataset_dir}")

    # Define file paths based on the dataset name
    base_file = os.path.join(dataset_dir, f"{dataset_name}_base.fvecs")
    query_file = os.path.join(dataset_dir, f"{dataset_name}_query.fvecs")
    learn_file = os.path.join(dataset_dir, f"{dataset_name}_learn.fvecs")
    gt_file = os.path.join(dataset_dir, f"{dataset_name}_groundtruth.ivecs")

    # Create output directory
    output_dir = os.path.join(dataset_dir, "processed")
    os.makedirs(output_dir, exist_ok=True)

    # Process base vectors
    if os.path.exists(base_file):
        print(f"[INFO] Reading base vectors from {base_file}...")
        base_vectors = read_fvecs(base_file)
        save_as_npy(base_vectors, os.path.join(output_dir, "base.npy"))

    # Process query vectors
    if os.path.exists(query_file):
        print(f"[INFO] Reading query vectors from {query_file}...")
        query_vectors = read_fvecs(query_file)
        save_as_npy(query_vectors, os.path.join(output_dir, "query.npy"))

    # Process learn vectors
    if os.path.exists(learn_file):
        print(f"[INFO] Reading learn vectors from {learn_file}...")
        learn_vectors = read_fvecs(learn_file)
        save_as_npy(learn_vectors, os.path.join(output_dir, "learn.npy"))

    # Process groundtruth
    if os.path.exists(gt_file):
        print(f"[INFO] Reading groundtruth from {gt_file}...")
        groundtruth = read_ivecs(gt_file)
        save_as_npy(groundtruth, os.path.join(output_dir, "groundtruth.npy"))

    print(f"[INFO] Dataset {dataset_name} processed and saved to {output_dir}")


if __name__ == "__main__":
    # Command-line interface
    parser = argparse.ArgumentParser(description="Process TEXMEX datasets for DiskANN.")
    parser.add_argument(
        "--main_dir",
        type=str,
        required=True,
        help="Path to the main directory containing datasets (e.g., ./data)."
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        required=True,
        help="Name of the dataset to process (e.g., siftsmall, sift1m)."
    )
    args = parser.parse_args()

    # Process the dataset
    process_dataset(args.main_dir, args.dataset_name)