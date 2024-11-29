# **DiskANN-Py**

DiskANN-Py is a simplified Python implementation of DiskANN, designed to handle large-scale Approximate Nearest Neighbor (ANN) search efficiently using graph-based algorithms and SSD storage.

---

## **Project Structure**

```plaintext
diskann_py/
│
├── main.py                 # Entry point to run the entire DiskANN system
├── graph_construction/
│   ├── __init__.py         # Makes this directory a Python package
│   ├── graph.py            # Core graph data structure and helper functions
│   ├── greedy_search.py    # Implementation of the GreedySearch algorithm
│   ├── robust_prune.py     # Implementation of the RobustPrune algorithm
│   ├── vamana.py           # Implementation of the Vamana graph construction algorithm
│
├── disk_index/
│   ├── __init__.py         # Makes this directory a Python package
│   ├── diskann_index.py    # DiskANN index construction (partitioning, merging)
│   ├── beam_search.py      # BeamSearch implementation for querying SSD-based index
│   ├── pq_compression.py   # Product Quantization (PQ) compression and storage utilities
│
├── utils/
│   ├── __init__.py         # Makes this directory a Python package
│   ├── metrics.py          # Distance metrics (e.g., Euclidean distance)
│   ├── dataset.py          # Dataset loading and preprocessing utilities
│   ├── clustering.py       # K-means clustering for overlapping partitions
│   ├── caching.py          # Caching frequently visited nodes
│
├── tests/
│   ├── test_graph.py       # Unit tests for graph construction algorithms
│   ├── test_disk_index.py  # Unit tests for DiskANN index construction
│   ├── test_search.py      # Unit tests for BeamSearch and query performance
│
├── README.md               # Project overview and instructions
└── requirements.txt        # Required Python libraries
```

---

## **Installation**

1. **Python Version**: Ensure you have Python 3.7 or newer installed.
2. **Install Dependencies**: Run the following command in the root of the project to install all required libraries:

   ```bash
   pip install -r requirements.txt
   ```

---

## **Download and Prepare Dataset**

To process datasets (e.g., `siftsmall`, `sift1m`) for use with DiskANN-Py:

1. **Download Dataset**

   ### **Linux/MacOS**
   ```bash
   mkdir -p ./data
   cd ./data
   wget ftp://ftp.irisa.fr/local/texmex/corpus/siftsmall.tar.gz
   tar -zxvf siftsmall.tar.gz
   ```

   ### **Windows**
   ```powershell
   mkdir ./data
   cd ./data
   Invoke-WebRequest -Uri "ftp://ftp.irisa.fr/local/texmex/corpus/siftsmall.tar.gz" -OutFile "siftsmall.tar.gz"
   tar -zxvf siftsmall.tar.gz
   ```

2. **Process Dataset**

   Use the `dataset.py` utility to preprocess the dataset:

   ```bash
   python utils/dataset.py --main_dir ./data --dataset_name siftsmall
   ```

   This will create a `processed/` subdirectory inside the dataset folder:

   ```plaintext
   ./data/siftsmall/processed/
       base.npy
       query.npy
       learn.npy
       groundtruth.npy
   ```

---

## **Usage**

1. **Running the System**

   The entry point for running the full DiskANN pipeline is `main.py`. Customize it based on your dataset and configuration.

   Example:
   ```bash
   python main.py --dataset ./data/siftsmall/processed/ --index_file ./index/siftsmall_index
   ```

2. **Graph Construction**

   Customize algorithms like `Vamana` and `RobustPrune` in the `graph_construction/` package to build the graph index.

3. **Search on Disk**

   Use the `disk_index/` package to build and query the DiskANN index with algorithms like **BeamSearch** and **PQ Compression**.

4. **Utilities**

   The `utils/` package includes helper tools for dataset preprocessing, metrics computation, clustering, and caching.

---

## **Testing**

To run unit tests for various components of the project:

```bash
pytest tests/
```

This will execute tests for:

- Graph construction algorithms
- Disk-based index construction
- Querying and search algorithms

---

## **Summary of Commands**

### **Install Requirements**

```bash
pip install -r requirements.txt
```

### **Download and Extract Dataset**

#### Linux/MacOS:
```bash
mkdir -p ./data
cd ./data
wget ftp://ftp.irisa.fr/local/texmex/corpus/siftsmall.tar.gz
tar -zxvf siftsmall.tar.gz
```

#### Windows:
```powershell
mkdir ./data
cd ./data
Invoke-WebRequest -Uri "ftp://ftp.irisa.fr/local/texmex/corpus/siftsmall.tar.gz" -OutFile "siftsmall.tar.gz"
tar -zxvf siftsmall.tar.gz
```

### **Process Dataset**

```bash
python utils/dataset.py --main_dir ./data --dataset_name siftsmall
```

### **Run DiskANN System**

```bash
python main.py --dataset ./data/siftsmall/processed/ --index_file ./index/siftsmall_index
```

---

## **Key Features**

1. **Graph Construction**:
   - Implements state-of-the-art algorithms like **Vamana** and **RobustPrune** for ANN graph construction.
2. **Disk-Based Indexing**:
   - Efficiently builds and queries ANN indices stored on SSDs.
   - Includes **BeamSearch** and **Product Quantization (PQ)** for memory-efficient queries.
3. **Utilities**:
   - Dataset preprocessing, clustering, caching, and distance metrics.

---

## **Contributing**

Contributions are welcome! If you'd like to add features or fix issues, please fork the repository, make changes, and submit a pull request.

---

## **License**

This project is licensed under the Apache License. See the `LICENSE` file for details.