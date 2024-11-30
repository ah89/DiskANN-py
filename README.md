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

Here’s a **"How to Run the Tests"** section for your GitHub `README.md`. It explains how to set up the environment and run the tests for your `diskann_py` project.

---

## **How to Run the Tests**

To ensure that the components of the project (e.g., graph construction, search algorithms, and disk-based indexing) work correctly, unit tests are provided in the `tests/` directory. Follow these steps to run the tests:

### **1. Install Dependencies**

Before running the tests, make sure you have all the required Python libraries installed. Use the `requirements.txt` file to install them:

```bash
pip install -r requirements.txt
```

### **2. Run All Tests**

You can run all the tests in the `tests/` directory using **pytest**:

```bash
pytest tests/
```

This will execute all unit tests in the project and display the results.

### **3. Run Specific Test Files**

If you want to run tests for a specific module, you can specify the test file. For example:

- To test graph construction algorithms (e.g., Vamana):
  ```bash
  pytest tests/test_graph.py
  ```

- To test search algorithms (e.g., BeamSearch):
  ```bash
  pytest tests/test_search.py
  ```

- To test DiskANN index construction:
  ```bash
  pytest tests/test_disk_index.py
  ```

### **4. View Detailed Test Output**

To see detailed output for each test (e.g., print statements or assertions), use the `-v` flag:

```bash
pytest -v tests/
```

### **5. Debugging with a Single Test Function**

If you need to run a specific test function for debugging, use the `-k` flag with the test's name. For example:

```bash
pytest -k "test_vamana_graph_construction" -v
```

---

### **Testing Framework**

The tests are built using **pytest**, a simple and powerful testing framework for Python. If you don’t have it installed, you can install it with:

```bash
pip install pytest
```

---

Let me know if you need additional sections or further adjustments!

## **Contributing**

Contributions are welcome! If you'd like to add features or fix issues, please fork the repository, make changes, and submit a pull request.

---

## **License**

This project is licensed under the Apache License. See the `LICENSE` file for details.