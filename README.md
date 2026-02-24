# UAV

## Framework

This project uses a **python3.10** virtual environment with GPU - accelerated machine learning support.

### Core Machine-Learning

- **Tensorflow2.20.x (GPU Build)**:
    Used for the deep learning, neural networking modeling, and hardware-accelerated computations which leverage the performance boost of **CUDA** in Nvidia GPU support.

- **scikit-learn**:
    Classical machine learning algorithms such as (Clustering, regression, preprocessing, and metrics) whereas **preprocessing** and **metrics** the core feature is the key - feature used in this application.

- **numba**:
    Just-in-Time (**JIT**) compilation for accelerating numerical python functions, especially useful for 
    - Custom simulation loops
    - Signal processing routines
    - High-performance numerical kernels
    
    Notice that this is highly experimental for speed up angle computations which does indeed add additional compilation time so may only be useful if compute multiple times in order to account for additional compilation time (although not devastating time loss either way)

### Data-Processing & performance

 - **pandas**: 
    Structured data manipulation and experiment result handling

 - **pyarrow**: 
    Efficient data formats useful in larger scale logging or dataset storage 

- **orjson**:
    High-Performance _JSON Serialization_ which is significantly faster than standard _json_ module

- **tqdm**:
    Lightweight progress bars for simulations and training loops.

### Visualization

- **matplotlib**:
    Core plotting library for experimental visualization analysis which is fundamental for visualize the signal loss rates etc.

- **seaborn**:
    Statistical plotting built on top of matplotlib for simpler experiment visualization (may like numba be unnecessary)
