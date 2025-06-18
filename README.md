# Gene Expression Data Analysis and Cancer Subtype Classification (GSE45827)

This repository contains a Jupyter Notebook (designed for Google Colab) that comprehensively analyzes a gene expression dataset (GSE45827 from GEO), focusing on breast cancer subtypes. The project involves data loading, preprocessing, dimensionality reduction using Principal Component Analysis (PCA), visualization, and training various machine learning models to classify cancer subtypes based on gene expression profiles.

## Project Overview

The goal of this project is to demonstrate a typical workflow for analyzing high-dimensional gene expression data. Key steps include:

1.  Data Acquisition: Downloading a gzipped GEO series matrix file directly from NCBI.
2.  Data Loading and Parsing: Developing a robust function to load the complex GEO matrix format, separating metadata from expression data.
3.  Initial Data Visualization: Exploring the distribution of raw expression values.
4.  Data Preprocessing: Transposing data, handling missing values (imputation), extracting target labels (cancer subtypes) from metadata, encoding labels, and standardizing features.
5.  Dimensionality Reduction (PCA): Applying PCA to reduce the number of features while retaining key variance, and visualizing the data in 2D and 3D PCA space (interactive visualization).
6.  Model Building and Evaluation: Training and evaluating several common classification models (Random Forest, Logistic Regression, SVM, KNN, Gradient Boosting, Decision Tree, MLP) on *both* the full gene expression data and the reduced PCA data.
7.  Visualization of Decision Boundaries/Regions: Visualizing the decision-making process of selected models (Decision Tree structure, KNN and Logistic Regression decision regions) in the 3D PCA space using interactive plots.
8.  Model Comparison: Comparing the performance (accuracy) of models trained on full data vs. PCA data.

## Dataset

The dataset used is GSE45827 from the Gene Expression Omnibus (GEO) database. This dataset contains gene expression profiles from human breast tumor samples, including various molecular subtypes (Luminal A, Luminal B, HER2, Triple Negative) and normal breast tissue samples.

*   Source: [https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE45827](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE45827)
*   File Used: `GSE45827_series_matrix.txt.gz`

## Repository Structure

*   `Breast_Cancer_Gene_Expression_Analysis.ipynb`: The main Jupyter Notebook containing all the code and analysis steps.

## How to Run the Notebook (using Google Colab)

1.  Open in Colab: The notebook is best run in Google Colab due to the use of Colab-specific shell commands (`!wget`, `%`) and the need for GPU acceleration for potentially larger models (though not strictly necessary for this dataset size with current models). Click the "Open in Colab" badge or upload the `.ipynb` file to your Google Drive and open it with Colaboratory.
2.  Run Cells Sequentially: Execute the notebook cells one by one from top to bottom.
    *   The first cell handles library imports and data download.
    *   Subsequent cells perform data loading, preprocessing, visualization, PCA, model training, and decision boundary visualization.
    *   Ensure each cell completes successfully before moving to the next. Pay attention to output messages and potential warnings/errors.
3.  Interactive Plots: Cells using `plotly` will generate interactive plots directly in the output.
4.  Graphviz Tree Visualization: The Decision Tree visualization cell uses `graphviz`. This should render correctly as an interactive SVG in Colab output.

## Libraries Used
*   `pandas`
*   `numpy`
*   `matplotlib`
*   `seaborn`
*   `gzip`, `io` (for data handling)
*   `sklearn` (for PCA, scaling, encoding, model selection, classifiers, metrics)
*   `plotly`, `plotly.graph_objects`, `plotly.express` (for interactive 3D visualization)
*   `graphviz` (for interactive Decision Tree visualization)
*   `requests` (for reliable data download)
*   `IPython.display` (for displaying Graphviz output)

## Analysis Highlights

*   Successfully loaded and parsed the complex GEO matrix format.
*   Extracted and processed cancer subtype labels from metadata, handling multiple metadata fields and potential inconsistencies.
*   Used PCA to reduce dimensionality from nearly 30,000 genes to just 3 principal components, capturing the most significant variance.
*   Visualized the data distribution and the PCA space interactively, showing how different subtypes cluster.
*   Trained and compared several classification models, demonstrating performance differences between using full gene expression data and PCA-reduced data.
*   Visualized the learned decision regions/boundaries in 3D PCA space for KNN and Logistic Regression models using interactive Plotly plots.
*   Visualized the structure of a single Decision Tree from the Random Forest model trained on PCA features using Graphviz.

## Potential Extensions

*   Hyperparameter tuning for the machine learning models using techniques like cross-validation and Grid Search or Randomized Search.
*   Exploring different dimensionality reduction techniques (e.g., t-SNE, UMAP) for visualization and potentially modeling.
*   Performing feature selection to identify key genes instead of relying solely on PCA components or using all genes.
*   Evaluating other classification models suitable for high-dimensional data.
*   Investigating the biological significance of the top PCA components or important features identified.
*   Implementing more sophisticated methods for handling multi-class imbalance if observed.

