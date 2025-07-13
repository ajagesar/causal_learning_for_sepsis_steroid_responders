# Causal deep learning to personalize medicine; which intensive care patients with sepsis will benefit from corticosteroid therapy?

This repository contains the code for the manuscript 'Causal deep learning to personalize medicine; which intensive care patients with sepsis will benefit from corticosteroid therapy?'

---

## Files Overview

The repository is structured to facilitate a complete pipeline, from data extraction and preprocessing to model training, evaluation, and interpretation.

* `run.py`: The main script to execute the entire pipeline.
* `PipeLine.py`: Orchestrates the various steps of the causal inference pipeline, including training the model.
* `EvaluationPipeline.py`: Manages the evaluation aspects of the trained models, excluding training the model. Specifically built for external validation.
* `utils.py`: A collection of general utility functions used across the project.

### Extraction and Preprocessing Modules:

* 1\. `_Extraction.py`: Handles data extraction from the source by executing SQL scripts that extracts data from Google Bigquery.
* 2\. `_PreProcessing.py`: Contains functions for data cleaning and initial preparation.
* 3a\. `_Split.py`: Manages the splitting of data into training and test sets.
* 3b\. `_FakeSplit.py`: Only for external dataset.It effectively renames the entire external data set to the test set, so the rest of the pipeline can be run.
* 4a\. `_ScaleImpute.py`: Manages data scaling and imputation of missing values.
* 4b\. `_ExternalScaleImpute.py`: Only for external dataset. Uses the imputer of the internal dataset (AmsterdamUMCdb) to impute the external dataset (MIMIC-IV).

### Feature Engineering & Selection:

* 5a\. `_FeatureSelection.py`: Implements recursive feature elimination for selecting relevant features for the model.
* 5b\. `_ExternalFeatureSelection.py`: Only for external dataset. Applies selected features from (5a) to the external dataset.

### Modeling & Evaluation:

* 6\. `_HyperparameterTuning.py`: Manages the optimization of model hyperparameters.
* 7\. `_Modeling.py`: Contains the code for building and training the causal inference models, including TARNet.
  * imafcrnet.py: Contains the code of the TARNet model to which (7) refers
* 8\. `_LoadModel.py`: Utility for loading pre-trained models.
* 9\. `_Evaluate.py`: Contains functions for evaluating model performance with metrics such as AUC and Brier score.

### Interpretation & Visualization:

* 10\. `_EffectEstimation.py`: Specifically focuses on the estimation of Individual Treatment Effects (ITEs).
* 11\. `_Interpretation.py`: Provides tools for interpreting model predictions by classifying patients in responders/non-responders/harmers.
* 12\. `_Plotting.py`: Contains scripts for generating visualizations of results, such as model performance and estimation of ITEs.

---

# Getting Started

Further instructions on how to set up the environment, prepare data, and run the pipeline will be provided here. Typically, this would involve:

1. Clone the repository
2. Configure google bigquery access to databases in the `_Extraction.py` file
2. Run the pipeline with `python run.py`

---

## Manuscript Details

This code accompanies a manuscript that delves into the methodology, results, and clinical implications of using TARNet to predict steroid benefit in sepsis patients. Please refer to the manuscript for a comprehensive understanding of the research. DOI: TBD
