# ps-alb-semi-automated-analysis
Semi-automated image analysis pipeline for glomerular Ps'alb quantification from time-lapse fluorescence microscopy data.

# Semi-automated Ps′alb Analysis

This repository contains the code developed for my MSc Bioinformatics research project for the semi-automated analysis of glomerular capillary permeability (Ps′alb) from time-lapse fluorescence microscopy recordings.

The workflow was developed to reduce repetitive manual image analysis while retaining user input for biologically important decisions, including capillary-loop selection and measurement ROI definition.

## Workflow

The analysis pipeline consists of the following main steps:

1. Load multi-channel ND2 time-lapse microscopy data.
2. Estimate frame-to-frame image displacement using the R18 structural channel.
3. Apply translation-based image registration to the corresponding image channels.
4. Manually initialise a capillary-loop ROI.
5. Track the selected capillary loop across the registered image sequence using CSRT tracking.
6. Extract AF488 fluorescence intensity from the tracked ROI.
7. Identify the post-wash fluorescence decay phase and fit the decay curve.
8. Calculate Ps′alb using the fitted decay constant and estimated capillary radius.

## Repository structure

### `main.ipynb`

Main analysis notebook containing the complete workflow, including data loading, registration, ROI tracking, fluorescence decay analysis and Ps′alb calculation.

### `registration_utils.py`

Functions for estimating translational image drift and applying the calculated transformations to the microscopy image stacks.

### `capillary_roi.py`

Functions for interactive capillary-loop ROI selection, ROI radius definition, CSRT-based tracking and AF488 fluorescence intensity extraction.

## Input data

The workflow was developed for multi-channel ND2 time-lapse fluorescence microscopy recordings containing:

- AF488/FITC fluorescence
- R18/Rhodamine structural fluorescence
- Transmitted-light images

Raw microscopy data are not included in this repository.

## Ps′alb calculation

For an accepted capillary-loop ROI, the post-wash AF488 fluorescence decay is fitted to estimate the decay constant (`k`). Ps′alb is subsequently calculated using:

**Ps′alb = −kR / 2**

where `R` is the estimated capillary radius.

## Requirements

The workflow was developed in Python and uses the following main packages:

- NumPy
- pandas
- Matplotlib
- OpenCV
- SimpleITK
- nd2

## Notes

The current workflow is semi-automated. User input is retained for capillary-loop selection, ROI size definition and identification of the appropriate post-wash analysis window. Automated image registration, ROI tracking and fluorescence intensity extraction are then used to reduce repetitive frame-by-frame manual analysis.

This code was developed as part of an MSc Bioinformatics research project at the University of Bristol.

## Streamlit prototype

A prototype Streamlit interface is included in `app.py` to demonstrate the
semi-automated Ps′alb analysis workflow through an interactive interface.

The application includes ND2 file loading, image registration, interactive
capillary-loop ROI selection, ROI tracking and quality control, fluorescence
intensity extraction, decay fitting, capillary diameter measurement, and
Ps′alb calculation.

To run the application locally:

```bash
streamlit run app.py
To run the interface locally:

python3 -m streamlit run app.py
