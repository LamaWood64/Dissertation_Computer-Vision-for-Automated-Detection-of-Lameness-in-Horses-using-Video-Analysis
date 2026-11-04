# Computer Vision for Automated Detection of Lameness in Horses using Video Analysis

A computer vision and machine learning pipeline for the preliminary detection of lameness in horses using ordinary video recordings.

This project was developed as part of my Bachelor's dissertation in Computer Science at BSBI, in partnership with the University for the Creative Arts (UCA).

The system combines markerless animal pose estimation, biomechanical feature extraction, and machine learning classification to investigate whether gait abnormalities can be identified automatically from video data without specialized sensors or motion-capture equipment.

<p align="center">
  <img src="assets/images/horse-pose-estimation.png" width="700" alt="Horse pose estimation using YOLO11-Pose">
</p>

## Table of Contents

- [Project Overview](#project-overview)
- [Research Objectives](#research-objectives)
- [System Architecture](#system-architecture)
- [Dataset](#dataset)
- [Technologies](#technologies)
- [Methodology](#methodology)
- [Experimental Results](#experimental-results)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Limitations](#limitations)
- [Future Work](#future-work)
- [References](#references)
- [Author](#author)

## Project Overview

Lameness is a common locomotor disorder in horses that can negatively affect their health, welfare, and athletic performance.

Traditional lameness assessment relies heavily on veterinary visual examination. Although this approach is widely used, its reliability can depend on the examiner's experience, observation conditions, and the severity of the gait abnormality.

This research investigates the potential of computer vision to support the preliminary detection of lameness through automated analysis of horse movement.

The proposed approach uses YOLO11-Pose to estimate anatomical landmarks from video recordings. The resulting keypoint trajectories are processed to extract biomechanical features, which are subsequently used by machine learning classifiers to distinguish between healthy and lame horses.

The project is designed as a proof of concept rather than a clinically validated diagnostic system.

## Research Objectives

The main objectives of the research were:

- Develop a markerless computer vision pipeline for analyzing horse locomotion.
- Automatically detect anatomical landmarks using a deep learning pose estimation model.
- Extract spatial and temporal features describing gait characteristics.
- Investigate the ability of machine learning algorithms to distinguish healthy and lame horses.
- Evaluate the performance and limitations of the proposed approach using video-level classification.

## System Architecture

The system consists of several interconnected stages, from video acquisition to the final classification of gait condition.

<p align="center">
  <img src="assets/images/pipeline.png" width="850" alt="Research pipeline for automated horse lameness detection">
</p>

### Pipeline

1. **Video acquisition:** Collect publicly available recordings of horses moving at a trot.
2. **Data preparation:** Extract video frames and prepare them for annotation.
3. **Pose annotation:** Manually annotate anatomical landmarks using CVAT.
4. **Pose estimation:** Train YOLO11-Pose to detect the horse's anatomical keypoints.
5. **Feature extraction:** Normalize and smooth keypoint trajectories and calculate biomechanical features.
6. **Classification:** Train machine learning models to distinguish between healthy and lame horses.
7. **Evaluation:** Assess model performance using stratified cross-validation and standard classification metrics.

## Dataset

The experimental dataset consisted of 30 videos, each representing a different horse.

| Property | Description |
|---|---|
| Number of videos | 30 (1421 manually labeled frames) |
| Independent observations | 30 horses |
| Data source | Publicly available video recordings |
| Movement | Trot |
| Classification task | Healthy vs Lame |
| Pose estimation landmarks | 12 keypoints |
| Extracted features | 216 per video |

The videos were collected from publicly accessible sources, including YouTube and Reddit.

Each video was treated as an independent observation during classification. Individual video frames were not considered independent classification samples.

## Technologies

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| YOLO11-Pose | Animal pose estimation |
| CVAT | Manual annotation of anatomical landmarks |
| NumPy | Numerical computation |
| Pandas | Data processing and analysis |
| SciPy | Statistical analysis and signal processing |
| Scikit-learn | Machine learning and model evaluation |
| Matplotlib | Data visualization |

## Methodology

### 1. Pose Estimation

YOLO11-Pose was trained to identify 12 anatomical landmarks on the horse's body.

The model was used to extract keypoint coordinates from video recordings, providing the spatial information required for subsequent gait analysis.

### 2. Feature Engineering

The extracted coordinates were normalized and smoothed using the Savitzky–Golay filter.

Based on the resulting keypoint trajectories, statistical and dynamic characteristics of movement were calculated.

A total of 216 features were extracted for each video.

### 3. Statistical Analysis

The extracted features were analyzed using statistical methods, including:

- Cohen's d for effect size estimation.
- Welch's t-tests for statistical comparison.
- Feature importance analysis to identify characteristics contributing to classification.

### 4. Classification

Three machine learning algorithms were evaluated:

- Random Forest
- Support Vector Machine (SVM)
- Gradient Boosting

Random Forest was additionally evaluated using limb-specific feature selection, while SVM and Gradient Boosting were trained using the full extracted feature set.

### 5. Model Evaluation

The classifiers were evaluated using five-fold stratified cross-validation.

The following metrics were used:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC

## Experimental Results

### Pose Estimation Performance

The YOLO11-Pose model achieved the following results:

| Metric | Score |
|---|---:|
| Precision | 0.988 |
| Recall | 0.979 |
| mAP@0.5 (Pose) | 0.984 |

These results indicate that the model performed well at detecting the selected anatomical landmarks in the experimental dataset.

### Classification Performance

The highest average classification scores were obtained by Random Forest using 20 selected features.

| Metric | Random Forest |
|---|---:|
| Accuracy | 0.633 |
| Precision | 0.697 |
| Recall | 0.667 |
| F1-score | 0.658 |

SVM achieved the highest ROC-AUC among the three evaluated classifiers, with a score of 0.676.

<p align="center">
  <img src="assets/images/classification-results.png" width="750" alt="Classification results and model comparison">
</p>

### Feature Importance

The feature importance analysis indicated that characteristics related to knee and hoof movements contributed significantly to classification, particularly those associated with the hind limbs.

<p align="center">
  <img src="assets/images/feature-importance.png" width="750" alt="Feature importance analysis">
</p>

## Repository Structure

```text
horse-lameness-detection/
│
├── assets/
│   └── images/
│       ├── horse-pose-estimation.png
│       ├── pipeline.png
│       ├── classification-results.png
│       └── feature-importance.png
│
├── data/
│   └── README.md
│
├── notebooks/
│   └── README.md
│
├── src/
│   └── README.md
│
├── requirements.txt
├── .gitignore
└── README.md
```

The `assets/images/` directory contains selected figures from the dissertation.

The `data/` directory is intended for dataset documentation and instructions. The original video recordings are not included in this repository.

## Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/horse-lameness-detection.git
cd horse-lameness-detection
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it:

**macOS / Linux**

```bash
source .venv/bin/activate
```

**Windows**

```bash
.venv\Scripts\activate
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

The complete research workflow consists of video preprocessing, pose estimation, feature extraction, and classification.

The repository will contain the implementation of these stages as the research code is organized and published.

For the original experimental setup, the workflow is:

1. Prepare the video dataset and extract frames.
2. Annotate anatomical landmarks using CVAT.
3. Train the YOLO11-Pose model.
4. Extract keypoint trajectories from the videos.
5. Generate the biomechanical feature dataset.
6. Train and evaluate the classification models.

**Note:** The exact commands and configuration files will be documented here once the codebase is finalized.

## Limitations

The experimental results should be interpreted in the context of the study's limitations.

- **Small dataset:** The classification dataset contains only 30 independent observations.
- **Limited generalization:** The small sample size and variability between cross-validation folds limit the reliability of conclusions about performance on unseen horses.
- **Video variability:** Differences in camera position, lighting, movement speed, and recording conditions may affect the quality of pose estimation.
- **No external validation:** The approach requires further evaluation using a larger dataset containing independent horses and diverse recording conditions.
- **Preliminary classification:** The system is not intended to replace veterinary examination or provide a definitive diagnosis.

The results demonstrate the potential of combining markerless pose estimation and biomechanical feature analysis for preliminary gait assessment, but further research is necessary before practical veterinary deployment.

## Future Work

Potential directions for further development include:

- Expanding the dataset with additional horses and more diverse recording conditions.
- Performing external validation on independent video recordings.
- Investigating additional biomechanical features and feature selection methods.
- Improving robustness to occlusion, camera movement, and changes in lighting.
- Exploring alternative pose estimation and classification architectures.
- Developing a more complete and reproducible software pipeline.

## References

This project builds on research in computer vision, animal pose estimation, and equine gait analysis.

Selected references:

- Mathis, A. et al. (2018). *DeepLabCut: markerless pose estimation of user-defined body parts with deep learning.*
- Nath, T. et al. (2019). *Using DeepLabCut for 3D markerless pose estimation across species and behaviors.*
- Redmon, J. et al. (2016). *You Only Look Once: Unified, Real-Time Object Detection.*
- LeCun, Y., Bengio, Y. and Hinton, G. (2015). *Deep learning.*
- Serra Bragança, F. et al. (2018). Research on objective equine gait analysis.

For the complete bibliography, see the dissertation.

## Author

**Anastasia Shadakh**

BSc (Hons) Computer Science and Digitisation  
Berlin School of Business and Innovation (BSBI)  
University for the Creative Arts (UCA)  
2026

[GitHub Profile](https://github.com/LamaWood64)

---

*This repository contains research developed as part of a Bachelor's dissertation. The project is intended for educational and research purposes.*
