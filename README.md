# Introduction to Machine Learning — Coursework

Assignments from COMPSCI 711: Introduction to Machine Learning (graduate level, University of Wisconsin–Milwaukee, Spring 2024), by John Quinn.

This repository holds my own work from the course: four assignments, each with code, a written report, figures, and the data and trained models where relevant. The instructor's materials (slides, handouts, exams, syllabus) and the textbook are not included.

Each assignment picks a public dataset (from [OpenML](https://www.openml.org/) or Kaggle), trains and tunes models with scikit-learn or Keras, and compares them in a short written report.

**Contents**

- [Assignments](#assignments): [1](#assignment-1--decision-trees) · [2](#assignment-2--regression-model-comparison) · [3](#assignment-3--neural-networks) · [4](#assignment-4--convolutional-neural-networks)
- [Running the code](#running-the-code)
- [Course topics](#course-topics)
- [Repository layout](#repository-layout)
- [Credits](#credits)

## Assignments

| # | Topic | Models | Datasets | Tools |
|---|---|---|---|---|
| [1](#assignment-1--decision-trees) | Classification, ROC/AUC, hyperparameter tuning | Decision trees (entropy vs. Gini) | steel-plates-fault, OVA_Prostate | scikit-learn |
| [2](#assignment-2--regression-model-comparison) | Regression, preprocessing, learning curves | k-nearest neighbors, decision tree, linear regression | Moneyball | scikit-learn |
| [3](#assignment-3--neural-networks) | Neural networks for regression and classification | Dense feed-forward networks | wine_quality, spambase | Keras |
| [4](#assignment-4--convolutional-neural-networks) | Image classification, transfer learning, error analysis | Two custom CNNs, fine-tuned EfficientNetV2S | 10 Species of Monkey | Keras |

## Assignment 1 — Decision trees

[Code](Assignments/Assignment%201/JQ-CompSci711-Assign1.py) · [Report](Assignments/Assignment%201/CompSci%20711%20Assignment%201%20Report.pdf)

Compares the entropy and Gini split criteria on two binary classification datasets: steel plate faults (34 features) and prostate tumor gene expression (about 1,000 features).

- ROC curves and AUC from 10-fold cross-validated probability predictions.
- `min_samples_leaf` tuned over 1–5 with a grid search nested inside the cross-validation.

| Dataset | Entropy AUC | Gini AUC | Best `min_samples_leaf` (entropy / Gini) |
|---|---|---|---|
| OVA_Prostate | 0.955 | 0.976 | 5 / 3 |
| steel-plates-fault | 0.978 | 0.978 | 1 / 1 |

The split criterion only mattered on the high-dimensional gene expression data, which also preferred larger leaves.

![ROC curves for both datasets and both split criteria](Assignments/Assignment%201/figures/Figure_1.png)

Also in [the folder](Assignments/Assignment%201/): the report as `.docx` and a scratch script (`test.py`).

## Assignment 2 — Regression model comparison

[Code](Assignments/Assignment%202/JQ-CompSci711-Assign2.py) · [Report](Assignments/Assignment%202/CompSci%20711%20Assignment%202%20Report.pdf)

Predicts runs scored per team season on the Moneyball baseball dataset (1,232 examples).

- Missing values imputed and the nominal features (team, league) one-hot encoded.
- k-nearest neighbors compared at k = 3, 7 and 11 using learning curves.
- Tuned k-nearest neighbors, a tuned decision tree and linear regression compared by 10-fold cross-validated RMSE.

| k | 3 | 7 | 11 |
|---|---|---|---|
| RMSE | 6.73 | 7.24 | 8.18 |

![Learning curves for k-nearest neighbors at three values of k](Assignments/Assignment%202/figures/Figure_1_CompSci_A2.png)

**Known issue:** the script adds the target column to the feature table before training, so every model in this assignment can see the answer. That is why linear regression reaches an RMSE of effectively zero in the model comparison.

I did not identify this cause at the time. The report notes that the result looked like it could be an error, but goes on to attribute it to the data being strongly linear. I found the leak later, when reviewing the code for this repository. The numbers above are as submitted and should be read with that in mind.

Also in [the folder](Assignments/Assignment%202/):

- `figures/`: the remaining learning-curve plots, including the model comparison.
- `data/baseball.arff`: a local copy of the Moneyball data. The script itself downloads it from OpenML.
- `data/SeoulBikeData.csv` and its source zip: a second dataset (Seoul bike sharing demand) that I started with and dropped.
- The report as `.docx`.

## Assignment 3 — Neural networks

[Code](Assignments/Assignment%203/Assignement3_JohnQuinn_CompSci711.py) · [Report](Assignments/Assignment%203/CompSci%20711%20Assignment%203%20Report.pdf)

Trains four dense network architectures on each of two tasks for 200 epochs with the Adam optimizer, using an 80/20 train/validation split.

| Model | Hidden layers | Activation | Wine quality: best validation MSE | Spambase: best validation accuracy |
|---|---|---|---|---|
| 1 | 30, 10 | sigmoid | 0.507 | 0.939 |
| 2 | 30, 10 | ReLU | 0.522 | 0.936 |
| 3 | 15, 5 | sigmoid | 0.509 | 0.941 |
| 4 | 10 | sigmoid | 0.527 | 0.934 |

All four architectures ended close together on both tasks. The main difference was early training: the ReLU network started with far lower error than the sigmoid networks, and the smaller sigmoid networks started worst.

| Regression (wine quality) | Classification (spambase) |
|---|---|
| ![Validation MSE for model 2](Assignments/Assignment%203/figures/Regress_Model_2.png) | ![Validation accuracy for model 2](Assignments/Assignment%203/figures/Class_Model_2.png) |

Also in [the folder](Assignments/Assignment%203/): training curves for all eight models in `figures/` (`Regress_Model_1`–`4` and `Class_Model_1`–`4`), the report as `.docx` and a scratch script (`Test.py`).

## Assignment 4 — Convolutional neural networks

[Code](Assignments/Assignment%204/Assignement_4_JohnQuinn_CompSci711.py) · [Report](Assignments/Assignment%204/CompSci%20711%20Assignment%204%20Report.pdf)

Classifies photos of ten monkey species using the Kaggle dataset [10 Species of Monkey](https://www.kaggle.com/datasets/utkarshsaxenadn/10-species-of-monkey-multiclass-classification) (9,669 training images and 1,270 test images, resized to 100×100).

- **Task 1:** two custom CNNs, each with two convolution and max-pooling stages, a dense layer and dropout. Model 2 changes the filter counts and halves the dense layer.
- **Task 2:** transfer learning with EfficientNetV2S pretrained on ImageNet. The base is frozen and a global-average-pooling layer, a 1,024-node dense layer and a 10-way softmax are trained on top.
- **Task 3:** error analysis on test images the custom CNN got wrong, checking whether the fine-tuned model corrects them.

All three models were trained for 15 epochs.

| Model | Final training accuracy | Test accuracy |
|---|---|---|
| Custom CNN 1 | 0.972 | 0.711 |
| Custom CNN 2 | 0.944 | 0.669 |
| Fine-tuned EfficientNetV2S | 0.983 | 0.887 |

The fine-tuned model was 18 points more accurate on the test set than the better custom CNN. Both made the same kind of mistake, confusing species that look alike, such as gray langur with vervet monkey.

![A test image the custom CNN misclassified and the fine-tuned model got right](Assignments/Assignment%204/figures/Figure_1.png)

Also in [the folder](Assignments/Assignment%204/):

| File or folder | What it is |
|---|---|
| `Monkey Species Data/` | The image dataset: `Training Data/` and `Prediction Data/`, each with one subfolder per species |
| `CNN_Model_1.keras`, `CNN_Model_2.keras` | The two trained custom CNNs from Task 1 |
| `Pretrianed_CNN_Model.keras` | The fine-tuned EfficientNetV2S from Task 2 (94 MB) |
| `old models/my_cnn1.keras` | An earlier training run of the first CNN |
| `figures/` | Two of the error-analysis images from Task 3 |
| `Test.py`, `test.jpeg` | Scratch script and a single test image used while developing |

## Running the code

Assignments 1–3 download their datasets from OpenML on first run, so they need an internet connection. Assignment 4 reads the image dataset included in this repository and must be run from inside its own folder. Its Task 1 training block is commented out in the script as submitted; uncomment it, or load the saved models, before running the later sections.

```bash
pip install -r requirements.txt
python "Assignments/Assignment 1/JQ-CompSci711-Assign1.py"
```

Each script prints its result tables and opens its figures in a matplotlib window. Assignment 1 runs nested cross-validation on a 1,000-feature dataset and takes a while.

## Course topics

The course covered, in roughly this order:

1. Introduction to machine learning
2. Information-based learning (decision trees)
3. Evaluation measures and methodologies
4. Similarity-based learning (k-nearest neighbors)
5. Probability-based learning (naive Bayes)
6. Data exploration
7. Linear and logistic regression
8. Neural networks and deep learning
9. Convolutional neural networks
10. Recurrent neural networks
11. Attention and transformers
12. Generative models
13. Ensembles

The course textbook was *Fundamentals of Machine Learning for Predictive Data Analytics* by John D. Kelleher, Brian Mac Namee and Aoife D'Arcy (The MIT Press).

## Repository layout

```
README.md
requirements.txt
Assignments/
  Assignment 1/   code, report (.docx, .pdf), figures/
  Assignment 2/   code, report (.docx, .pdf), figures/, data/
  Assignment 3/   code, report (.docx, .pdf), figures/
  Assignment 4/   code, report (.docx, .pdf), figures/,
                  trained models (.keras), old models/, Monkey Species Data/
```

## Credits

- The assignment code, reports and figures are my own work.
- The course was taught by Rohit Kate at the University of Wisconsin–Milwaukee, who set the assignments.
- Datasets come from [OpenML](https://www.openml.org/) (steel-plates-fault, OVA_Prostate, Moneyball, wine_quality, spambase), the UCI Machine Learning Repository (Seoul bike sharing demand) and Kaggle ([10 Species of Monkey](https://www.kaggle.com/datasets/utkarshsaxenadn/10-species-of-monkey-multiclass-classification)).
