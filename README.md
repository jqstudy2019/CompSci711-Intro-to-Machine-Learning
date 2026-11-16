# Introduction to Machine Learning — Coursework

Assignments from COMPSCI 711: Introduction to Machine Learning (graduate level, University of Wisconsin–Milwaukee, Spring 2024), by John Quinn.

This repository holds everything from the course: my four assignments (code, reports, figures, data and trained models) and the course materials they were built on (lecture slides, syllabus, textbook and a sample midterm).

Each assignment picks a public dataset (from [OpenML](https://www.openml.org/) or Kaggle), trains and tunes models with scikit-learn or Keras, and compares them in a short written report.

**Contents**

- [Assignments](#assignments): [1](#assignment-1--decision-trees) · [2](#assignment-2--regression-model-comparison) · [3](#assignment-3--neural-networks) · [4](#assignment-4--convolutional-neural-networks)
- [Running the code](#running-the-code)
- [Course materials](#course-materials): [lecture slides](#lecture-slides) · [syllabus, textbook and exam](#syllabus-textbook-and-exam)
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

Also in [the folder](Assignments/Assignment%201/): the report as `.docx`, the instructor's `IDLE.pptx` and `sklearn.pptx` setup tutorials, and a scratch script (`test.py`).

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
- The report as `.docx` and the instructor's assignment handout (`A2.pptx`).

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

Also in [the folder](Assignments/Assignment%203/): training curves for all eight models in `figures/` (`Regress_Model_1`–`4` and `Class_Model_1`–`4`), the report as `.docx`, the instructor's handout (`A3.pptx`) and a scratch script (`Test.py`).

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
| `A4.pptx` | The instructor's assignment handout |
| `Test.py`, `test.jpeg` | Scratch script and a single test image used while developing |

## Running the code

Assignments 1–3 download their datasets from OpenML on first run, so they need an internet connection. Assignment 4 reads the image dataset included in this repository and must be run from inside its own folder. Its Task 1 training block is commented out in the script as submitted; uncomment it, or load the saved models, before running the later sections.

```bash
pip install -r requirements.txt
python "Assignments/Assignment 1/JQ-CompSci711-Assign1.py"
```

Each script prints its result tables and opens its figures in a matplotlib window. Assignment 1 runs nested cross-validation on a 1,000-feature dataset and takes a while.

## Course materials

### Lecture slides

The instructor's slide decks, in [Lecture Slides/](Lecture%20Slides/), one folder per topic.

| # | Topic | Decks |
|---|---|---|
| 01 | [Introduction](Lecture%20Slides/01%20Introduction/) | Parts 1–3 |
| 02 | [Decision Trees](Lecture%20Slides/02%20Decision%20Trees/) | Parts 1–2 |
| 03 | [Evaluation](Lecture%20Slides/03%20Evaluation/) | Parts 1–2 |
| 04 | [Similarity-Based Learning](Lecture%20Slides/04%20Similarity-Based%20Learning/) | Parts 1–3 |
| 05 | [Probability-Based Learning](Lecture%20Slides/05%20Probability-Based%20Learning/) | Parts 1–3 |
| 06 | [Data Exploration](Lecture%20Slides/06%20Data%20Exploration/) | Parts 1–2 |
| 07 | [Linear Regression](Lecture%20Slides/07%20Linear%20Regression/) | Parts 1–2 |
| 08 | [Logistic Regression](Lecture%20Slides/08%20Logistic%20Regression/) | Parts 1–2 |
| 09 | [Neural Networks](Lecture%20Slides/09%20Neural%20Networks/) | Parts 1–3 |
| 10 | [Deep Learning](Lecture%20Slides/10%20Deep%20Learning/) | 1 deck |
| 11 | [Ensembles](Lecture%20Slides/11%20Ensembles/) | 1 deck |
| 12 | [Convolutional Neural Networks](Lecture%20Slides/12%20CNN/) | Parts 1–2 |
| 13 | [Recurrent Neural Networks](Lecture%20Slides/13%20RNNs/) | 1 deck |
| 14 | [Attention and Transformers](Lecture%20Slides/14%20Attention%20and%20Transformers/) | 1 deck |
| 15 | [Generative Models](Lecture%20Slides/15%20Generative%20Models/) | 1 deck |

Folders 01–09 follow the order of topics in the syllabus. The numbering from 10 onward is my own ordering and may not match the order the lectures were given in.

### Syllabus, textbook and exam

- [Syllabus.pdf](Syllabus.pdf): the Spring 2024 course syllabus, with the topic list and grading scheme.
- [Textbook/](Textbook/): *Fundamentals of Machine Learning for Predictive Data Analytics* by Kelleher, Mac Namee and D'Arcy (MIT Press), the 2015 edition. The course lists the 2020 second edition as required.
- [Exams/](Exams/): the sample midterm and its solution.

## Repository layout

```
README.md
requirements.txt
Syllabus.pdf
Assignments/
  Assignment 1/   code, report (.docx, .pdf), figures/, setup tutorials
  Assignment 2/   code, report (.docx, .pdf), figures/, data/, handout
  Assignment 3/   code, report (.docx, .pdf), figures/, handout
  Assignment 4/   code, report (.docx, .pdf), figures/, handout,
                  trained models (.keras), old models/, Monkey Species Data/
Lecture Slides/   15 numbered topic folders of .pptx decks
Textbook/         course textbook (.pdf)
Exams/            sample midterm and solution (.pdf)
```

## Credits

- The assignment code, reports and figures are my own work.
- Lecture slides, assignment handouts, the syllabus and the sample midterm were written by the course instructor, Rohit Kate, at the University of Wisconsin–Milwaukee.
- The textbook is by John D. Kelleher, Brian Mac Namee and Aoife D'Arcy, published by The MIT Press.
- Datasets come from [OpenML](https://www.openml.org/) (steel-plates-fault, OVA_Prostate, Moneyball, wine_quality, spambase), the UCI Machine Learning Repository (Seoul bike sharing demand) and Kaggle ([10 Species of Monkey](https://www.kaggle.com/datasets/utkarshsaxenadn/10-species-of-monkey-multiclass-classification)).
