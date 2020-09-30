# Notebook NLP
* [Notebook Alex - Press here to open it in Colab](https://colab.research.google.com/github/AlexSperka/BigData_Project/blob/master/BigData_A.Sperka.ipynb)

# NYC fare prediction using Rapids and XG Boost running on GPU

[This notebook](TBD) shows my approach for predicting the fare amount for a taxi ride in NYC when given the pickup and dropoff locations of the passangers regarding the [New York City Taxi Fare Prediction Challange]( https://www.kaggle.com/c/new-york-city-taxi-fare-prediction).



---
# Introduction

This notebook is seperated into different sections, relating to the common data science workflow (except the hypothesis and data collection where already done). I tested different models to get used to Rapids and Cudf by starting with simple linear models and ended up with XG Boost and Light GBM. The best performant results are deffinitely the XG Boost ones. Different other Kernels state that they achieve similar results with LGBM hence I had problems with running the model on the GPU although I installed the right version (some bug reports also suggest there is a problem with the model).


0.   Previous Commits
1.   Setup and Check Infrastructure
2.   Having a first look at the Data (EDA)
3.   Data Cleaning (Feature Engineering)
4.   Linear Regression GPU
5.   Ridge Regression GPU
6.   K-Nearest Neighbor Regression GPU
7.   Random Forest GPU
8.   XG Boost on GPU
9.   Light GBM on GPU (not running on Colab)
10.   Stacked Ensemble XGB and LGBM
11.   Evaulation

---

## Approach:

I started with this project by getting used to the Rapids environment and the Cuda librarys and went on for more advanced models (XGB and LGBM) with testing them on common parameters mentioned in different notebooks on kaggle (see inspiration in first section) to get a feeling what influences what and how does it generelly work. After gaining some experience I combined several data cleaning approaches from differnt notebooks and general information from the internet. The most promising score yet was 3.19347 which would result in rank 432 of the public leaderboard (if the competition would still run). My kaggle notebook can be found [here, user handle: AlexS2020](https://www.kaggle.comrapdis-and-xg-boost-running-on-gpu)


## Evaluation:
As will be mentioned in the last section, I used to compare the different RMSE and R2 score values of different models as well as uploading the submission files to get a direct comperision via the score. Last step to evaluate are the confusion_matrices to see how many false positive/negative and true postivie/negative there are.

## Conclusion:
Overall it was an interesting project / competition which enabled me to learn new stuff and try out several tools. It is interesting to see, that the same code running on the GPU is so much faster compared to CPU implementations when looking at XGB for example. Interesting to me was also that relatively simple models like the KNN or Random Forest performed quite well in relation to the training runtime and ressource consumption. It would have been nice to directly compare the LBGM model against XGB since I read quite a bit about both. Unfortunately this was not possible within the desired scope due to runtime issues mostly. 

Another interesting aspect was to see the huge difference a properly cleaned dataset can make when using the same model.


## Outlook:
There are several ways how to improve these scores from my point of view. Some of them can be seen in other notebooks already (especially the larger ones).

1.   Run the code on several GPUs to figure out how well Rapids really performs in combination with DASK compared to one GPU or CPU.
2.   Ensemble different models based on their R2 score to test if the gap can be reduced this way (depending on the strength of the used models) An initial implementation for this is given in section 10. but I could not test it yet,
3.   Use larger grid searches to find better parameters for the given models
4.   An adaptive learning rate could also help with optimizing the results

## Limitations:
- It was possible to install the LBGM Version for GPU and the memory was consumed by the data but it did not run on the GPU somehow. So it was quite slow to do some training on this model and I could not compare it directly to XGB (or combine these two as intended in section 10)
- Some of the points mentioned in Outlook were not possible since running code in browser + only 36h GPU-time per week is hardly enough
- Automation of parameter search on large scale not feasible (not even with the GPU)

---

# Collaborators:
T. Dagner - https://github.com/dagnert
