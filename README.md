# CCFraud
Small report for Active Learning methods for Credit Card Fraud as part of course 02463 in DTU

Data from Kaggle:
https://www.kaggle.com/datasets/dhanushnarayananr/credit-card-fraud

# Astract
Acquiring accurately labeled data for training machine learning models can be time-consuming and expensive, posing challenges in various domains, including fraudulent credit card transactions. This project explores the potential of active machine learning, specifically query by committee (QBC) and uncertainty sampling, in improving the convergence speed and accuracy compared to random sampling. The study focuses on a pool-based active learning scenario using a logistic regression model to categorize credit card transactions as valid or fraudulent. Fifteen permutations of the dataset are evaluated using different query strategies, starting with random initial points and iteratively selecting points based on the chosen strategy. The accuracy of the trained models is tested on separate test datasets. The results demonstrate that both least confidence uncertainty sampling and query by committee outperform random sampling, with least confidence achieving the highest mean accuracy. Furthermore, these active learning methods exhibit lower standard deviation, indicating more consistent model performance across multiple runs. It is acknowledged that different query strategies may yield better results depending on the specific scenario and dataset. Additionally, it should be noted that QBC requires additional training time compared to other methods.

# Relevant results
<img src="https://github.com/TheLucanus/CCFraud/blob/main/figures/uncertain.png" width="500" height=400\>

|   | Mean | Std |
|---|-----|-----|
| QBC | 0.831| 0.011 |
| Least Confidence | 0.832| 0.007  |
| Random | 0.633 |  0.064 |
