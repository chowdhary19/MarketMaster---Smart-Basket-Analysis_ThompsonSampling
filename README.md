
# Optimizing Ad Selection Using Thompson Sampling Algorithm

 ![Thompson Sampling](https://img.shields.io/badge/Thompson_Sampling-blue.svg)
 ![Reinforcement Learning](https://img.shields.io/badge/Reinforcement_Learning-red.svg)
 ![Machine Learning](https://img.shields.io/badge/Machine_Learning-yellow.svg)
 ![MIT License](https://img.shields.io/badge/License-MIT-green.svg)

## Objective
The task is to optimize ad selection by identifying the best-performing ad using the Thompson Sampling algorithm. This method helps in maximizing the click-through rate (CTR) by leveraging the principles of reinforcement learning.

## Developer Information
Name: Yuvraj Singh Chowdhary  
LinkedIn: [Connect with me](https://www.linkedin.com/in/yuvraj-singh-chowdhary/)  
GitHub Repo: [MarketMaster - Smart Basket Analysis Thompson Sampling](https://github.com/chowdhary19/MarketMaster---Smart-Basket-Analysis_ThompsonSampling.git)  
Reddit: [Connect on Reddit](https://www.reddit.com/user/SuccessfulStrain9533/)

## Overview
This project uses the Thompson Sampling algorithm to analyze a dataset of user interactions with 10 ads over 10,000 rounds. The goal is to identify the ad with the highest click-through rate and optimize marketing strategies accordingly.
![Formula](source/1.png)

## Dataset
The dataset used in this project is `ads_ctr_optimisation.csv`, containing 10,000 user interactions over 10,000 rounds for 10 different ads.

## Libraries Used
- numpy: For numerical operations.
- pandas: For data manipulation and analysis.
- matplotlib: For data visualization.

## Installation
To install the necessary libraries, run the following command:
```sh
pip install numpy pandas matplotlib
```

## Data Preprocessing
The dataset is preprocessed to format the data into a suitable structure for applying the Thompson Sampling algorithm.

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
dataset = pd.read_csv('ads_ctr_optimisation.csv')

# Implementing Thompson Sampling
import random

N = 10000
d = 10
ads_selected = []
numbers_of_rewards_1 = [0] * d
numbers_of_rewards_0 = [0] * d
total_reward = 0

for n in range(0, N):
    ad = 0
    max_random = 0
    for i in range(0, d):
        random_beta = random.betavariate(numbers_of_rewards_1[i] + 1, numbers_of_rewards_0[i] + 1)
        if random_beta > max_random:
            max_random = random_beta
            ad = i
    ads_selected.append(ad)
    reward = dataset.values[n, ad]
    if reward == 1:
        numbers_of_rewards_1[ad] += 1
    else:
        numbers_of_rewards_0[ad] += 1
    total_reward += reward

# Visualizing the results
plt.hist(ads_selected)
plt.title('Histogram of ads selections')
plt.xlabel('Ads')
plt.ylabel('Number of times each ad was selected')
plt.show()
```

## Key Findings
From the analysis, ad 4 was identified as having the highest click-through rate. This insight can be used to optimize ad display strategies to maximize engagement.

## Instructions for Collaborators and Users
- Modify Data Preprocessing: Adapt the data preprocessing phase to fit your business dataset. Ensure that each user interaction is represented correctly.
- Adjust Algorithm Parameters: Depending on your business requirements, you might need to adjust the parameters of the Thompson Sampling algorithm to suit your specific needs.
- Analyze Results: Use the provided code to inspect the results and identify the best-performing ads.

## Example Adjustments
- Data Preprocessing: If your dataset has a different structure, modify the preprocessing code to correctly format your user interactions.
- Algorithm Parameters: Depending on the nature of your data, you may need to tweak the parameters for optimal performance.

## Conclusion
This project demonstrates the application of the Thompson Sampling algorithm for optimizing ad selection. By identifying the best-performing ads, businesses can make informed decisions to enhance their marketing strategies.



The dataset and full source code have been deployed in the GitHub repository. Looking forward to positive collaborations!


```

This README file includes all necessary information, follows the specified format, and incorporates the requested logos and image from the Source folder.
