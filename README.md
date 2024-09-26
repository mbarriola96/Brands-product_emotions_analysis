# Brands-product_emotions_analysis

-------------------------------------------

## Description

This project focuses on helping companies optimize their marketing strategies by analyzing public sentiment towards technological products using social media data. As a third-party consulting firm, we specialize in sentiment analysis to identify which features of products are most appreciated by consumers, enabling businesses to fine-tune their marketing and product development efforts. For this analysis, we utilized a semi-structured dataset of 9,093 tweets, each labeled with a sentiment—positive, negative, neutral, or no emotion—associated with major brands like Apple and Google.

Our goal was to detect tweets expressing positive sentiment, critical for companies seeking to understand consumer feedback. After preprocessing the dataset and applying various machine learning models, we found that XGBoost performed best, achieving a precision score of 0.78 on the training set and 0.67 on the test set.

In this project, we have focused on identifying the overall sentiment towards technological products, but the next step would involve providing detailed reports on which specific features of these products are receiving the most positive feedback. This additional insight would allow companies to prioritize enhancements to features that resonate most with consumers, further optimizing product offerings and marketing campaigns. Ultimately, this approach would enable businesses to make data-driven decisions and stay competitive in the ever-evolving technological landscape.


## Sources

The dataset consists of 9,093 tweets related to technological products, each labeled with one of four sentiment categories: positive, negative, neutral, or no emotion. In addition to sentiment, each tweet is associated with a specific technological product from brands such as Apple or Google. We were provided the following dataset:

- judge-1377884607_tweet_product_company

The dataset can be found in this site: 

https://data.world/crowdflower/brands-and-product-emotions


## Methodology

Four notebooks were created in order to analyze the data and based on that predict the dependent variable (ie whether the sentiment of a tweet is positive or not):

- 1. 00_primary_notebook.

In this notebook, there is a high level overview of all the steps carried out in this 	project in order to do train and test our model. Moreover, the notebooks contains the 	links to the different notebooks in case the reader wants a deeper understanding of the 	processes. 

- 2. 01_data_understanding notebook. 

In this notebook, there is a high-level overview of all the steps carried out in this project to train and test our model. Moreover, the notebook contains links to the different notebooks in case the reader wants a deeper understanding of the processes.

Additionally, this notebook contains the pipeline that includes all transformations such as TF-IDF, One-Hot Encoding, and the best-performing model chosen during the model selection process. We applied this pipeline to the X_test dataset, achieving the same results as those obtained separately in the 03_modelling notebook.

- 3. 02_data_preprocessing.

In this notebook, we imported the data from notebook 01_data_understanding and applied transformations to clean the dataset for prediction. We used One-Hot Encoding and Label Encoding on categorical columns, scaled the numerical columns, and carried out an NLP process on the tweet text, including generalizing product names, removing stopwords, and tokenizing the text for analysis.

- 4. 03_model_creation.

In this notebook, we imported the transformed data from notebook 02_data_preprocessing 	to create our models. After trying several models such as Logistic Regression, XGBoost, Random Forest, and Neural Networks; we used the precision score to compare performance among the models and we selected XGBoost as the final model due to its superior results. We achieved 0.78 on the training set and 0.67 on the test set. The XGBoost model was then deployed to accurately identify tweets with positive sentiment, providing key insights for companies to enhance their product strategies based on consumer feedback. 


## Conclusion

Looking into the distribution of the dependent variable. There is clearly no imbalance, as can be seen below:

![Distribution Positive and Non-positive](/visualizations/bar_graph_target_variable.png)

Moreover, we have built the best model to detect tweets with positive sentiment, achieving 58% success by optimizing for precision. Our focus is to avoid mistakes when identifying a tweet as having a positive emotion. The reason for this is tied to our business case and the next steps of this project, which involve distinguishing the characteristics of technological products that evoke positive emotions in the public. If we fail to minimize false positives, this would negatively impact our future model aimed at identifying the features of a technological product that trigger positive emotions.

This model enables us to gain reasonable insights when extracting positive features from tweets. By focusing on precision, we ensure that the positive sentiment tweets we identify are highly likely to reflect genuine positivity, which is crucial for the reliability of subsequent analyses.

Here we have the confusion matrix of the XGBoost that was selected:

![Confusion Matrix](/visualizations/confusion_matrix.png)

As seen in the confusion matrix, a false positive rate of 31.01% in predicting positive sentiment means that a notable portion of non-positive tweets are incorrectly classified as positive. This could impact companies' understanding of consumer feedback. However, with a true positive rate of 57.62%, the model effectively identifies over half of the positive sentiment tweets, providing valuable insights for marketing and product strategies.

To improve these results, we could focus on two key areas:

- First, increasing the size of the training dataset would allow the model to learn from more examples, improving its ability to generalize. It's also important to increase the number of tweets with positive sentiment, as the dataset is currently imbalanced with more non-positive tweets. A more balanced dataset would prevent the model from becoming biased towards the majority class, ensuring a fair representation of both positive and non-positive sentiments.

- Second, incorporating bigrams or trigrams (n-grams) would enable the model to capture word dependencies that are not evident with single tokens, thus enhancing its ability to identify sentiment patterns more accurately.

In the future, the company's intention is that once the positive tweets are extracted, a model will be built to determine the characteristics of technological products that lead to positive sentiment towards the product. This approach will provide deeper insights into what aspects of these products resonate positively with the audience, further enhancing our overall analysis.

## Author

My name is Miguel Barriola Arranz. I am an Industrial Engineer and a Duke graduate student in Engineering Management. 
I am currently working in the microchip industry and further expanding my skillset in data science. 

- LinkedIn: https://www.linkedin.com/in/miguel-barriola-arranz/
- Medium: https://medium.com/@mbarriolaarranz

## Technologies

I have used **Python** with Jupyter notebook.

## Project Status

The project is in a development process at this moment. 

## What to find in the repository

There is a folder called notebooks that contains all the used notebooks and a python file named project_functions.py. This file is used to store all the functions that were created in this project.

There is a requirements.txt that contains the information of the libraries used in this project.

There is a .gitignore that allows to exclude files that are of no interest.

There is a results_pdf_files folder that contains the resultant final presentation and the notebook where the analysis was carried out in pdf format.  

