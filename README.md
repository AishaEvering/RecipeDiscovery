<p align="center">
   <img src="https://github.com/AishaEvering/RecipeDiscovery/blob/main/recipe_loader.jpeg" height="300px" alt="Recipe Discovery Header">
</p>

# Recipe Discovery

For this project, I was tasked with analyzing data for a fictitious company that operates a recipe-hosting website. The company aims to identify which recipes
drive the highest web traffic. The goal is to develop a model that achieves at least 80% accuracy in predicting the traffic-driving potential of recipes.

## Technologies
[![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)](https://www.python.org/)
[![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)](https://numpy.org/)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-%23ffffff.svg?style=for-the-badge&logo=Matplotlib&logoColor=black)](https://matplotlib.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/stable/)
[![Jupyter Notebook](https://img.shields.io/badge/jupyter-%23FA0F00.svg?style=for-the-badge&logo=jupyter&logoColor=white)](https://jupyter.org/)

## 📃Overview

This binary classification problem was tackled using logistic regression. The goal was to develop a model with at least 80% accuracy that reliably identifies 
recipes likely to drive significant web traffic. In accordance with the requirements, I built and trained two different models and compared their performance.

## ⚙️ The Process

1. ${{\color{teal}\Huge{\textsf{Data\ Validation\ \}}}}\$: The first and crucial step, beyond simply loading the data, is to clean it effectively. In this project, I addressed issues such as missing values 
and unexpected categories—for example, where "Chicken Breast" was listed under a category intended for "Chicken." I handled missing values by imputing them with the column mean. 
Identifying and correcting these issues before training the model is essential for achieving reliable results.

2. ${{\color{teal}\Huge{\textsf{Exploratory\ Data\ Analysis\ \}}}}\$: Analyzing data summaries, such as feature distributions, provided valuable insights into the dataset. I identified the top three recipes
driving high traffic as Vegetable, Potato, and Pork, with a serving size of 4. During EDA, I observed some features with right-skewed distributions due to outliers.
To ensure accurate model performance, I aimed to approach a normal distribution but refrained from applying transformations at this stage to avoid data leakage.
I planned to address these issues after splitting the data.

3. ${{\color{teal}\Huge{\textsf{Model\ Development\ \}}}}\$: The task was to create both a baseline and a comparison model. For the baseline, I selected a Logistic Regression model with regularization
to manage outliers effectively. For comparison, I chose a Random Forest model, known for its robustness to outliers and
strong performance in practice. I started by splitting the data into 80% training and 20% test sets.
    <p>
      In the preprocessing pipeline, I applied log transformation to normalize right-skewed data and reduce the impact of outliers. I also used a RobustScaler for further 
      stabilization. Categorical features were encoded using One-Hot Encoding. After setting up the pipeline, I trained both models.
    </p>
    <p>
      It was insightful to observe feature importance. For the Logistic Regression model, the most significant feature was the category, which was expected. In contrast, 
      the Random Forest model prioritized recipe details such as protein, calories, and sugar. The effectiveness of each model will be evaluated in the next step, Model Evaluation.
    </p>
6. ${{\color{teal}\Huge{\textsf{Model\ Evaluation\ \}}}}\$: Based on the results and the business requirement of "Correctly predict high traffic
recipes 80% of the time," the best-performing model is the Comparison Model (Random Forest Classifier).

   - Baseline Model (Logistic Regression) Results:
      - **Precision for High Traffic (0.85):** Of the recipes predicted as high traffic, 85% were indeed high traffic.
      - **Recall for High Traffic (0.77): 77%** of the actual high traffic recipes were correctly identified.
      - **F1-Score for High Traffic (0.81):** This score provides a balance between precision and recall.
        
   - Comparison Model (Random Forest Classifier) Results:
      - **Precision for High Traffic (0.82):** Of the recipes predicted as high traffic, 82% were indeed high traffic.
      - **Recall for High Traffic (0.80):** 80% of the actual high traffic recipes were correctly identified.
      - **F1-Score for High Traffic (0.81):** This score provides a balance between precision and recall.

## 🔑 Key Takeaways

* **Data Cleaning:** Ensuring clean data is crucial; as the saying goes, "*Garbage in, garbage out.*"
* **Data Understanding and Normalization:** After cleaning, understanding the data distribution and normalizing it is essential.
* **Preventing Data Leakage:** Always split the data before applying transformations to avoid data leakage.
* **Random Forest Models:** I have consistently achieved good results with Random Forest models (both Regression and Classification). A Random Forest is an ensemble of decision trees. While individual decision trees are simple and interpretable, they can overfit. Random Forests address this by training multiple decision trees on random subsets of features, which helps prevent overfitting and improves model performance.

### 😤 Challenges Faced

Oh, the pressure! I’m comfortable with data cleaning, processing, and model training, but this project carried a lot of weight. After working so long to get to this point, I was finally facing the final submission. I meticulously reviewed the notebook multiple times, knowing how particular DataCamp is about their requirements. Plus, there was the added pressure of a time limit 😰. But I did it! I’m now moving on to more complex model architectures, but for this moment, I’m celebrating this accomplishment with a 🥳.

### ☑️ DataCamp Portfilio

If you'd like to see more of the projects I've completed on DataCamp, feel free to check out my (Datacamp Porfolio)[https://www.datacamp.com/portfolio/shovon3000g]. 
  
## Author

Aisha Evering  
[Email](<shovon3000g@gmail.com>) | [Portfolio](https://aishaeportfolio.com/)

