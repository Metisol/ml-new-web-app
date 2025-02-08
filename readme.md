
  # 🗳️ Election Win Predictor

A **machine learning-powered web application** that predicts whether a candidate will **win or lose** an election based on various factors such as total votes, assets, criminal cases, and party affiliation.

---

## 📌 1. Problem Definition  
Elections play a crucial role in democracy, and predicting the chances of winning can be beneficial for **political analysts, candidates, and strategists**.  
The goal of this project is to build a **machine learning model** that predicts whether a candidate will win or lose based on past election data.

---

## 📌 2. Data Source & Description  

- The dataset used is **Election.csv**, containing **real election data** with details on candidates, votes received, assets, liabilities, and more.  
- It includes **both numerical and categorical features**.
- **Target Variable:** `"WINNER"` (1 = Won, 0 = Lost).

### **Dataset Features**
| **Feature**        | **Description** |
|--------------------|---------------|
| AGE               | Candidate's age |
| TOTAL_VOTES       | Total votes received |
| GENERAL_VOTES     | General category votes |
| POSTAL_VOTES      | Postal votes received |
| TOTAL_ELECTORS    | Total voters in constituency |
| CRIMINAL_CASES    | Number of criminal cases |
| ASSETS            | Candidate's total assets (₹) |
| LIABILITIES       | Candidate's total liabilities (₹) |
| EDUCATION         | Education level (Encoded) |
| CATEGORY          | Candidate category (General, OBC, SC, etc.) |
| GENDER           | Candidate gender (0=Male, 1=Female) |
| PARTY            | Political party (Encoded number) |

---

## 📌 3. Exploratory Data Analysis (EDA)  

- **Distribution of Winners vs. Losers**
- **Feature Correlations (Heatmaps)**
- **Vote Share Distribution**
- **Effect of Party & Education on Winning Chances**

✅ **EDA Results and Visualizations** can be found in the `EDA_report.ipynb` (if applicable).

---

## 📌 4. Data Preprocessing  

### **Key Steps:**
✅ **Handling Missing Values:**  
- Categorical columns (`SYMBOL`, `GENDER`, `CATEGORY`, `EDUCATION`) filled with `"Unknown"`.  
- Numerical columns filled with median values.  

✅ **Feature Engineering & Cleaning:**  
- **Converted currency fields** (ASSETS, LIABILITIES) from `"Rs 5 Crore"` to numeric values.  
- **Encoded categorical variables** (`GENDER`, `EDUCATION`, `CATEGORY`, `PARTY`) using **Label Encoding**.  

✅ **Balanced Training Data:**  
- Used **undersampling** to ensure equal `"WINNER"` and `"LOST"` cases.

---

## 📌 5. Model Selection & Training  

### **Selected Model: RandomForestClassifier**  
- Chosen due to its **high accuracy** and ability to handle mixed data types.  
- **Trained using:**  
  ```python
  rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
  rf_model.fit(X_train, y_train)


"# ml-assignment-" 
"# ml-projects-" 
"# ml-web-app" 
"# ml-web-app" 
