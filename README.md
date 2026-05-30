# 🚢 SCT_DS_2 — Titanic EDA (Exploratory Data Analysis)

## Task Overview
Perform data cleaning and exploratory data analysis on the Titanic dataset. Explore relationships between variables and identify patterns and trends.

## Dataset
**Source:** Kaggle — Titanic: Machine Learning from Disaster  
**Files:** `titanic/train.csv`, `titanic/test.csv`, `titanic/gender_submission.csv`  
**Records:** 891 passengers (training set)

## What the Script Does
- Loads the Titanic training dataset and inspects shape, missing values, and stats
- **Data Cleaning:**
  - Fills missing `Age` values with the median age
  - Fills missing `Embarked` with the mode
  - Drops the `Cabin` column (too many missing values)
- **Visualizations (6-panel figure):**
  1. Overall survival rate (pie chart)
  2. Age distribution (histogram)
  3. Survival rate by passenger class (bar chart)
  4. Survival rate by gender (bar chart)
  5. Age vs Survival comparison (overlapping histograms)
  6. Correlation heatmap of numeric features

## Key Findings
- Women had a significantly higher survival rate than men
- 1st class passengers survived at a much higher rate than 3rd class
- Younger passengers had a slightly better survival chance
- Fare and Pclass are strongly negatively correlated

## Project Structure
```
SCT_DS_2/
├── titanic/
│   ├── train.csv                  ← main training dataset
│   ├── test.csv
│   └── gender_submission.csv
├── SCT_DS_2.py                    ← main script
├── task02_titanic_eda.png         ← output visualization
└── README.md
```

## Libraries Used
| Library | Purpose |
|---|---|
| `pandas` | Data loading, cleaning, analysis |
| `matplotlib` | Multi-panel plotting |
| `seaborn` | Correlation heatmap |

## How to Run
```bash
cd SCT_DS_2
python SCT_DS_2.py
```

## Output
A 6-panel figure saved as `task02_titanic_eda.png` showing survival patterns across different features.

---
**Internship:** SkillCraft Technology — Data Science  
**Task:** 02 | Data Cleaning & EDA
