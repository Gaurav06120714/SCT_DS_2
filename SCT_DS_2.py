import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

os.makedirs("images", exist_ok=True)

train = pd.read_csv("titanic/train.csv")

print("Shape:", train.shape)
print("\nMissing values:\n", train.isnull().sum())
print("\nBasic stats:\n", train.describe())
print(f"\nCabin null %: {train['Cabin'].isnull().mean() * 100:.1f}%")

train["Age"]        = train["Age"].fillna(train["Age"].median())
train["Embarked"]   = train["Embarked"].fillna(train["Embarked"].mode()[0])
train.drop(columns=["Cabin"], inplace=True)
train["FamilySize"] = train["SibSp"] + train["Parch"] + 1

print("\nAfter cleaning — missing values:\n", train.isnull().sum())

survival_rate    = train.groupby("Pclass")["Survived"].mean() * 100
gender_survival  = train.groupby("Sex")["Survived"].mean() * 100
family_survival  = train.groupby("FamilySize")["Survived"].mean() * 100
embarked_survival = train.groupby("Embarked")["Survived"].mean() * 100

# ── Figure 1: Main EDA ────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle("Titanic Dataset — Exploratory Data Analysis", fontsize=16, fontweight="bold", y=1.01)

axes[0, 0].pie(
    train["Survived"].value_counts(),
    labels=["Did Not Survive", "Survived"],
    colors=["#E57373", "#81C784"],
    autopct="%1.1f%%",
    startangle=90,
    wedgeprops={"edgecolor": "white", "linewidth": 1.5},
)
axes[0, 0].set_title("Overall Survival Rate")

train["Age"].hist(bins=30, ax=axes[0, 1], color="#42A5F5", edgecolor="white")
axes[0, 1].set_title("Age Distribution")
axes[0, 1].set_xlabel("Age")
axes[0, 1].set_ylabel("Count")

survival_rate.plot(kind="bar", ax=axes[0, 2], color=["#EF9A9A", "#FFCC80", "#80CBC4"], edgecolor="white")
axes[0, 2].set_title("Survival Rate by Passenger Class")
axes[0, 2].set_xlabel("Pclass")
axes[0, 2].set_ylabel("Survival Rate (%)")
axes[0, 2].set_xticklabels(["1st", "2nd", "3rd"], rotation=0)

gender_survival.plot(kind="bar", ax=axes[1, 0], color=["#F48FB1", "#90CAF9"], edgecolor="white")
axes[1, 0].set_title("Survival Rate by Gender")
axes[1, 0].set_xlabel("Sex")
axes[1, 0].set_ylabel("Survival Rate (%)")
axes[1, 0].set_xticklabels(["Female", "Male"], rotation=0)
axes[1, 0].set_ylim(0, 100)

train[train["Survived"] == 1]["Age"].hist(bins=25, ax=axes[1, 1], alpha=0.7, label="Survived",     color="#66BB6A")
train[train["Survived"] == 0]["Age"].hist(bins=25, ax=axes[1, 1], alpha=0.7, label="Not Survived", color="#EF5350")
axes[1, 1].set_title("Age vs Survival")
axes[1, 1].set_xlabel("Age")
axes[1, 1].legend()

sns.heatmap(
    train[["Survived", "Pclass", "Age", "SibSp", "Parch", "Fare", "FamilySize"]].corr(),
    annot=True, fmt=".2f", cmap="coolwarm", ax=axes[1, 2],
    linewidths=0.5, linecolor="white",
)
axes[1, 2].set_title("Correlation Heatmap")

plt.tight_layout()
plt.savefig("images/task02_titanic_eda.png", dpi=150, bbox_inches="tight")
plt.show()

# ── Figure 2: Extended Analysis ───────────────────────────────────────────────
fig2, axes2 = plt.subplots(1, 3, figsize=(18, 5))
fig2.suptitle("Titanic — Extended Analysis", fontsize=14, fontweight="bold")

train["Fare"].hist(bins=40, ax=axes2[0], color="#FFA726", edgecolor="white")
axes2[0].set_title("Fare Distribution")
axes2[0].set_xlabel("Fare ($)")
axes2[0].set_ylabel("Count")

sns.boxplot(x=train["Fare"], ax=axes2[1], color="#FFD54F")
axes2[1].set_title("Fare Outliers")
axes2[1].set_xlabel("Fare ($)")

embarked_survival.plot(kind="bar", ax=axes2[2], color=["#CE93D8", "#80DEEA", "#FFAB91"], edgecolor="white")
axes2[2].set_title("Survival Rate by Embarkation Port")
axes2[2].set_xlabel("Port (C=Cherbourg, Q=Queenstown, S=Southampton)")
axes2[2].set_ylabel("Survival Rate (%)")
axes2[2].set_xticklabels(["C", "Q", "S"], rotation=0)

plt.tight_layout()
plt.savefig("images/task02_titanic_extended.png", dpi=150, bbox_inches="tight")
plt.show()

# ── Figure 3: Family Size ─────────────────────────────────────────────────────
fig3, ax3 = plt.subplots(figsize=(10, 5))
family_survival.plot(kind="bar", ax=ax3, color="#80CBC4", edgecolor="white")
ax3.set_title("Survival Rate by Family Size")
ax3.set_xlabel("Family Size (self + siblings/parents/children)")
ax3.set_ylabel("Survival Rate (%)")
ax3.set_xticklabels(ax3.get_xticklabels(), rotation=0)
plt.tight_layout()
plt.savefig("images/task02_titanic_family.png", dpi=150, bbox_inches="tight")
plt.show()

train.to_csv("Titanic_Cleaned.csv", index=False)
print("\nAll charts saved to images/")
print("Cleaned dataset saved as Titanic_Cleaned.csv")
