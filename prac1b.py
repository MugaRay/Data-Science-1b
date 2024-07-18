# %%
%matplotlib inline
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import numpy as np

# %% [markdown]
# ## Task 1: Data transformation

# %%
# importing the dataset
df = pd.read_csv("titanic.csv")
df.describe()

# %%
# 1 
df["FamilySize"] = df["Parch"] + df["SibSp"] + 1 
print(f"The max size of FamilySize is {df["FamilySize"].max()}")

# %%
# 2
df["FareCategory"] = pd.cut(df["Fare"], bins=[0, 10, 20, 30, 50, df["Fare"].max()])

df.head()

# %%
print(df["FareCategory"].value_counts())


# %%
# 3

def extract_Title(x):
    split_name = x.split(',')
    return split_name[1].split(".")[0]


df["Title"] = df["Name"].map(extract_Title)

print(f"There are {df["Title"].nunique()} unquie values")

# %% [markdown]
# ## Data Reduction

# %%
df.columns

# %%
df.drop(["PassengerId"], axis=1, inplace=True)  # PassengerID does not give any insignht whenter one Survived and is just an index
df.drop(["Name"],  axis=1, inplace=True)         # Name is purely random data (bunch of words) and does not show if someone survived, Title is a better metric since it shows class and importance

print(df["Ticket"].value_counts())
print(df.shape)
df.drop(["Ticket"],  axis=1, inplace=True)             # There are too many unique Tickets to extract any valuable data to identify if someone survived or not. There are 681 unique entries which seem to have no consistency


# %%
def convertCatergorical_df(df):
    # making a copy of the data
    df_copy = df.copy()

    # converting all the categorical data into Intgers
    for i in df_copy.columns:
        df_copy[i] = LabelEncoder().fit_transform(df_copy[i])
    
    return df_copy


def create_corrleted_df(df, target):

    df_copy = convertCatergorical_df(df)
    matrix = df_copy.corr()   # making a correlation matrix 
    top_5 = matrix.nlargest(6, target).index
    return df[top_5].copy()

# %%
df_corr_columns = create_corrleted_df(df, "Survived")

df_corr_columns.head()

# %%
df_heatmap = convertCatergorical_df(df_corr_columns) # to stop FareCategory or any categorial data from giving me issues

corr_df_heatmap = df_heatmap.corr().abs()
mask = np.zeros_like(corr_df_heatmap)
mask[np.triu_indices_from(mask)] = True
sns.heatmap(corr_df_heatmap, annot=True, cmap="coolwarm", mask=mask)


