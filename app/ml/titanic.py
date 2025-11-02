# %%
import numpy as np 
import pandas as pd 
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score,confusion_matrix
import matplotlib.pyplot as plt 
import seaborn as sns

# %%
df=sns.load_dataset("titanic")

# %%
df.head()

# %%
df=df.drop(["alive","embark_town","deck","who"],axis=1)

# %%
df.head()

# %%
df["alone"].value_counts()

# %%
df["sex"].value_counts()

# %%
sns.countplot(x=df["adult_male"],hue=df["survived"])

# %%
df.head()

# %%
from sklearn.preprocessing import  OneHotEncoder,OrdinalEncoder

# %%
onehot=OneHotEncoder(drop="first",sparse_output=False)

part=onehot.fit_transform(df[["sex","adult_male","embarked"]])
df_part=pd.DataFrame(part,columns=onehot.get_feature_names_out())

# %%
df_part.head()

# %%
df=pd.concat([df,df_part],axis=1)

# %%
df=df.drop(["sex","adult_male"],axis=1)

# %%
sns.countplot(x=df["alone"],hue=df['survived'])

# %%
df["alone"]=df["alone"].map({False:0,True:1})

# %%
df["embarked"].value_counts()

# %%
df.head()

# %%
df.isnull().sum()

# %%
df.shape

# %%
df=df.dropna()

# %%
df=df.drop("embarked",axis=1)

# %%
df["class"].value_counts()

# %%
ordinal=OrdinalEncoder(categories=[["First","Third","Second"]])

df["class"]=ordinal.fit_transform(df[["class"]])

# %%
df.head()

# %%
sns.heatmap(df.corr()[["survived"]],annot=True)

# %%
# model=DecisionTreeClassifier()
# x,y=df.drop("survived",axis=1),df["survived"]

# x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=1)

# model.fit(x_train,y_train)

# y_pred=model.predict(x_test)
# acc=accuracy_score(y_test,y_pred)
# print("accuracy is -->",acc)

# %%
from sklearn.tree import plot_tree
from io import BytesIO
import base64

# %%



def tree(criterion="gini", max_depth=None, min_samples_leaf=1, min_samples_split=2):
    x,y=df.drop("survived",axis=1),df["survived"]

    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=1)

    # Train-test split

    # Decision Tree model
    model = DecisionTreeClassifier(
        criterion=criterion,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        min_samples_split=min_samples_split,
        random_state=42
    )

    model.fit(x_train, y_train)
    fn=model.feature_names_in_
    cn=model.classes_
    # Accuracy & Confusion Matrix
    y_pred = model.predict(x_test)
    acc = accuracy_score(y_test, y_pred)
    acc=round(acc,2)
    matrix = confusion_matrix(y_test, y_pred)

    # Tree depth and node count
    depth = model.get_depth()
    total_nodes = model.tree_.node_count

    # ✅ Generate image in memory (NO FILE SAVING)
    fig = plt.figure(figsize=(10, 7))
    plot_tree(model, filled=True, feature_names=fn, class_names=[str(c) for c in cn])
    plt.tight_layout()

    # Convert to base64
    buffer = BytesIO()
    fig.savefig(buffer, format="png")
    buffer.seek(0)
    img_base64 = base64.b64encode(buffer.getvalue()).decode()

    plt.close(fig)

    return acc, img_base64, matrix, depth, total_nodes


