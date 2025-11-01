import os
import numpy as np 
import pandas as pd 
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix
import base64
from io import BytesIO

def tree(criterion="gini", max_depth=None, min_samples_leaf=1, min_samples_split=2):

    # Load data
    data = load_iris()
    X, y = data.data, data.target
    fn, cn = data.feature_names, data.target_names

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Decision Tree model
    model = DecisionTreeClassifier(
        criterion=criterion,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        min_samples_split=min_samples_split,
        random_state=42
    )

    model.fit(X_train, y_train)

    # Accuracy & Confusion Matrix
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    acc=round(acc,2)
    matrix = confusion_matrix(y_test, y_pred)

    # Tree depth and node count
    depth = model.get_depth()
    total_nodes = model.tree_.node_count

    # ✅ Generate image in memory (NO FILE SAVING)
    fig = plt.figure(figsize=(10, 7))
    plot_tree(model, filled=True, feature_names=fn, class_names=cn)
    plt.tight_layout()

    # Convert to base64
    buffer = BytesIO()
    fig.savefig(buffer, format="png")
    buffer.seek(0)
    img_base64 = base64.b64encode(buffer.getvalue()).decode()

    plt.close(fig)

    return acc, img_base64, matrix, depth, total_nodes
