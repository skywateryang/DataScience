import pandas as pd
import streamlit as st
from sklearn.datasets import load_breast_cancer
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from PIL import Image
from sklearn.tree import export_graphviz
from io import StringIO
import pydotplus
import os
import numpy as np
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt

# add graphviz to path so that decision tree visualizaton can be showed in web
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'

st.header('This is the machine learning playground for decision tree.')


# load and cache raw data
@st.cache
def get_data():
    data = load_breast_cancer()
    cancer = pd.DataFrame(data.data, columns=data.feature_names)
    cancer['target'] = data.target
    return cancer


df = get_data()

# sidebar configure
st.sidebar.header('Configure model')
criterion = st.sidebar.selectbox(
    'Set criterion',
    ('gini', 'entropy')
)

max_depth = st.sidebar.slider(
    'Set max depth',
    1, 20
)

min_samples_split = st.sidebar.slider(
    'Set min samples split',
    2, 15
)

ccp_alpha = st.sidebar.selectbox(
    'Set cost-complexity pruning',
    (0, 0.001, 0.005, 0.008, 0.01, 0.02, 0.05, 0.1, 0.2, 1)
)

# set display data or not
if st.checkbox('Review dataset'):
    st.dataframe(df)

# build model
X_train, X_test, y_train, y_test = train_test_split(df.iloc[:, :-1], df['target'], test_size=0.3, stratify=df['target'],
                                                    random_state=42)
clf = DecisionTreeClassifier(criterion=criterion, max_depth=max_depth, min_samples_split=min_samples_split,
                             ccp_alpha=ccp_alpha,
                             random_state=42)
clf.fit(X_train, y_train)
score_train = clf.score(X_train, y_train)
score_test = clf.score(X_test, y_test)

st.write('Training Accuracy: ', round(score_train, 3))
st.write('Test Accuracy: ', round(score_test, 3))

# visualzie tree
dotfile = StringIO()
export_graphviz(clf, out_file=dotfile, feature_names=df.columns[:-1], class_names=['malignant', 'benign'],
                rounded=True, proportion=False,
                precision=2, filled=True)

graph = pydotplus.graph_from_dot_data(dotfile.getvalue())

graph.write_png("dtree.png")

image = Image.open('dtree.png')
st.image(image, caption='Decision tree visualization', use_column_width=True)

# learning curve
X = df.iloc[:, :-1]
y = df.target
cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=0)
train_sizes = np.linspace(0.1, 1.0, 5)

st.subheader('Learning curve review')


@st.cache
def plot_leanring_curve(model, X, y, cv, train_sizes):
    train_sizes, train_scores, test_scores = learning_curve(model, X=X, y=y, cv=cv, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.set_xlabel("Training examples",fontsize=15)
    ax.set_ylabel("Accuracy",fontsize=15)
    ax.set_title("Model learning curve",fontsize=20)
    ax.grid()
    ax.fill_between(train_sizes, train_scores_mean - train_scores_std,
                    train_scores_mean + train_scores_std, alpha=0.1,
                    color="r")
    ax.fill_between(train_sizes, test_scores_mean - test_scores_std,
                    test_scores_mean + test_scores_std, alpha=0.1,
                    color="g")
    ax.plot(train_sizes, train_scores_mean, 'o-', color="r",
            label="Training score")
    ax.plot(train_sizes, test_scores_mean, 'o-', color="g",
            label="Test score")
    ax.legend(loc="best",fontsize=14)


plot_leanring_curve(clf, X, y, cv, train_sizes)
st.pyplot()


