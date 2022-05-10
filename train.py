#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
from sklearn import __version__ as sklearn_version
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, StandardScaler, PowerTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier
import pickle
import datetime

raw_data_clean = pd.read_csv('raw_data_clean.csv')

# ## 4.5 Train Test Split<a id='4.5'></a>

# In[23]:


X_train, X_test, y_train, y_test = train_test_split(raw_data_clean.drop(columns=['satisfaction', 'id']),
                                                    raw_data_clean.satisfaction,
                                                    test_size=0.2,
                                                    random_state=42)

print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)

categorical_cols = ['Gender', 'Type of Travel', 'Class', 'Customer Type']
ordinal_cols = list(raw_data_clean.loc[:, 'Inflight wifi service': 'Cleanliness'].columns)
continuous_cols_1 = ['Age', 'Flight Distance']
continuous_cols_2 = ['Departure Delay in Minutes', 'Arrival Delay in Minutes']


# Define custom transformer
class ColumnSelector(BaseEstimator, TransformerMixin):
    """Select only specified columns."""

    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[self.columns]


class ModeImputer(TransformerMixin):

    def __init__(self):
        """Impute missing values.

        Columns of dtype object are imputed with the most frequent value 
        in column.

        Columns of other types are imputed with mean of column.

        """

    def fit(self, X, y=None):
        self.fill = pd.Series([X[c].value_counts().index[0]
                               if X[c].dtype == np.dtype('O') else X[c].mean() for c in X],
                              index=X.columns)

        return self

    def transform(self, X, y=None):
        return X.fillna(self.fill)


cat_pipe = Pipeline([('selector', ColumnSelector(categorical_cols)),
                     ('imputer', ModeImputer()),
                     ('encoder', OneHotEncoder(handle_unknown='ignore', sparse=False))])


ord_pipe = Pipeline([('selector', ColumnSelector(ordinal_cols)),
                     ('imputer', SimpleImputer(strategy='constant', fill_value=0)),
                     ('scaler', MinMaxScaler())])


cont_1_pipe = Pipeline([('selector', ColumnSelector(continuous_cols_1)),
                        ('imputer', SimpleImputer(strategy='mean')),
                        ('scaler', StandardScaler())])


cont_2_pipe = Pipeline([('selector', ColumnSelector(continuous_cols_2)),
                        ('imputer', SimpleImputer(strategy='constant', fill_value=0)),
                        ('scaler', PowerTransformer())])


# Fit feature union to training data
preprocessor = FeatureUnion(transformer_list=[('cat', cat_pipe),
                                              ('ord', ord_pipe),
                                              ('cont_1', cont_1_pipe),
                                              ('cont_2', cont_2_pipe)])
preprocessor.fit(X_train)


# Prepare column names
cat_columns = preprocessor.transformer_list[0][1]['encoder'].get_feature_names(categorical_cols)
columns = np.append(np.append(np.append(cat_columns, ordinal_cols), continuous_cols_1), continuous_cols_2)

# Combine categorical and numerical pipeline
def tune_clf_func(model, X_train, X_test, y_test, y_train, param_grid):
    pipe = Pipeline(steps=[('preprocessor', preprocessor), ('model', model)])

    grid_cv = GridSearchCV(pipe, param_grid, cv=5, n_jobs=-1, scoring='roc_auc')
    grid_cv.fit(X_train, y_train)

    # Print the tuned parameters and score
    print('----------------------Hyper Parameter Tuning---------------------------\n')
    print(f"Tuned {model} Parameters: {grid_cv.best_params_}")
    print(f"Best score is {grid_cv.best_score_}")
    print('-----------------------------------------------------------------------\n')

    best_model = grid_cv.best_estimator_

    best_model.fit(X_train, y_train)

    # Predict training data
    y_train_pred = best_model.predict(X_train)
    print(f"Predictions on training data: {y_train_pred}")

    # Predict test data
    y_pred = best_model.predict(X_test)
    print(f"Predictions on test data: {y_pred}")

    print(confusion_matrix(y_test, y_pred))
    print(f'{best_model} Model AUC score {round(roc_auc_score(y_test, y_pred) * 100, 2)}%')
    print(classification_report(y_test, y_pred))

    return best_model, round(roc_auc_score(y_test, y_pred) * 100, 4)


scores = {}

print(scores)

X = raw_data_clean.drop(columns=['satisfaction', 'id'])
y = raw_data_clean.satisfaction
n_estimators = [500]
max_depth = [30]
min_samples_split = [5]
param_grid = {'model__n_estimators': n_estimators, 'model__max_depth': max_depth,
              'model__min_samples_split': min_samples_split}
pipe_final = Pipeline(steps=[('preprocessor', preprocessor), ('model', RandomForestClassifier())])
grid_cv_final = GridSearchCV(pipe_final, param_grid, cv=5, n_jobs=-1, scoring='roc_auc')
grid_cv_final.fit(X, y)


# Let's call this model version '1.0'
best_model = grid_cv_final.best_estimator_
best_model.version = '1.0'
best_model.pandas_version = pd.__version__
best_model.numpy_version = np.__version__
best_model.sklearn_version = sklearn_version
best_model.X_columns = [col for col in X_train.columns]
best_model.build_datetime = datetime.datetime.now()

# In[42]:


output_file = f'model__n_estimators={n_estimators}_max_depth__{max_depth}_min_samples_split__{min_samples_split}.bin'
print(output_file)


with open(output_file, 'wb') as f_out:
    pickle.dump((best_model), f_out)

