import pickle
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from flask import Flask
from flask import request
from flask import jsonify


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


n_estimators = [500]
max_depth = [30]
min_samples_split = [5]
model_file = f'model__n_estimators={n_estimators}_max_depth__{max_depth}_min_samples_split__{min_samples_split}.bin'

with open(model_file, 'rb') as f_in:
    best_model = pickle.load(f_in)

app = Flask('customer_sat')


@app.route('/predict', methods=['POST'])
def predict():
    X_customer = request.get_json()
    X_customer = pd.DataFrame([X_customer])
    y_pred = best_model.predict_proba(X_customer)[0, 1]
    satisfied = y_pred >= 0.5

    result = {
        'sat_probability': float(y_pred),
        'satisfied': bool(satisfied)
    }

    return jsonify(result)


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=9696)
