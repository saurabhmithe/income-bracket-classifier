import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split

# read data from csv into a pandas data frame
census = pd.read_csv('census_data.csv')


# convert the label column to 0 and 1
def fix_label(label):
    return 0 if label == ' <=50K' else 1


census['income_bracket'] = census['income_bracket'].apply(fix_label)


# split the data into test data and train data
x_data = census.drop('income_bracket', axis=1)
y_labels = census['income_bracket']
# 30% data should go into test
X_train, X_test, y_train, y_test = train_test_split(x_data, y_labels, test_size=0.3, random_state=101)

# create feature columns for categorical values
gender = tf.feature_column.categorical_column_with_vocabulary_list("gender", ["Female", "Male"])
occupation = tf.feature_column.categorical_column_with_hash_bucket("occupation", hash_bucket_size=1000)
marital_status = tf.feature_column.categorical_column_with_hash_bucket("marital_status", hash_bucket_size=1000)
relationship = tf.feature_column.categorical_column_with_hash_bucket("relationship", hash_bucket_size=1000)
education = tf.feature_column.categorical_column_with_hash_bucket("education", hash_bucket_size=1000)
work_class = tf.feature_column.categorical_column_with_hash_bucket("workclass", hash_bucket_size=1000)
native_country = tf.feature_column.categorical_column_with_hash_bucket("native_country", hash_bucket_size=1000)

# create feature columns for continuous values
age = tf.feature_column.numeric_column("age")
education_num = tf.feature_column.numeric_column("education_num")
capital_gain = tf.feature_column.numeric_column("capital_gain")
capital_loss = tf.feature_column.numeric_column("capital_loss")
hours_per_week = tf.feature_column.numeric_column("hours_per_week")

feat_cols = [gender, occupation, marital_status, relationship, education, work_class, native_country, age, education_num, capital_gain, capital_loss, hours_per_week]

# to train the model
# batch size indicates how many record to read in one training batch to the model
# num_epochs indicates how many times the entire data needs to be passed to the model
# 1 epoch = batch_size * no. of batches
input_func = tf.estimator.inputs.pandas_input_fn(x=X_train, y=y_train, batch_size=100, num_epochs=None, shuffle=True)

# creating the model with the defined columns
model = tf.estimator.LinearClassifier(feature_columns=feat_cols)

model.train(input_fn=input_func, steps=5000)

# make predictions using the test data set
pred_fn = tf.estimator.inputs.pandas_input_fn(x=X_test, batch_size=len(X_test), shuffle=False)
predictions = list(model.predict(input_fn=pred_fn))
print(predictions[0])

final_preds = []
for pred in predictions:
    final_preds.append(pred['class_ids'][0])

print(final_preds[:10])
