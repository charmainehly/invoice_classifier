#!/usr/bin/env python
# coding: utf-8

# # Using elasticsearch fuzzy match and machine learning methods to categorise short text descriptions

# Prerequisites:
# - setup virtual env: `python -m venv <your_new_virtual_environment_name>`
# - activate virtual env: `source <your_new_virtual_environment_name>`
# - clone and install this repo `pip install git+https://github.com/chilledgeek/elasticsearch-simple-client.git`
# - install extra packages for machine learning: `pip install tensorflow matplotlib scikit-learn xgboost`
# - have a running elasticsearch docker image: 
#   - `sudo docker pull elasticsearch:7.5.2`
#   - `sudo docker run -d --name elasticsearch -p 9200:9200 -p 9300:9300 -e "discovery.type=single-node" elasticsearch:7.5.2` 
#   - (`-d` runs command line in background)
# - If required, clear out previous data on elasticsearch with: `curl -X DELETE "localhost:9200/<index_name>"`

# In[1]:


get_ipython().system('curl -X DELETE "localhost:9200/simple_text"')


# In[2]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
plt.style.use('dark_background')


# ## Data preparation (mainly for machine learning techniques)

# ### Load data

# In[3]:


import pandas as pd

filepath = "business_category_train.csv"
df = pd.read_csv(filepath)
df.head()


# #### Filter out categories that have less than 10 annotated entries

# In[4]:


indices_of_interest = df["Category"].value_counts()[df["Category"].value_counts() >= 10 ].index


# In[5]:


df = df[df["Category"].isin(indices_of_interest)]


# ### Label encode categories and apply to category

# In[6]:


from sklearn import preprocessing

le = preprocessing.LabelEncoder()
le.fit(["UNKNOWN"] + list(df["Category"])) # Add an extra UNKNOWN label in case outcome cannot be predicted


# In[7]:


df["Category (encoded)"] = le.transform(df["Category"])
df.head()


# ### Split data to train and test sets

# In[8]:


import random

# Set a seed
random.seed(123)


# In[9]:


raw_train = df.sample(frac=0.8).sort_index()
raw_train.head()


# In[10]:


raw_test = df[~df.index.isin(raw_train.index)]
raw_test.head()


# In[11]:


print(f"training entries: {len(raw_train)}")
print(f"test entries: {len(raw_test)}")
print(f"number of unique categories (with enough annotations): {len(set(df['Category']))}")


# In[12]:


accumulated_category_count_df = pd.concat([
    raw_train["Category"].value_counts(),
    raw_test["Category"].value_counts()], 
    axis=1, 
    sort=False,
    keys = ["Train", "Test"])
accumulated_category_count_df.plot(kind="bar", figsize=(10,5), title="Category occurrence")


# #### Create a bag of words using a count vectorizer

# In[13]:


from sklearn.feature_extraction.text import CountVectorizer

desc_vectorizer = CountVectorizer(analyzer="word", max_features=100)

training_bag_of_words = desc_vectorizer.fit_transform(raw_train["Expense"])

feature_names = desc_vectorizer.get_feature_names_out()
x_train = pd.DataFrame(training_bag_of_words.toarray(),
                      columns=feature_names).astype(int)
# x_train = pd.DataFrame(training_bag_of_words.toarray(),
#                        columns=[x for x in desc_vectorizer.get_feature_names()]).astype(int)

x_train.head()


# In[14]:


test_bag_of_words = desc_vectorizer.transform(raw_test["Expense"])

feature_names = desc_vectorizer.get_feature_names_out()
x_test = pd.DataFrame(test_bag_of_words.toarray(),
                      columns=feature_names).astype(int)
x_test.head()


# In[15]:


print(f"Shape of x_train: {x_train.shape}")
print(f"Shape of x_test: {x_test.shape}")


# In[36]:


# !pip freeze
import numpy
print(numpy.__version__)


# ## Model building

# ### Neural Network

# In[16]:


import tensorflow as tf
from tensorflow import keras

model = keras.Sequential([
    keras.layers.Input(shape=(100,)),
    keras.layers.Dense(10, activation='relu'),
    keras.layers.Dense(len(set(df["Category"])) + 1, activation='softmax') # extra unit for "UNKNOWN" tag
])


# In[17]:


model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])


# In[18]:


model.fit(x_train.values, raw_train["Category (encoded)"], epochs=100)


# ### Random Forest

# In[19]:


from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier()
rf.fit(x_train, raw_train["Category (encoded)"])


# ### XGBoost

# In[20]:


from xgboost import XGBClassifier

xgb = XGBClassifier()
xgb.fit(x_train, raw_train["Category (encoded)"])


# ### Elasticsearch

# ##### Upload training data (descriptions + category) to elasticsearch

# In[21]:


from elasticsearch_simple_client.uploader import Uploader

es_uploader = Uploader()
df_tmp = raw_train[["Expense","Category"]]

es_uploader.post_df(df=df_tmp)


# #### Query test data and lookup nearest (fuzzy) match to training data and get corresponding category

# In[22]:


import time
time.sleep(5) # allow time for elasticsearch indices to be updated


# In[23]:


from elasticsearch_simple_client.searcher import Searcher

searcher = Searcher()
es_category_lookup_on_train_data = []

for entry in raw_train["Expense"]:
    result = searcher.execute_search(field="Expense",
                                     shoulds=[entry])["hits"]["hits"]
    predicted_category = result[0]["_source"]["Category"]
    es_category_lookup_on_train_data.append(predicted_category)


# In[24]:


es_category_lookup_on_test_data = []

for entry in raw_test["description"]:
    result = searcher.execute_search(field="description",
                                     shoulds=[entry])["hits"]["hits"]
    if len(result) > 0:
        es_category_lookup_on_test_data.append(result[0]["_source"]["Category"])
    else:
        es_category_lookup_on_test_data.append("UNKNOWN")


# ## Analysis

# In[25]:


from sklearn.metrics import accuracy_score, balanced_accuracy_score


# In[26]:


def analysis_result(model_name, 
                    train_prediction,
                    train_target,
                    test_prediction, 
                    test_target):
    results = dict()
    results["model"] = model_name
    results["train_accuracy"] = accuracy_score(train_prediction, train_target)
    results["balanced_train_accuracy"] = balanced_accuracy_score(train_prediction, train_target)
    results["test_accuracy"] = accuracy_score(test_prediction, test_target)
    results["balanced_test_accuracy"] = balanced_accuracy_score(test_prediction, test_target)

    for key, value in results.items():
        if isinstance(value, str):
            print(f"\n{value}")
        else:
            print(f"\t{key}: {'{:.2f}'.format(value)}")
    
    return results


# In[27]:


accumulated_results = []

train_pred_nn = [list(x).index(max(x)) for x in model.predict(x_train)]
test_pred_nn = [list(x).index(max(x)) for x in model.predict(x_test)]

accumulated_results.append(analysis_result("Neural Network", 
                                           train_pred_nn, 
                                           raw_train['Category (encoded)'],               
                                           test_pred_nn, 
                                           raw_test['Category (encoded)']))

train_pred_rf = rf.predict(x_train)
test_pred_rf = rf.predict(x_test)

accumulated_results.append(analysis_result("Random Forest", 
                                           train_pred_rf, 
                                           raw_train['Category (encoded)'],
                                           test_pred_rf, 
                                           raw_test['Category (encoded)']))

train_pred_xgb = xgb.predict(x_train)
test_pred_xgb = xgb.predict(x_test)

accumulated_results.append(analysis_result("XGBoost", 
                                           train_pred_xgb, 
                                           raw_train['Category (encoded)'],
                                           test_pred_xgb, 
                                           raw_test['Category (encoded)']))

# train_pred_es = le.transform(es_category_lookup_on_train_data)
# test_pred_es = le.transform(es_category_lookup_on_test_data)

# accumulated_results.append(analysis_result("Elasticsearch", 
#                                            train_pred_es, 
#                                            raw_train['Category (encoded)'],
#                                            test_pred_es, 
#                                            raw_test['Category (encoded)']))


# In[28]:


model_results_df = pd.DataFrame(accumulated_results)
model_results_df.set_index("model").plot(
    kind="bar", figsize=(13,7), ylim=(0.5,1), fontsize=15, 
    title="Performance of models on predicting train and test data"
)


# In[29]:


raw_train.loc[:,"nn_prediction"] = le.inverse_transform(train_pred_nn)
raw_test.loc[:,"nn_prediction"] = le.inverse_transform(test_pred_nn)

raw_train.loc[:,"rf_prediction"] = le.inverse_transform(train_pred_rf)
raw_test.loc[:,"rf_prediction"] = le.inverse_transform(test_pred_rf)

raw_train.loc[:,"xgb_prediction"] = le.inverse_transform(train_pred_xgb)
raw_test.loc[:,"xgb_prediction"] = le.inverse_transform(test_pred_xgb)

# raw_train.loc[:,"es_prediction"] = es_category_lookup_on_train_data
# raw_test.loc[:,"es_prediction"] = es_category_lookup_on_test_data


# In[30]:


view_columns = [x for x in raw_train.keys() if x != "Category (encoded)"]


# In[31]:


raw_train[view_columns].head(10)


# In[ ]:


raw_test[view_columns].head(10)


# In[ ]:


raw_train[view_columns].sample(n=20)


# In[ ]:


raw_test[view_columns].sample(n=20)


# ### Run everything above again but with Kfold cross validation for better assessment of accuracy

# #### Split dataset into 10 fold

# In[ ]:


from sklearn.model_selection import KFold

def split_dataset(x_data,y_data,n_splits):
    # shuffle true is important as the dataset is sorted by alphabetical order of the description column
    # if shuffle is false then the model will be trained on data very different to that of test!!!
    for train_index, test_index in KFold(n_splits, shuffle=True).split(x_data):
        x_train, x_test = x_data[train_index], x_data[test_index]
        y_train, y_test = y_data[train_index], y_data[test_index]
        yield x_train,y_train,x_test,y_test
        
cv_dataset = split_dataset(df[["Expense"]].values,
                           df["Category (encoded)"].values,
                           10)


# - The following is bad coding but i couldn't be bothered to refactor this!!!
# - That said, most of this is a copy (with modification) of the above but over a loop of all the kfold splits

# In[ ]:


get_ipython().run_cell_magic('time', '', '\noverall_results = []\n\nfor n, (x_train, y_train, x_test, y_test) in enumerate(cv_dataset):\n    print(f"\\nProcessing run {n+1}\\n")\n    run_result = dict() # to store results\n    \n    # create count vectorizer and apply to train data\n    run_result["desc_vectorizer"] = CountVectorizer(analyzer="word", max_features=100)\n    training_bag_of_words = run_result["desc_vectorizer"].fit_transform([x[0] for x in x_train])\n    feature_names = desc_vectorizer.get_feature_names_out()\n    x_train_count_vectorised = pd.DataFrame(\n        training_bag_of_words.toarray(),\n        columns=feature_names).astype(int)\n    \n    # apply count vectorizer to test data\n    test_bag_of_words = run_result["desc_vectorizer"].transform([x[0] for x in x_test])\n    feature_names = desc_vectorizer.get_feature_names_out()\n    x_test_count_vectorised = pd.DataFrame(\n        test_bag_of_words.toarray(),\n        columns=feature_names).astype(int)\n    \n    # train neural network model\n    run_result["nn_model"] = keras.Sequential([\n        keras.layers.Input(shape=(100,)),\n        keras.layers.Dense(10, activation=\'relu\'),\n        keras.layers.Dense(len(set(df["Category"])) + 1, \n                           activation=\'softmax\') # extra unit for "UNKNOWN" tag\n    ])\n    \n    run_result["nn_model"].compile(optimizer=\'adam\',\n                     loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),\n                     metrics=[\'accuracy\'])\n    run_result["nn_model"].fit(x_train_count_vectorised.values, y_train, epochs=100, verbose=0)\n    \n    # train random forest model\n    run_result["rf"] = RandomForestClassifier()\n    run_result["rf"].fit(x_train_count_vectorised, y_train)\n\n    # train xgboost model\n    run_result["xgb"] = XGBClassifier()\n    run_result["xgb"].fit(x_train_count_vectorised, y_train)\n    \n    # reconstruct df for elasticsearch input\n    # df_train_for_es = pd.concat(\n    #     [pd.DataFrame(x_train, columns=["description"]), \n    #      pd.Series(y_train, name="Category_encoded")], \n    #     axis=1\n    # )\n    \n    # # clear elastic search and upload new training data\n    # !curl -X DELETE "localhost:9200/simple_text"\n    # es_uploader = Uploader()\n    # es_uploader.post_df(df_train_for_es)\n    \n    # time.sleep(5) # allow time for elasticsearch indices to be updated\n    \n    # # lookup train and test entries from elasticsearch (this time categories are encoded)\n    # searcher = Searcher()\n    # run_result["es_category_lookup_on_train_data"] = []\n\n    # for entry in df_train_for_es["description"]:\n    #     result = searcher.execute_search(field="description",\n    #                                      shoulds=[entry])["hits"]["hits"]\n    #     predicted_category = result[0]["_source"]["Category_encoded"]\n    #     run_result["es_category_lookup_on_train_data"].append(predicted_category)\n    \n    # run_result["es_category_lookup_on_test_data"] = []\n\n    # for entry in [x[0] for x in x_test]:\n    #     result = searcher.execute_search(field="description",\n    #                                      shoulds=[entry])["hits"]["hits"]\n    #     if len(result) > 0:\n    #         run_result["es_category_lookup_on_test_data"].append(\n    #             result[0]["_source"]["Category_encoded"]\n        #     )\n        # else:\n        #     run_result["es_category_lookup_on_test_data"].append(le.transform(["UNKNOWN"]))\n    \n    # analyse\n    accumulated_results = []\n\n    train_pred_nn = [list(x).index(max(x)) for x in run_result["nn_model"].predict(x_train_count_vectorised)]\n    test_pred_nn = [list(x).index(max(x)) for x in run_result["nn_model"].predict(x_test_count_vectorised)]\n\n    accumulated_results.append(analysis_result("Neural Network", \n                                               train_pred_nn, \n                                               y_train,\n                                               test_pred_nn, \n                                               y_test))\n\n    train_pred_rf = run_result["rf"].predict(x_train_count_vectorised)\n    test_pred_rf = run_result["rf"].predict(x_test_count_vectorised)\n\n    accumulated_results.append(analysis_result("Random Forest", \n                                               train_pred_rf, \n                                               y_train,\n                                               test_pred_rf, \n                                               y_test))\n\n    train_pred_xgb = run_result["xgb"].predict(x_train_count_vectorised)\n    test_pred_xgb = run_result["xgb"].predict(x_test_count_vectorised)\n\n    accumulated_results.append(analysis_result("XGBoost", \n                                               train_pred_xgb, \n                                               y_train,\n                                               test_pred_xgb, \n                                               y_test))\n\n    # train_pred_es = [int(x) for x in run_result["es_category_lookup_on_train_data"]]\n    # test_pred_es = [int(x) for x in run_result["es_category_lookup_on_test_data"]]\n\n    # accumulated_results.append(analysis_result("Elasticsearch", \n    #                                            train_pred_es, \n    #                                            y_train,\n    #                                            test_pred_es, \n    #                                            y_test))\n    \n    # reconstruct df for overview later\n    df_train = pd.concat(\n        [pd.DataFrame(x_train, columns=["Expense"]), \n         pd.Series(le.inverse_transform(y_train), name="Category")], \n        axis=1\n    )\n\n    df_test = pd.concat(\n        [pd.DataFrame(x_test, columns=["Expense"]), \n         pd.Series(le.inverse_transform(y_test), name="Category")], \n        axis=1\n    )\n\n    \n    df_train.loc[:,"nn_prediction"] = le.inverse_transform(train_pred_nn)\n    df_test.loc[:,"nn_prediction"] = le.inverse_transform(test_pred_nn)\n\n    df_train.loc[:,"rf_prediction"] = le.inverse_transform(train_pred_rf)\n    df_test.loc[:,"rf_prediction"] = le.inverse_transform(test_pred_rf)\n\n    df_train.loc[:,"xgb_prediction"] = le.inverse_transform(train_pred_xgb)\n    df_test.loc[:,"xgb_prediction"] = le.inverse_transform(test_pred_xgb)\n\n    # df_train.loc[:,"es_prediction"] = le.inverse_transform(train_pred_es)\n    # df_test.loc[:,"es_prediction"] = le.inverse_transform(test_pred_es)\n    \n    # save results\n    run_result["accumulated_results"] = accumulated_results\n    run_result["df_train"] = df_train\n    run_result["df_test"] = df_test\n    \n    overall_results.append(run_result)\n')


# ####  Plot KFold cv model results

# In[ ]:


data_for_plotting = dict()
# models = ["Neural Network", "Random Forest", "XGBoost", "Elasticsearch"]
# metrics = ["train_accuracy","balanced_train_accuracy","test_accuracy","balanced_test_accuracy"]
models = ["Neural Network", "Random Forest", "XGBoost"]
metrics = ["train_accuracy","balanced_train_accuracy","test_accuracy","balanced_test_accuracy"]
for metric in metrics:
    data_for_plotting[metric] = dict()
    for model in models:
        data_for_plotting[metric][model] = []
        for result in overall_results:
            for entry in result["accumulated_results"]:
                if entry["model"] == model:
                    data_for_plotting[metric][model].append(entry[metric])
print(metrics)


# In[ ]:


for metric in data_for_plotting:
    labels = []
    data = []
    for model, values in data_for_plotting[metric].items():
        labels.append(model)
        data.append(values)
    fig, ax = plt.subplots(figsize=(10,7))
    ax.set_title(f'{metric} (KFold of 10)')
    ax.set_ylim([0, 1])
    ax.set_ylabel("accuracy")
    bp = ax.boxplot(data, labels=labels)
    for box in bp['boxes']:
        # change outline color
        box.set( color='limegreen', linewidth=2)

    ## change color and linewidth of the whiskers
    for whisker in bp['whiskers']:
        whisker.set(color='limegreen', linewidth=2)

    ## change color and linewidth of the caps
    for cap in bp['caps']:
        cap.set(color='limegreen', linewidth=2)

    ## change color and linewidth of the medians
    for median in bp['medians']:
        median.set(color='w', linewidth=2)

    ## change the style of fliers and their fill
    for flier in bp['fliers']:
        flier.set(marker='o', color='limegreen', alpha=0.5)

        
        
        
        
    plt.show()


# In[ ]:





# In[ ]:





# In[ ]:




