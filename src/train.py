#!/usr/bin/env python
# coding: utf-8

# # Startup Success Probability 
# ```
# Determining the success probabilty of a startup project using determining factors such as target amount, backer count, project member count, and etc. This is to help startup teams to reassess their strategies especially in crowdfunding as well as take care of the important variables that may define the project's success.
# ```

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, recall_score, log_loss, precision_score
from scipy.stats.mstats import winsorize
from xgboost import XGBClassifier
import xgboost as xgb
from imblearn.over_sampling import SMOTE
import pickle

df = pd.read_csv('../turkish_crowdfunding_startups/turkishCF.csv', delimiter=';')

df.columns = ['id', 'platform_name', 'mass_funding_type', 'project_category', 'funding_method',
       'project_name', 'project_owner', 'project_owner_gender',
       'project_supported', 'subscribed_projects', 'number_of_projects_owned',
       'number_of_teams', 'owner_location', 'project_region', 'launch_year', 'project_start_date',
       'project_end_date', 'project_duration', 'promo_video', 'promo_video_length',
       'image_count', 'faq', 'updates', 'comments', 'backer_count',
       'reward_count', 'project_member_count', 'website', 'social_media',
       'social_media_count', 'social_media_followers', 'total_tags', 'project_description_length',
       'project_description', 'target_amount', 'collected_amount', 'target_amount_percentage',
       'success_status']

### Data Preparation
for cols in df.select_dtypes('object').columns:
    df[cols] = df[cols].str.lower()

binary_cols = ['promo_video', 'website', 'social_media']

for col in binary_cols:
    df[col] = df[col].map({'var':1, 'yok':0})

# funding method
df['funding_method'] = df['funding_method'].map({'ya hep ya hiç': 'all-or-nothing', 'hepsi kalsın': 'keep-it-all'})

# funding type
df['mass_funding_type'] = df['mass_funding_type'].map({'ödül': 'prize', 'bağış': 'donation'})

# success status
df['success_status'] = df['success_status'].map({'başarısız': 0, 'başarılı': 1})

# whether the project has FAQ
df['faq'] = np.where(df['faq'] > 0, 1, 0)

# project category
df['project_category'] = df['project_category'].map({
    'film-video-fotoğraf': 'film-video-photography',
    'teknoloji': 'technology',
    'kültür-sanat': 'culture-art',
    'eğitim': 'education',
    'diğer': 'other',
    'çevre': 'environment',
    'müzik': 'music',
    'sağlık-güzellik': 'health-beauty',
    'tasarım': 'design',
    'yayıncılık': 'publishing',
    'gıda-yeme-içme': 'food-eating-drinking',
    'spor': 'sports',
    'hayvanlar': 'animals',
    'moda': 'fashion',
    'sosyal sorumluluk': 'social_responsibility',
    'dans-performans': 'dance-performance',
    'turizm': 'tourism',
})

df['target_amount_percentage'] = df['target_amount_percentage'].str.rstrip('%').astype(float)
df['target_amount_percentage'] = df['target_amount_percentage'] / 100

del df['project_description_length']
del df['project_description']
del df['id']


index_to_drop = df[pd.isnull(df.project_region)].index

df.drop(index_to_drop, inplace=True)
pd.isnull(df.project_region).any()


target_value = df['success_status'].values

df_full_train, df_test, y_full_train, y_test = train_test_split(df, target_value, test_size=0.2, random_state=42)

df_full_train.reset_index(inplace=True, drop=True)

winsorized_data_full = winsorize(df_full_train['backer_count'], limits=[0.05, 0.05])
winsorized_data_test = winsorize(df_test['backer_count'], limits=[0.05, 0.05])

df_full_train['log_backer_count'] = np.log1p(winsorized_data_full)
df_test['log_backer_count'] = np.log1p(winsorized_data_test)


# ## Model Selection
df_train, df_val, y_train, y_val = train_test_split(df_full_train, y_full_train, test_size=len(df_test), random_state=42)

del df_train['success_status']
del df_val['success_status']
del df_test['success_status']

df_train.reset_index(inplace=True, drop=True)
df_val.reset_index(inplace=True, drop=True)
df_test.reset_index(inplace=True, drop=True)


# ##### Dropping columns after EDA
# these are columns that did not affect the target variable or has been opted out due to the creator's decision
to_drop = ['project_end_date', 'project_start_date', 'launch_year', 'project_owner', 
    'owner_location', 'project_name', 'collected_amount', 'project_owner_gender', 
    'subscribed_projects', 'target_amount_percentage', 'backer_count', 'success_status',
    'project_region', 'platform_name',
]
# just to experiment, I am dropping the target_amount_percentage that contributes so much to the result

df_full_train.drop(columns=to_drop, inplace=True, errors='ignore')
df_test.drop(columns=to_drop, inplace=True, errors='ignore')

print(f'Training the model...')

dv = DictVectorizer(sparse=False)

# convert to dictionary before transforming or DictVectorizer
full_train_dicts = df_full_train.to_dict(orient='records')
test_dicts = df_test.to_dict(orient='records')

X_full = dv.fit_transform(full_train_dicts)
X_test = dv.transform(test_dicts)


# handle imbalance by using SMOTE
smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X_full, y_full_train)

# # create model instance
# model = XGBClassifier(objective='binary:logistic', random_state=42)
# fit model

early_stop = xgb.callback.EarlyStopping(
    rounds=8, metric_name='logloss', data_name='validation_0', save_best=True
)

model = XGBClassifier (
    objective='binary:logistic', 
    random_state=42,
    n_estimators=300,
    learning_rate=0.5263,
    tree_method='hist',
    callbacks=[early_stop],
    max_depth=2
)

model.fit(X_res, y_res, eval_set=[(X_test, y_test)]) # use the balanced dataset

y_preds_proba = model.predict_proba(X_test)
y_preds = model.predict(X_test)

accuracy = np.round(accuracy_score(y_test, y_preds), 3)
recall = np.round(recall_score(y_test, y_preds), 3)
f1score = np.round(f1_score(y_test, y_preds), 3)
precision = np.round(precision_score(y_test, y_preds), 3)
auc_score = np.round(roc_auc_score(y_test, y_preds_proba[:,1]), 3)
log_ls = np.round(log_loss(y_test, y_preds_proba), 3)

print(f'Model: {model.__class__.__name__}')
print(f'Accuracy Score: {accuracy}')
print(f'Recall Score: {recall}')
print(f'f1 Score: {f1score}')
print(f'Precision Score: {precision}')
print(f'ROC_AUC Score: {auc_score}')
print(f'Log Loss: {log_ls}')

# ### Saving with Pickle
output_file=f'startup-success-predictor.bin'

# f_out = open(output_file, 'wb')
with open(output_file, 'wb') as f_out:
    pickle.dump((model, dv), f_out)

print(f'Trained model saved to {output_file}')