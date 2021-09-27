import pandas as pd
enguins = pd.read_csv('https://github.com/SibitendaHarriet/PythonPackageStructure/blob/main/data/processed_telecom.csv')

# Ordinal feature encoding
# https://www.kaggle.com/pratik1120/penguin-dataset-eda-classification-and-clustering
df = enguins.copy()
target = 'DecileRank'
encode = ['MSISDN_Number','No_of_xDRsessions','Session_Duration_s','Total_MB','Social_Media_MB','Google_MB','Email_MB','Youtube_MB','Netflix_MB', 'Gaming_MB', 'Other_MB']

for col in encode:
    dummy = pd.get_dummies(df[col], prefix=col)
    df = pd.concat([df,dummy], axis=1)
    del df[col]

target_mapper = {'neutral':0, 'bad':1, 'fair':2, 'good':3, 'very_good':4}
def target_encode(val):
    return target_mapper[val]

df['DecileRank']= df['DecileRank'].apply(target_encode)

# Separating X and y
X = df.drop('DecileRank', axis=1)
Y = df['DecileRank']

# Build random forest model
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier()
clf.fit(X, Y)

# Saving the model
import pickle
pickle.dump(clf, open('enguins_clf.pkl', 'wb'))
