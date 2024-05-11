from mlxtend.frequent_patterns import apriori, association_rules
from ucimlrepo import fetch_ucirepo
import pandas as pd
import numpy as np
from mlxtend.preprocessing import TransactionEncoder
from sklearn.preprocessing import KBinsDiscretizer
import warnings
from ucimlrepo import fetch_ucirepo
from datasketch import MinHashLSH, MinHash

health_data = fetch_ucirepo(id=519)
X= health_data.data.features
y = health_data.data.targets
X.to_csv("olddataset_features.csv")
y.to_csv("targetsofdata.csv")
if isinstance(X, pd.DataFrame):
    print("We have found a dataframe")
else:
    "Dataframe not found"
print("=====================================")
print(health_data.data.features)
print("=====================================")
print(health_data.data.targets)
print("====================================")
print(health_data.variables)
print("=====================================")

print("Check for missing values")
print("=====================================")
missing_values= X.isna().sum()

if missing_values.any():
    print("no missing value go ahead")

max_value= X['time'].max()

def categorize_age(age):
    if age<30:
        return 'young'
    elif 30<=age<=50:
        return 'middleaged'
    elif 50<age<100:
        return 'old'
X['age']=X['age'].apply(categorize_age)

print("=================================================")
def categorize_anemia(anameia):
    if  anameia==1:
        return('yes')
    else:
        return('no')
X['anaemia']=X['anaemia'].apply(categorize_anemia)

def categorize_diabetes(diabetes):
    if diabetes==1:
        return('yes')
    else:
        return('no')

X['diabetes']=X['diabetes'].apply(categorize_diabetes)
def categorize_bloodpressure(pressure):
    if pressure==1:
        return('yes')
    else:
        return('no')
X['high_blood_pressure']=X['high_blood_pressure'].apply(categorize_bloodpressure)
def categorize_gender(gender):
    if gender==1:
        return('male')
    else:
        return('female')
X['sex']=X['sex'].apply(categorize_gender)

def catergorize_smoking(smoking):
    if  smoking==1:
        return('smoker')
    else:
        return('nonsmoker')
X['smoking']=X['smoking'].apply(catergorize_smoking)
print("================Lets Categorize Non Binary Variables Now=============================================")
quartilies_cret=np.percentile(X['creatinine_phosphokinase'],[33,66])
low_range_cret = 0 
medium_cret= quartilies_cret[0]
high_cret = quartilies_cret[1]
def catergorize_cretinine(cret):
    if low_range_cret<=cret<medium_cret:
        return('low')
    elif medium_cret<=cret<high_cret:
        return('medium')
    else:
        return('high')
X['creatinine_phosphokinase']=X['creatinine_phosphokinase'].apply(catergorize_cretinine)
quartilies_eject=np.percentile(X['ejection_fraction'],[33,66])
low_range_eject=0
medium_eject=quartilies_eject[0]
high_eject=quartilies_eject[1]
def categorize_eject(eject):
    if low_range_eject<=eject<medium_eject:
        return('low')
    elif medium_eject<=eject<high_eject:
        return('medium')
    else:
        return('high')
X['ejection_fraction']=X['ejection_fraction'].apply(categorize_eject)
quartiles_platelets = np.percentile(X['platelets'], [33, 66])  
low_range_platelets = 0
medium_range_platelets = quartiles_platelets[0]
high_range_platelets = quartiles_platelets[1]

def categorize_platelets(platelets):
    if platelets <= medium_range_platelets:
        return 'low'
    elif medium_range_platelets < platelets <= high_range_platelets:
        return 'medium'
    else:
        return 'high'

X['platelets'] = X['platelets'].apply(categorize_platelets)
# Calculate quartiles for the 'serum_creatinine' column
quartiles_serum_creatinine = np.percentile(X['serum_creatinine'], [33.33, 66.66])  # Dividing into three equal parts

# Define ranges based on quartiles
low_range_serum_creatinine = 0
medium_range_serum_creatinine = quartiles_serum_creatinine[0]
high_range_serum_creatinine = quartiles_serum_creatinine[1]

# Function to categorize serum_creatinine
def categorize_serum_creatinine(serum_creatinine):
    if serum_creatinine <= medium_range_serum_creatinine:
        return 'low'
    elif medium_range_serum_creatinine < serum_creatinine <= high_range_serum_creatinine:
        return 'medium'
    else:
        return 'high'

# Apply categorization to the data
X['serum_creatinine'] = X['serum_creatinine'].apply(categorize_serum_creatinine)

quartiles_serum_sodium = np.percentile(X['serum_sodium'], [33.33, 66.66])  # Dividing into three equal parts

low_range_serum_sodium = 0
medium_range_serum_sodium = quartiles_serum_sodium[0]
high_range_serum_sodium = quartiles_serum_sodium[1]

def categorize_serum_sodium(serum_sodium):
    if serum_sodium <= medium_range_serum_sodium:
        return 'low'
    elif medium_range_serum_sodium < serum_sodium <= high_range_serum_sodium:
        return 'medium'
    else:
        return 'high'

# Apply categorization to the data
X['serum_sodium'] = X['serum_sodium'].apply(categorize_serum_sodium)


# Calculate quartiles for the 'serum_sodium' column
quartiles_time = np.percentile(X['time'], [33.33, 66.66])  # Dividing into three equal parts

# Define ranges based on quartiles
low_range_time = 0
medium_range_time = quartiles_time[0]
high_range_time = quartiles_time[1]

# Function to categorize serum_sodium
def categorize_time(time):
    if time <= medium_range_time:
        return 'firstquarter'
    elif medium_range_time < time <= high_range_time:
        return 'secondquarter'
    else:
        return 'thirdquarter'

# Apply categorization to the data
X['time'] = X['time'].apply(categorize_time)
print(X)

X.to_csv("dataset.csv")

'''print((X['time'].min()))
print(max_value)
print(y['death_event'].get(1))
valuecount = (y['death_event']==1).sum()
print(valuecount)
'''
# Perform one-hot encoding
X_encoded = pd.get_dummies(X)
X_encoded.to_csv("encodeddata.csv")
# Display the encoded DataFrame
print(X_encoded.head())
print((X['age']=='middleaged').sum())
 
frequent_items= apriori(X_encoded,min_support = 0.2,use_colnames=True)

print(frequent_items)
# Generating association rules
rules = association_rules(frequent_items, metric="confidence", min_threshold=0.6)

# Display the association rules
print(rules)
rules.to_csv('rules.csv')
from datasketch import MinHash, MinHashLSH

# Initialize MinHashLSH
lsh = MinHashLSH(threshold=0.6, num_perm=128)

# Construct MinHash signatures and insert into MinHashLSH
for idx, rows in X_encoded.iterrows():
    mh = MinHash(num_perm=128)
    for col, value in rows.items():
        if value:
            mh.update(col.encode('utf8'))
    lsh.insert(str(idx), mh)

# Group patients based on similarities
patient_groups = {}
for i, patient_row in X_encoded.iterrows():
    patient_id = str(i)
    query_mh = MinHash(num_perm=128)
    for col, value in patient_row.items():
        if value:
            query_mh.update(col.encode('utf8'))
    similar_patients = lsh.query(query_mh)
    
    # Add patient to group
    if patient_id not in patient_groups:
        patient_groups[patient_id] = set()
    patient_groups[patient_id].add(patient_id)
    
    # Add similar patients to the same group
    for similar_patient_id in similar_patients:
        patient_groups[patient_id].add(similar_patient_id)

# Convert patient_groups to a list of sets to remove duplicate groups
patient_groups = [set(group) for group in patient_groups.values()]

# Print patient groups
for i, group in enumerate(patient_groups):
    print(f"Group {i+1}:")
    for patient_id in group:
        print("Patient ID:", patient_id)
    print("-" * 20)