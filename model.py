import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import ensemble
from sklearn import model_selection
from sklearn import metrics
from sklearn import neighbors
from sklearn import tree

from sklearn.metrics import confusion_matrix, roc_auc_score, accuracy_score, f1_score, recall_score, precision_score
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_validate

# Load the csv file
train_data = pd.read_csv('train.csv')
test_data  = pd.read_csv('test.csv')

train_data = train_data.drop(columns=['ID', 'Name', 'SSN'])
test_data = test_data.drop(columns=['ID', 'Name', 'SSN'])

int_fixed = ['Age', 'Num_Bank_Accounts', 'Num_Credit_Card', 'Interest_Rate', 'Num_of_Loan']
int_var = ['Delay_from_due_date', 'Num_of_Delayed_Payment']

def convert_to_int(string):
    try:
        return int(str(string).replace('_', ''))
    except:
        return 'nan'

for data in [train_data, test_data]:
    int_col_with_str = int_fixed + int_var
    for int_col in int_col_with_str:
        data[int_col] = data[int_col].apply(lambda x: convert_to_int(x))

float_fixed = ['Annual_Income', 'Monthly_Inhand_Salary', 'Num_Credit_Inquiries', 'Outstanding_Debt', 'Total_EMI_per_month']
float_var = ['Credit_Utilization_Ratio', 'Amount_invested_monthly', 'Monthly_Balance', 'Changed_Credit_Limit']

def convert_to_float(string):
    try:
        return float(str(string).replace('_', ''))
    except:
        return 'nan'
    
# удалим '_' чтобы конвертировать в float
for data in [train_data, test_data]:
    float_col_with_str = float_fixed + float_var
    for float_col in float_col_with_str:
        data[float_col] = data[float_col].apply(lambda x: convert_to_float(x))

def convert_credit_history(credit_history_age):
    if str(credit_history_age) == 'nan':
        return 'nan'
    else:
        years = int(credit_history_age.split(' ')[0])
        months = int(credit_history_age.split(' ')[3])
        return 12 * years + months

def convert_payment_behaviour(behaviour, split_name):
    try:
        if split_name == 'spent':
            return behaviour.split('_')[0]
        elif split_name == 'value':
            return behaviour.split('_')[2]
        else:
            return 'nan'
    except:
        return 'nan'

def convert_type_of_loan(original_text, loan_type):
    if original_text == '':
        return 'nan'
    
    try:
        loans = original_text.split(', ')
        if loan_type in loans:
            return 1
        else:
            return 0
    except:
        return 'nan'

merged_data = pd.concat([train_data, test_data])
loan_type_column = merged_data['Type_of_Loan']

loan_type_all = []
for i in range(len(merged_data)):
    try:
        loan_types = loan_type_column.iloc[i].split(', ')
        for loan_type in loan_types:
            if len(loan_type) >= 5 and loan_type[:4] == 'and ':
                loan_type_all.append(loan_type[4:])
            else:
                loan_type_all.append(loan_type)
    except:
        pass

loan_type_all = list(set(loan_type_all) - set(['Not Specified']))

for data in [train_data, test_data]:
    data['Credit_History_Age'] = data['Credit_History_Age'].apply(
        lambda x: convert_credit_history(x)
    )
    data['Payment_Behaviour_Spent'] = data['Payment_Behaviour'].apply(
        lambda x: convert_payment_behaviour(x, 'spent')
    )
    data['Payment_Behaviour_Value'] = data['Payment_Behaviour'].apply(
        lambda x: convert_payment_behaviour(x, 'value')
    )
    
    for loan_type in loan_type_all:
        data['Loan_Type_' + loan_type.replace(' ', '_')] = data['Type_of_Loan'].apply(
            lambda x: convert_type_of_loan(x, loan_type)
        )

train_data = train_data.drop(columns=['Payment_Behaviour', 'Type_of_Loan'])
test_data  = test_data .drop(columns=['Payment_Behaviour', 'Type_of_Loan'])

def map_month(month_str):
    months = ['January', 'February', 'March', 'April', 'May', 'June',
                'July', 'August', 'September', 'October', 'November', 'December']
    return months.index(month_str) / 11
    
for data in [train_data, test_data]:
    data['Month'] = data['Month'].apply(lambda x: map_month(x))

fixed_numeric_columns = int_fixed + float_fixed

nRowsTrain = len(train_data)
nRowsTest = len(test_data)

for col in fixed_numeric_columns:
    print('current processing column : ' + col)
    
    for i in range(nRowsTrain // 8):
        column = train_data.loc[train_data['Customer_ID'] == train_data['Customer_ID'].iloc[i * 8]][col]
        most_frequent_values = column.dropna().mode()
        
        if len(most_frequent_values) > 0:
            train_data[col].iloc[8 * i : 8 * (i + 1)] = most_frequent_values[0]
        
    for i in range(nRowsTest // 4):
        column = test_data.loc[test_data['Customer_ID'] == test_data['Customer_ID'].iloc[i * 4]][col]
        most_frequent_values = column.dropna().mode()
        
        if len(most_frequent_values) > 0:
            test_data[col].iloc[4 * i : 4 * (i + 1)] = most_frequent_values[0]

for col in ['Monthly_Inhand_Salary', 'Num_Credit_Inquiries', 'Amount_invested_monthly', 'Monthly_Balance']:
    
    train_data[col] = train_data[col].apply(
        lambda x: x if (pd.notnull(x) and convert_to_float(x) != 'nan') else 'NaN_float'
    )
    test_data [col] = test_data [col].apply(
        lambda x: x if (pd.notnull(x) and convert_to_float(x) != 'nan') else 'NaN_float'
    )

null_values = {
    'Occupation': '_______',
    'Monthly_Inhand_Salary': 'NaN_float', 'Num_Credit_Inquiries': 'NaN_float', 'Amount_invested_monthly': 'NaN_float', 'Monthly_Balance': 'NaN_float',
    'Loan_Type_Mortgage_Loan': 'nan', 'Loan_Type_Auto_Loan': 'nan', 'Loan_Type_Student_Loan': 'nan', 'Loan_Type_Payday_Loan': 'nan',
    'Loan_Type_Debt_Consolidation_Loan': 'nan', 'Loan_Type_Home_Equity_Loan': 'nan', 'Loan_Type_Personal_Loan': 'nan', 'Loan_Type_Credit-Builder_Loan': 'nan',
    'Num_of_Delayed_Payment': 'nan', 'Credit_History_Age': 'nan', 'Changed_Credit_Limit': 'nan', 'Payment_Behaviour_Value': 'nan',
    'Credit_Mix': '_', 'Payment_Behaviour_Spent': '!@9#%8'
}

for null_value_col in [
    'Occupation', 'Monthly_Inhand_Salary', 'Num_Credit_Inquiries', 'Credit_Mix',
    'Loan_Type_Mortgage_Loan', 'Loan_Type_Auto_Loan', 'Loan_Type_Student_Loan', 'Loan_Type_Payday_Loan',
    'Loan_Type_Debt_Consolidation_Loan', 'Loan_Type_Home_Equity_Loan', 'Loan_Type_Personal_Loan', 'Loan_Type_Credit-Builder_Loan'
]:
    print('current processing column : ' + null_value_col)
    
    for i in range(nRowsTrain // 8):
        column = train_data.loc[train_data['Customer_ID'] == train_data['Customer_ID'].iloc[i * 8]][null_value_col]
        mode_values = column.loc[column != null_values[null_value_col]].mode()
        
        if len(mode_values) > 0:
            most_frequent = mode_values[0]
            train_data[null_value_col].iloc[8 * i : 8 * (i + 1)] = most_frequent
        
    for i in range(nRowsTest // 4):
        column = test_data.loc[test_data['Customer_ID'] == test_data['Customer_ID'].iloc[i * 4]][null_value_col]
        mode_values = column.loc[column != null_values[null_value_col]].mode()
        
        if len(mode_values) > 0:
            most_frequent = mode_values[0]
            test_data[null_value_col].iloc[4 * i : 4 * (i + 1)] = most_frequent

train_data = train_data.drop(columns = ['Customer_ID'])
test_data  = test_data .drop(columns = ['Customer_ID'])

def replace_with_median(value, idx, data_arr, rows_per_customer, null_value, is_round = False):
    
    if value != null_value:
        return value
    
    start_idx = (idx // rows_per_customer) * rows_per_customer
    end_idx = (idx // rows_per_customer + 1) * rows_per_customer
    data_range = data_arr[start_idx:end_idx]
    
    values = []
    fraction = -1
    for data_value in data_range:
        if data_value != null_value:
            values.append(float(data_value))
            fraction = float(data_value) % 1.0

    if len(values) == 0:
        return null_value
    else:
        result = np.median(values)
        
        if is_round:
            return result if abs(result % 1.0 - fraction) < 0.25 else result + 0.5
        else:
            return result

for null_value_col in [
    'Amount_invested_monthly', 'Num_of_Delayed_Payment', 'Changed_Credit_Limit', 'Monthly_Balance'
]:
    print('current processing column : ' + null_value_col)
    
    train_data_np = []
    test_data_np  = []
    rounded = null_value_col in ['Num_of_Delayed_Payment', 'Changed_Credit_Limit']
    
    for i in range(nRowsTrain):
        train_data_np.append(
            replace_with_median(
                train_data[null_value_col].iloc[i], i, train_data[null_value_col], 8, null_values[null_value_col], rounded
            )
        )
    
    for i in range(nRowsTest):
        test_data_np.append(
            replace_with_median(
                test_data[null_value_col].iloc[i], i, test_data[null_value_col], 4, null_values[null_value_col], rounded
            )
        )
    
    train_data[null_value_col] = pd.Series(train_data_np)
    test_data [null_value_col] = pd.Series(test_data_np)

def fill_month_count_column(value, idx, data_arr, rows_per_customer, null_value):
    
    if value != null_value:
        return value
    
    start_idx = (idx // rows_per_customer) * rows_per_customer
    end_idx = (idx // rows_per_customer + 1) * rows_per_customer
    data_range = data_arr[start_idx:end_idx]
    
    first_valid_value = None
    for value_idx in range(rows_per_customer):
        if data_arr[value_idx] != null_value:
            first_valid_value = [value_idx - start_idx, data_arr[value_idx]]
            break

    if first_valid_value == None:
        return null_value
    else:
        return first_valid_value[1] + (idx % rows_per_customer)

train_data_np = []
test_data_np  = []
col           = 'Credit_History_Age'
    
for i in range(nRowsTrain):
    train_data_np.append(
        fill_month_count_column(train_data[col].iloc[i], i, train_data[col], 8, null_values[col])
    )
    
for i in range(nRowsTest):
    test_data_np.append(
        fill_month_count_column(test_data[col].iloc[i], i, test_data[col], 4, null_values[col])
    )
    
train_data[col] = pd.Series(train_data_np)
test_data [col] = pd.Series(test_data_np)

train_pb_spent = train_data['Payment_Behaviour_Spent'].value_counts()
test_pb_spent  = test_data ['Payment_Behaviour_Spent'].value_counts()
train_pb_value = train_data['Payment_Behaviour_Value'].value_counts()
test_pb_value  = test_data ['Payment_Behaviour_Value'].value_counts()

def fill_categorical_column(value, idx, data_arr, rows_per_customer, null_value, pb_count):
    
    if value != null_value:
        return value
    
    start_idx = (idx // rows_per_customer) * rows_per_customer
    end_idx = (idx // rows_per_customer + 1) * rows_per_customer
    data_range = data_arr[start_idx:end_idx]
    
    pb_count_copied = pb_count.copy()
    for data_value in data_range:
        pb_count_copied[data_value][1] += 1
            
    is_all_null = True
    pb_count_list_customer = []
    
    for cnt_key, cnt_value in pb_count_copied.items():
        pb_count_list_customer.append([cnt_key, cnt_value[0], cnt_value[1]])
        if cnt_key != null_value and cnt_value[1] > 0:
            is_all_null = False
        
    pb_count_list_customer.sort(key = lambda x: x[1], reverse = True)
    pb_count_list_customer.sort(key = lambda x: x[2], reverse = True)
    
    if is_all_null:
        return null_value
    else:
        return pb_count_list_customer[0][0]
    
cols_to_pb_count = {'Payment_Behaviour_Spent': train_pb_spent, 'Payment_Behaviour_Value': train_pb_value}

for col, pb_count in cols_to_pb_count.items():
    train_data_np = []
    test_data_np  = []
    
    pb_count_dict = {}
    for val, cnt in pb_count.items():
        pb_count_dict[val] = [cnt, 0]
        
    print('count of column ' + col + ': ', pb_count_dict)
    
    for i in range(nRowsTrain):
        train_data_np.append(
            fill_categorical_column(train_data[col].iloc[i], i, train_data[col], 8, null_values[col], pb_count_dict)
        )

    for i in range(nRowsTest):
        test_data_np.append(
            fill_categorical_column(test_data[col].iloc[i], i, test_data[col], 4, null_values[col], pb_count_dict)
        )

    train_data[col] = pd.Series(train_data_np)
    test_data [col] = pd.Series(test_data_np)

nullable_columns = null_values.keys()
loan_type_columns = []

for col in nullable_columns:
    if len(col) >= 10 and col[:10] == 'Loan_Type_':
        loan_type_columns.append(col)
        
for col in loan_type_columns:
    arr = np.array(train_data[col])
    print('(train) mean of [' + col + '] :', arr[arr != null_values[col]].mean())
    
print('')
for col in loan_type_columns:
    arr = np.array(test_data[col])
    print('(test) mean of [' + col + '] :', arr[arr != null_values[col]].mean())

for col in loan_type_columns:
    train_data[col] = train_data[col].apply(lambda x: x if x != 'nan' else 0)
    test_data [col] = test_data [col].apply(lambda x: x if x != 'nan' else 0)

for median_col in ['Monthly_Inhand_Salary', 'Monthly_Balance', 'Num_of_Delayed_Payment', 'Num_Credit_Inquiries']:
    arr_train = np.array(train_data[median_col])
    arr_test  = np.array(test_data [median_col])
    
    median_train = np.median(arr_train[arr_train != null_values[median_col]])
    median_test  = np.median(arr_test [arr_test  != null_values[median_col]])
    median_all   = (median_train * nRowsTrain + median_test * nRowsTest) / (nRowsTrain + nRowsTest)
    
    if median_col in ['Num_of_Delayed_Payment', 'Num_Credit_Inquiries']:
        median_all = median_all + 0.5 if abs(median_all % 1.0 - 0.5) < 0.25 else median_all
    
    print('median of [' + median_col + '] :', median_all)
    
    for data in [train_data, test_data]:
        data[median_col] = data[median_col].apply(lambda x: median_all if x == null_values[median_col] else x)

for data in [train_data, test_data]:
    data['Credit_Mix'] = data['Credit_Mix'].apply(lambda x: 'Low' if x == null_values['Credit_Mix'] else x)

copy_train = train_data.copy()
copy_test = test_data.copy()
del copy_train['Month']
del copy_test['Month']
del copy_train['Occupation']
del copy_test['Occupation']

def cod_score(x):
    if x == "Good":
        return 1
    elif x == "Standard":
        return 0
    else:
        return -1

copy_train['Credit_Score'] = copy_train['Credit_Score'].apply(cod_score)
copy_train['Credit_Mix'] = copy_train['Credit_Mix'].apply(cod_score)
copy_test['Credit_Mix'] = copy_test['Credit_Mix'].apply(cod_score)

cols_to_onehot = ['Payment_of_Min_Amount', 'Payment_Behaviour_Spent', 'Payment_Behaviour_Value']

for data in [copy_train, copy_test]:
    for col in cols_to_onehot:
        unique_values = data[col].unique()
        
        for uniq in unique_values:
            data[col + '_' + uniq] = data[col].apply(lambda x: 1.0 if x == uniq else 0.0)
            
copy_train = copy_train.drop(columns = cols_to_onehot)
copy_test  = copy_test.drop(columns = cols_to_onehot)

del copy_train['Monthly_Inhand_Salary']
del copy_test['Monthly_Inhand_Salary']

cols_to_log = ['Annual_Income', 'Total_EMI_per_month', 'Amount_invested_monthly']

train_mean = {}
train_std  = {}

for col in cols_to_log:
    train_mean[col] = data[col].mean()
    train_std [col] = data[col].std()

for data in [copy_train, copy_test]:
    for col in cols_to_log:
        data[col] = data[col].apply(lambda x: np.log(x + 1.0))
        data[col] = data[col].apply(lambda x: (x - train_mean[col]) / train_std[col])

names_to_norm = ['Age', 'Num_Bank_Accounts', 'Num_Credit_Card', 
                    'Interest_Rate', 'Num_of_Loan', 'Delay_from_due_date', 
                    'Num_of_Delayed_Payment','Changed_Credit_Limit', 'Num_Credit_Inquiries', 
                    'Outstanding_Debt', 'Credit_History_Age', 'Monthly_Balance']

train_mean = {}
train_std  = {}

for col in names_to_norm:
    train_mean[col] = data[col].mean()
    train_std [col] = data[col].std()

for data in [copy_train, copy_test]:
    for col in names_to_norm:
        data[col] = data[col].apply(lambda x: (x - train_mean[col]) / train_std[col])

# Select independent and dependent variable
y = copy_train["Credit_Score"]
X = copy_train
del copy_train["Credit_Score"]

# Split the dataset into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, shuffle=True, random_state=42)


import optuna
import pickle

classifier = RandomForestClassifier(n_estimators=181, max_depth=30, min_samples_leaf=2, random_state=42)

# Fit the model
classifier.fit(X_train, y_train)

# Make pickle file of our model
pickle.dump(classifier, open("model.pkl", "wb"))
