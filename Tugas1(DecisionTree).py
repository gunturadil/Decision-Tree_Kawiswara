#!/usr/bin/env python
# coding: utf-8

# # 1. Import library

# In[1]:


import pandas as pd
import numpy as np


# # 2. Import data

# In[2]:


train_data = pd.DataFrame({'Refund': ['Yes', 'No', 'No', 'Yes', 'No', 'No', 'Yes', 'No', 'No', 'No'], 
                     'Martial_Status': ['Single', 'Married', 'Single', 'Married', 'Divorced', 'Married', 'Divorced', 'Single', 'Married', 'Single'], 
                     'Taxable_Income': [125000, 100000, 70000, 120000, 95000, 60000, 220000, 85000, 75000, 90000], 
                     'Cheat': ['No', 'No', 'No', 'No', 'Yes', 'No', 'No', 'Yes', 'No', 'Yes']})
train_data


# In[3]:


test_data = pd.DataFrame({'Refund': ['No', 'Yes', 'No', 'Yes', 'No', 'No'], 
                          'Martial_Status': ['Single', 'Married', 'Married', 'Divorced', 'Single', 'Married'], 
                          'Taxable_Income': [75000, 50000, 150000, 90000, 40000, 80000]})
test_data


# # 3. Training

# In[14]:


def preprocessing(data):
    encode_boolean = lambda x: 0 if x == 'No' else 1
    encode_martial = lambda x: 0 if x == 'Single' else 1 if x == 'Married' else 2
    data.Refund = data.Refund.map(encode_boolean)
    data.Martial_Status = data.Martial_Status.map(encode_martial)
    if 'Cheat' in data.columns:
        data.Cheat = data.Cheat.map(encode_boolean)
    return data


# In[16]:


def probability(data, column, x):
    count_x = len(data[(data[column] == x) & (data.Cheat == 1)])
    count_y = len(data[data[column] == x][column])
    try:
        prob_1 = count_x / count_y
    except:
        prob_1 = 0
    prob_0 = 1 - prob_1
    return prob_1, prob_0


# In[ ]:





# In[ ]:


def print_rule(rule):
    len_rule = len(rule) - 1
    print('if {} {} {}:'.format(rule[0][0], rule[0][3], rule[0][1]))
    print('\tCheat == {}'.format(rule[0][2]))
    for i in range(1,  len_rule):
        print('else if {} {} {}:'.format(rule[i][0], rule[i][3], rule[i][1]))
        print('\tCheat == {}'.format(rule[i][2]))
    print('else: ')
    print('\tCheat == {}'.format(rule[len_rule][2]))


# In[ ]:


def condition(data, rule):
    if rule[3] == '==':
        return data == rule[1]
    elif rule[3] == '<':
        return data < rule[1]


# In[ ]:


def create_model(data):
    rule = {}
    category = data.drop('Cheat', axis=1).select_dtypes(exclude=np.number).columns.tolist()
    data = preprocessing(data)
    counter = 0 
    condition = False
    while(condition == False):
        column, value, target, equation = None, None, None, None
        for feature in data.drop('Cheat', axis=1).columns:
            if feature in category:
                for i in data[feature].unique():
                    prob_0, prob_1 = probability(data, feature, i)
                    if prob_0 == 1:
                        column = feature
                        value = i
                        target = 0
                        equation = '=='
                        break
                    if prob_1 == 1:
                        column = feature
                        value = i
                        target = 1
                        equation = '=='
                        break
                if column == feature:
                    rule[counter] = str(column), int(value), target, equation
                    data = data[data[rule[counter][0]] != rule[counter][1]]
                    counter += 1
                    column, value, target, equation = None, None, None, None
                    break
            else:
                column = feature
                value = data[data.Cheat == 1][feature].min()
                target = 0 #1
                equation = '<'
                rule[counter] = str(column), int(value), target, equation
                data = data[data[rule[counter][0]] < rule[counter][1]]
                counter += 1
                column, value, target, equation = None, None, None, None
                break
        if len(data) == 1:
            target = 1#data.Cheat.values[0]
            rule[counter] = None, None, target, None
            column, value, target, equation = None, None, None, None
            condition = True
    return rule


# In[ ]:


def predict(data, rule):
    for x in range(len(data)):
        for i in range(len(rule)):
            if data.index[x] == rule[i][0]:
                if condition(data[x], rule[i]):
                    return rule[i][2]


# In[ ]:


rule = create_model(train_data)
print_rule(rule)


# # 4. Testing

# In[ ]:


test_data = preprocessing(test_data)
test_data['Prediction'] =[predict(test_data.loc[i], rule) for i in range(len(test_data))]
test_data.head()

