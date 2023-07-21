import pandas as pd

pretrain_30 = pd.read_csv('data/cluster-30/pretrain_ec.csv')
pretrain_50 = pd.read_csv('data/cluster-50/pretrain_ec.csv')
pretrain_70 = pd.read_csv('data/cluster-70/pretrain_ec.csv')
pretrain_90 = pd.read_csv('data/cluster-90/pretrain_ec.csv')
train_30 = pd.read_csv('data/cluster-30/train_ec_3d.csv')
train_50 = pd.read_csv('data/cluster-50/train_ec_3d.csv')
train_70 = pd.read_csv('data/cluster-70/train_ec_3d.csv')
train_90 = pd.read_csv('data/cluster-90/train_ec_3d.csv')
test = pd.read_csv('data/cluster-30/test_ec_3d.csv') # test_ec_3d.csv is the same for all clusters

# count the number of unique EC numbers in each pretrain data. ec_number is a string of EC numbers separated by comma. seperate them and count the number of unique EC numbers.
all_ecs = []
for i in range(pretrain_30.shape[0]):
    all_ecs.extend(pretrain_30['ec_number'][i].split(','))
del pretrain_30

print('ec number of pretrain: ', len(list(set(all_ecs))))

# Save the unique EC numbers with their numerical representation in a csv file
unique_ecs = list(set(all_ecs))
unique_ecs = [[unique_ecs[i], i] for i in range(len(unique_ecs))]
unique_ecs = pd.DataFrame(unique_ecs, columns=['ec_number', 'ec_number_num'])
unique_ecs.to_csv('data/cluster-30/unique_ecs_pretrain.csv', index=False)

all_ecs = []
for i in range(pretrain_50.shape[0]):
    all_ecs.extend(pretrain_50['ec_number'][i].split(','))
del pretrain_50

print('ec number of pretrain: ', len(list(set(all_ecs))))
unique_ecs = list(set(all_ecs))
unique_ecs = [[unique_ecs[i], i] for i in range(len(unique_ecs))]
unique_ecs = pd.DataFrame(unique_ecs, columns=['ec_number', 'ec_number_num'])
unique_ecs.to_csv('data/cluster-50/unique_ecs_pretrain.csv', index=False)

all_ecs = []
for i in range(pretrain_70.shape[0]):
    all_ecs.extend(pretrain_70['ec_number'][i].split(','))
del pretrain_70

print('ec number of pretrain: ', len(list(set(all_ecs))))
unique_ecs = list(set(all_ecs))
unique_ecs = [[unique_ecs[i], i] for i in range(len(unique_ecs))]
unique_ecs = pd.DataFrame(unique_ecs, columns=['ec_number', 'ec_number_num'])
unique_ecs.to_csv('data/cluster-70/unique_ecs_pretrain.csv', index=False)

all_ecs = []
for i in range(pretrain_90.shape[0]):
    all_ecs.extend(pretrain_90['ec_number'][i].split(','))
del pretrain_90

print('ec number of pretrain: ', len(list(set(all_ecs))))
unique_ecs = list(set(all_ecs))
unique_ecs = [[unique_ecs[i], i] for i in range(len(unique_ecs))]
unique_ecs = pd.DataFrame(unique_ecs, columns=['ec_number', 'ec_number_num'])
unique_ecs.to_csv('data/cluster-90/unique_ecs_pretrain.csv', index=False)

# count the number of unique EC numbers in each train data. ec_number is a string of EC numbers separated by comma. seperate them and count the number of unique EC numbers.
all_ecs = []
for i in range(train_30.shape[0]):
    all_ecs.extend(train_30['ec_number'][i].split(','))
del train_30

print('ec number of train: ', len(list(set(all_ecs))))
unique_ecs = list(set(all_ecs))
unique_ecs = [[unique_ecs[i], i] for i in range(len(unique_ecs))]
unique_ecs = pd.DataFrame(unique_ecs, columns=['ec_number', 'ec_number_num'])
unique_ecs.to_csv('data/cluster-30/unique_ecs_train.csv', index=False)

all_ecs = []
for i in range(train_50.shape[0]):
    all_ecs.extend(train_50['ec_number'][i].split(','))
del train_50

print('ec number of train: ', len(list(set(all_ecs))))
unique_ecs = list(set(all_ecs))
unique_ecs = [[unique_ecs[i], i] for i in range(len(unique_ecs))]
unique_ecs = pd.DataFrame(unique_ecs, columns=['ec_number', 'ec_number_num'])
unique_ecs.to_csv('data/cluster-50/unique_ecs_train.csv', index=False)

all_ecs = []
for i in range(train_70.shape[0]):
    all_ecs.extend(train_70['ec_number'][i].split(','))
del train_70

print('ec number of train: ', len(list(set(all_ecs))))
unique_ecs = list(set(all_ecs))
unique_ecs = [[unique_ecs[i], i] for i in range(len(unique_ecs))]
unique_ecs = pd.DataFrame(unique_ecs, columns=['ec_number', 'ec_number_num'])
unique_ecs.to_csv('data/cluster-70/unique_ecs_train.csv', index=False)

all_ecs = []
for i in range(train_90.shape[0]):
    all_ecs.extend(train_90['ec_number'][i].split(','))
del train_90

print('ec number of train: ', len(list(set(all_ecs))))
unique_ecs = list(set(all_ecs))
unique_ecs = [[unique_ecs[i], i] for i in range(len(unique_ecs))]
unique_ecs = pd.DataFrame(unique_ecs, columns=['ec_number', 'ec_number_num'])
unique_ecs.to_csv('data/cluster-90/unique_ecs_train.csv', index=False)

# count the number of unique EC numbers in the test data. ec_number is a string of EC numbers separated by comma. seperate them and count the number of unique EC numbers.
all_ecs = []
for i in range(test.shape[0]):
    all_ecs.extend(test['ec_number'][i].split(','))
del test

print('ec number of test: ', len(list(set(all_ecs))))
unique_ecs = list(set(all_ecs))
unique_ecs = [[unique_ecs[i], i] for i in range(len(unique_ecs))]
unique_ecs = pd.DataFrame(unique_ecs, columns=['ec_number', 'ec_number_num'])
unique_ecs.to_csv('data/cluster-30/unique_ecs_test.csv', index=False)

