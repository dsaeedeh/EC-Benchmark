# Read unique ec numbers from train and test data and merge them in a single .txt file
import pandas as pd

# train_ec = pd.read_csv('data/cluster-30/unique_ecs_train.csv')
# test_ec = pd.read_csv('data/cluster-30/unique_ecs_test.csv')
# price_ec = pd.read_csv('data/cluster-30/unique_ecs_price.csv')

# train_ec = train_ec['ec_number'].tolist()
# test_ec = test_ec['ec_number'].tolist()
# price_ec = price_ec['ec_number'].tolist()

# all_ec = train_ec + test_ec + price_ec
# all_ec = list(set(all_ec))

# with open('data/cluster-30/ECNumberList.txt', 'w') as f:
#     for item in all_ec:
#         f.write("%s\n" % item)

#remove 3d info from train and test data
# train = pd.read_csv('data/cluster-30/train_ec_3d.csv')
# test = pd.read_csv('data/cluster-30/test_ec_3d.csv')

# train = train.drop(['3d_info'], axis=1)
# test = test.drop(['3d_info'], axis=1)

# train.to_csv('data/cluster-30/train_ec.csv', index=False)
# test.to_csv('data/cluster-30/test_ec.csv', index=False)

#convert test_ec.csv to test_ec.fasta
test = pd.read_csv('data/cluster-30/test_ec.csv')
with open('data/cluster-30/test_ec.fasta', 'w') as f:
    for i in range(test.shape[0]):
        f.write(">" + test.iloc[i, 0] + "\n")
        f.write(test.iloc[i, 1] + "\n")


