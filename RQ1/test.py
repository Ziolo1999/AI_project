import torch
from torch.utils.data import DataLoader, TensorDataset
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from AI_project.RQ1.model import MLP1, TwoTower
from AI_project.RQ1.data_reader import load_data, load_data_mf, customer_buckets, matrix_representation
import pandas as pd
from AI_project.RQ1.helper import train_softmax, validate_softmax
import numpy as np
from tqdm import tqdm
from scipy.sparse import csr_matrix
import torch.nn.functional as F
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

class TwoTower(nn.Module):
    def __init__(self, input_article_dim, input_customer_dim, output_dim=3):
        super(TwoTower,self).__init__()
        # Article tower
        self.afc1 = nn.Embedding(input_article_dim, output_dim)
        # Customer tower
        self.cfc1 = nn.Embedding(input_customer_dim, output_dim)

    def forward(self, x, y):
        #customers
        x = self.cfc1(x)
        # articles
        y = self.afc1(y)
        # return product 
        return torch.matmul(x,y.T).diag()

# read data
transactions = pd.read_csv("data/preprocessed/transactions.csv") 
articles = pd.read_csv("data/preprocessed/articles.csv") 
customers = pd.read_csv("data/preprocessed/customers.csv") 

# one hot encoding 
articles = articles.set_index("article_id")
customers = customers.set_index("customer_id")

article_enc = OneHotEncoder(sparse_output=True)
articles = article_enc.fit_transform(articles)

customers_categorical = ["FN",'Active',"club_member_status", "fashion_news_frequency"]
customers_cont = ["age"]

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(sparse_output=True), customers_categorical),
        ('cont', 'passthrough', customers_cont)  # 'passthrough' means no transformation for continuous variables
    ],
    remainder='drop'  # Drop any columns not explicitly transformed
)

customer_last_purchase = transactions.groupby('customer_id')['t_dat'].max()
merged = transactions.merge(customer_last_purchase, on='customer_id', suffixes=('', '_last_purchase'))
# filter train and test dataset
x_transactions = merged[merged['t_dat'] != merged['t_dat_last_purchase']]
y_transactions = merged[merged['t_dat'] == merged['t_dat_last_purchase']]

customers = csr_matrix(preprocessor.fit_transform(customers))
buckets = customer_buckets(transactions, train_test=False)
train_dataloader, val_dataloader = load_data_mf(transactions, batch_size=100)
input_article_dim = articles.shape[1]
input_customer_dim = customers.shape[1]
model = TwoTower(input_article_dim, input_customer_dim, output_dim=3)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
save_dir = "KarolZiolo/RQ1/models/TwoTower1.pt"

# Validatete for the epoch

it = iter(val_dataloader)
articles_id, customers_id = next(it)
# get features
articles_features = torch.tensor(articles[articles_id].todense(), dtype=torch.float32)
customer_features = torch.tensor(customers[customers_id].todense(), dtype=torch.float32)
# push them through model
outputs = model(customer_features, articles_features)
vals = torch.ones(len(outputs))
# calculate the loss
loss = criterion(outputs, vals.to(mps_device))


val_loss += loss.item()
top_k_values, top_k_indices = torch.topk(outputs, k=k, dim=1) # Use a threshold of 0.5 for binary prediction
predicted = torch.zeros_like(outputs)
predicted.scatter_(1, top_k_indices, 1)
correct_recommendations = predicted * targets
correct += correct_recommendations.sum().item() 
total += targets.sum()

x = torch.tensor([[1,2,3],[1,2,3]])
y = torch.tensor([[1,2,3],[1,2,3]])
nn.functional.sigmoid(torch.tensor([0,12,1]))


import pandas as pd

data = {
    'transaction_id': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'customer_id': [101, 102, 103, 104, 105, 101, 103, 102, 106, 104],
    'article_id': ['A', 'B', 'A', 'C', 'B', 'A', 'C', 'B', 'D', 'C'],
    'transaction_date': [
        '2023-01-01', '2023-01-01', '2023-01-02', '2023-01-02', '2023-01-03',
        '2023-01-04', '2023-01-04', '2023-01-05', '2023-01-06', '2023-01-07'
    ]
}

transactions = pd.DataFrame(data)

import pandas as pd
import random

# Assuming you have a transactions dataframe with columns: date, customer_id, article_id
# Replace "transactions" with the name of your original transactions dataframe

# Determine the range of customer and article IDs
unique_customers = transactions['customer_id'].unique()
unique_articles = transactions['article_id'].unique()

# Create a list to store negative samples
negative_samples = []
len(transactions)
# Set the number of negative samples you want to generate
num_negative_samples = 30_000_000  # Adjust this as needed

random_cust = np.random.choice(unique_customers, num_negative_samples)
random_articles = np.random.choice(unique_articles, num_negative_samples)
# Create a DataFrame for the negative samples
negative_samples_df = pd.DataFrame(zip(random_cust, random_articles), columns=["customer_id","article_id",])

unique_pairs = set(zip(transactions['customer_id'], transactions['article_id']))

filtered_df2 = negative_samples_df[~negative_samples_df.apply(lambda row: (row['customer_id'], row['article_id']) in unique_pairs, axis=1)].copy()

def create_random_candidates(transactions, save_dir=None, num_sample=30_000_000):
    # get unique customers and articles
    unique_customers = transactions['customer_id'].unique()
    unique_articles = transactions['article_id'].unique()
    # select random customers and articles
    random_cust = np.random.choice(unique_customers, num_sample)
    random_articles = np.random.choice(unique_articles, num_sample)
    # get negative candidates dataframe
    negative_samples_df = pd.DataFrame(zip(random_cust, random_articles), columns=["customer_id","article_id",])
    # delete duplicates from original dataset
    unique_pairs = set(zip(transactions['customer_id'], transactions['article_id']))
    filtered_df = negative_samples_df[~negative_samples_df.apply(lambda row: (row['customer_id'], row['article_id']) in unique_pairs, axis=1)].copy()
    # set purchased variable
    filtered_df["purchased"] = np.zeros(len(filtered_df))
    transactions["purchased"] = np.ones(len(transactions))
    # merge dataframes
    merge = pd.concat([transactions[["customer_id","article_id", "purchased"]],filtered_df[["customer_id","article_id", "purchased"]]])
    # return shuffled dataframe
    shuffled_df = merge.sample(frac=1).reset_index(drop=True)
    if save_dir != None:
        shuffled_df.to_csv(save_dir)
    return shuffled_df

merged_df = create_random_candidates(transactions, save=False, num_sample=30_000_000)

merged_df.to_csv("data/preprocessed/transactions_candidates.csv")
