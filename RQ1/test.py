import torch
from torch.utils.data import DataLoader, TensorDataset
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from AI_project.RQ1.model import MLP1
from AI_project.RQ1.data_reader import load_data, load_data_mf, customer_buckets, matrix_representation
import pandas as pd
from AI_project.RQ1.helper import train_softmax, validate_softmax
import numpy as np
from tqdm import tqdm
from scipy.sparse import csr_matrix
import torch.nn.functional as F


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

articles = csr_matrix(articles.values)
customers = csr_matrix(customers.values)

avg_basket_size = round(np.mean(transactions.groupby("customer_id")["customer_id"].count()))

from AI_project.RQ1.data_reader import customer_buckets

articles
buckets = customer_buckets(transactions, train_test=False)
all_articles_set = set(range(articles.shape[0]))
negative_buckets = {}
for customer in tqdm(transactions["customer_id"].unique()):
    negative_buckets[customer] = np.random.choice(list(all_articles_set - set(buckets[customer])), avg_basket_size)


all_articles = set(transactions['article_id'].unique())

# Define the number of negative samples per customer
avg_basket_size = 2  # Adjust as needed

data = {
    'transaction_id': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'customer_id': [101, 102, 103, 104, 105, 101, 103, 102, 106, 104],
    'article_id': ['A', 'B', 'A', 'C', 'B', 'A', 'C', 'B', 'D', 'C'],
    'quantity': [2, 1, 3, 2, 1, 2, 1, 3, 2, 1],
    'transaction_date': [
        '2023-01-01', '2023-01-01', '2023-01-02', '2023-01-02', '2023-01-03',
        '2023-01-04', '2023-01-04', '2023-01-05', '2023-01-06', '2023-01-07'
    ]
}

transactions = pd.DataFrame(data)
# Create a function to generate negative samples for a customer
def generate_negative_samples(customer_history):
    # Get the set of articles the customer has purchased
    customer_articles = set(customer_history)
    
    # Calculate the set difference to find items not purchased by the customer
    negative_items = list(all_articles - customer_articles)
    
    # Randomly sample negative items based on avg_basket_size
    num_samples = min(avg_basket_size, len(negative_items))
    sampled_negative_items = np.random.choice(negative_items, num_samples, replace=False)
    
    return sampled_negative_items

# Group transactions by customer and apply the generate_negative_samples function
negative_samples = transactions.groupby('customer_id')['article_id'].apply(generate_negative_samples)

buckets = customer_buckets(transactions, train_test=False)

train_dataloader, val_dataloader = load_data_mf(transactions, batch_size=100)
it = iter(train_dataloader)
articles_id, customers_id = next(it)
random_negative_articles = torch.tensor([np.random.choice(list(all_articles_set-set(buckets[cstmr.item()]))) for cstmr in customers_id])
articles_features = torch.tensor(articles[articles_id].todense(), dtype=torch.float32)
customer_features = torch.tensor(customers[customers_id].todense(), dtype=torch.float32)
negative_articles_features = torch.tensor(articles[random_negative_articles].todense(), dtype=torch.float32)
articles_features = articles_features.to(mps_device)
customer_features = customer_features.to(mps_device)
negative_articles_features = negative_articles_features.to(mps_device)
customer_features = np.vstack([customer_features, customer_features])
articles_features = np.vstack([articles_features, negative_articles_features])


buckets_list = torch.tensor(list(buckets.values()))
matrix = matrix_representation(transactions, train_test=False)
matrix[customers_id]

random_negative = torch.tensor([np.random.choice(list(all_articles_set-set(buckets[cstmr.item()]))) for cstmr in customers_id])
cstmr = customers_id[0]
x = all_articles_set-set(buckets[cstmr.item()])
x.shape
random.sample(all_articles_set,1)
x = torch.vstack([customer_features,customer_features])
customer_features.shape

np.hstack([torch.ones(10),torch.zeros(10)])