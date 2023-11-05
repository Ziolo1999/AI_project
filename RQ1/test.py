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