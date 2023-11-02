import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

# Define the training function for multi-label classification with validation
def train_softmax(model, train_dataloader, val_dataloader, criterion, optimizer, save_dir, num_epochs=5):
    mps_device = torch.device("mps")
    model = model.to(mps_device)
    val_loss_list = []
    val_acc_list = []
    max_acc = 0
    for epoch in tqdm(range(num_epochs)):
        model.train()
        for inputs, targets in train_dataloader:
            inputs = inputs.to_dense().to(mps_device)
            targets = targets.to_dense().to(mps_device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()       
        # Validatete for the epoch
        train_loss = loss.item()
        val_loss, val_accuracy = validate_softmax(model, val_dataloader, criterion)
        val_loss_list.append(val_loss)
        val_acc_list.append(val_accuracy)    
        if val_accuracy>max_acc:
            max_acc = val_accuracy
            torch.save(model, save_dir)
        print(f'Epoch [{epoch + 1}/{num_epochs}] - Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%')
    return val_loss_list, val_acc_list

# Define the validation function for multi-label classification
def validate_softmax(model, val_dataloader, criterion, k=5):
    mps_device = torch.device("mps")
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in val_dataloader:
            inputs = inputs.to_dense().to(mps_device)
            targets = targets.to_dense().to(mps_device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            val_loss += loss.item()
            top_k_values, top_k_indices = torch.topk(outputs, k=k, dim=1) # Use a threshold of 0.5 for binary prediction
            predicted = torch.zeros_like(outputs)
            predicted.scatter_(1, top_k_indices, 1)
            correct_recommendations = predicted * targets
            correct += correct_recommendations.sum().item() 
            total += targets.sum()
    
    val_loss /= len(val_dataloader)
    val_accuracy = (correct / total) * 100.0
    return val_loss, val_accuracy.item()


# Define the training function for multi-label classification with validation
def train_two_tower(model, customers, articles, train_dataloader, val_dataloader, criterion, optimizer, save_dir, num_epochs=5):
    mps_device = torch.device("mps")
    model = model.to(mps_device)
    val_loss_list = []
    val_acc_list = []
    max_acc = 0
    for epoch in tqdm(range(num_epochs)):
        model.train()
        for articles_id, customers_id in train_dataloader: 
            articles_features = torch.tensor(articles[articles_id].todense(), dtype=torch.float32)
            customer_features = torch.tensor(customers[customers_id].todense(), dtype=torch.float32)
            articles_features = articles_features.to(mps_device)
            customer_features = customer_features.to(mps_device)
            optimizer.zero_grad()
            outputs = model(customer_features, articles_features)
            loss = criterion(outputs, torch.ones(len(outputs)).to(mps_device))
            loss.backward()
            optimizer.step()
        # Validatete for the epoch
        # train_loss = loss.item()
        # val_loss, val_accuracy = validate_softmax(model, val_dataloader, criterion)
        # val_loss_list.append(val_loss)
        # val_acc_list.append(val_accuracy)    
        # if val_accuracy>max_acc:
        #     max_acc = val_accuracy
        #     torch.save(model, save_dir)
        # print(f'Epoch [{epoch + 1}/{num_epochs}] - Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%')
    return val_loss_list, val_acc_list

# Define the validation function for multi-label classification
def validate_softmax(model, val_dataloader, criterion, k=5):
    mps_device = torch.device("mps")
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in val_dataloader:
            inputs = inputs.to_dense().to(mps_device)
            targets = targets.to_dense().to(mps_device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            val_loss += loss.item()
            top_k_values, top_k_indices = torch.topk(outputs, k=k, dim=1) # Use a threshold of 0.5 for binary prediction
            predicted = torch.zeros_like(outputs)
            predicted.scatter_(1, top_k_indices, 1)
            correct_recommendations = predicted * targets
            correct += correct_recommendations.sum().item() 
            total += targets.sum()
    
    val_loss /= len(val_dataloader)
    val_accuracy = (correct / total) * 100.0
    return val_loss, val_accuracy.item()



