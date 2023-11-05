import torch 
import torch.nn.functional as nn

def recommender_softmax(model, dataloader, evaluate:bool=False, top_k=5):
    mps_device = torch.device("mps")
    model.eval()
    recommendations = torch.zeros(size=(0,top_k)).to(mps_device)
    model = model.to(mps_device)
    correct = 0
    total = 0
    with torch.no_grad():
        if evaluate:
            with torch.no_grad():
                for inputs, targets in dataloader:
                    inputs = inputs.to_dense().to(mps_device)
                    targets = targets.to_dense().to(mps_device)
                    # Get predictions
                    outputs = model(inputs)
                    # Select top k articles
                    _, top_k_indices = torch.topk(outputs, k=top_k, dim=1)
                    recommendations = torch.vstack([recommendations, top_k_indices])
                    predicted = torch.zeros_like(outputs)
                    predicted.scatter_(1, top_k_indices, 1)
                    correct_recommendations = predicted * targets
                    correct += correct_recommendations.sum().item() 
                    total += targets.sum()
            accuracy = correct / total
            return recommendations, accuracy
        else:
                for inputs in dataloader:
                    inputs = inputs.to_dense().to(mps_device)
                    # Get predictions
                    outputs = model(inputs)
                    # Select top k articles
                    _, top_k_indices = torch.topk(outputs, k=top_k, dim=1)
                    recommendations = torch.vstack([recommendations, top_k_indices])        
    return recommendations

def recommender_two_towers(model, dataloader_cust, dataloader_art, targets, evaluate: bool=False, top_k=5):
    mps_device = torch.device("mps")
    model = model.to(mps_device)
    # Generate customers and articles embeddings
    full_articles_embeddings = torch.zeros(size=(0,model.ArticleTower.fc2.output_dim))
    full_customers_embeddings = torch.zeros(size=(0,model.CustomerTower.fc2.output_dim))
    with torch.no_grad():
        # push customers through customer tower
        for customers_features in dataloader_cust:
            customers_embeddings = model.CustomerTower(customers_features)
            full_customers_embeddings = torch.vstack([full_customers_embeddings, customers_embeddings])
        # push articles through article tower
        for articles_features in dataloader_art:
            articles_features = model.ArticleTower(articles_features)
            full_articles_embeddings = torch.vstack([full_articles_embeddings, articles_features])
    # calculate probability of being purchased
    probability_matrix = nn.sigmoid(torch.matmul(full_customers_embeddings, full_articles_embeddings.T))
    _, top_k_indices = torch.topk(probability_matrix, k=top_k, dim=1)

    if evaluate:
        predicted = torch.zeros_like(probability_matrix)
        predicted.scatter_(1, top_k_indices, 1)
        correct_recommendations = predicted * targets
        total_correct = correct_recommendations.sum().item()
        total  = targets.sum().item()
        accuracy = total_correct / total
        return top_k_indices, accuracy
    else:
        return top_k_indices

# # Create a sample binary mask matrix (1s are the positions to set to 0)
# mask_matrix = np.array([[0, 1, 0],
#                         [1, 0, 1],
#                         [0, 0, 1]])

# # Create a sample matrix where you want to apply the mask
# matrix1 = np.array([[10, 20, 30],
#                    [40, 50, 60],
#                    [70, 80, 90]])

# # Apply the mask to set corresponding elements to 0
# result_matrix = matrix1 * (1 - mask_matrix)

# print("Resulting matrix:")
# print(result_matrix)