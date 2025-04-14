from load_encoder import load_encoder
from dataLoader.dataLoader import load_ratings_with_personality_traits
import time
import torch
from personalityClassifier.utils import get_device
import torch.optim as optim
from personalityClassifier.personality_prediction_model import *
import  pandas as pd

encoder = load_encoder("../output_autoencoder/")
seed = int(time.time())

train_data, validation_data, train_user_features, valid_user_features = load_ratings_with_personality_traits(
    path='../data/personality-isf2018/', valfrac=0.1, seed=123, transpose=False, feature_classification=True)
device = get_device()
train_data = torch.from_numpy(train_data).to(device).squeeze()
validation_data = torch.from_numpy(validation_data).to(device).squeeze()

X_train, _ = encoder.forward(train_data)
#X_train_np = X_train.detach().cpu().numpy()
#df = pd.DataFrame(X_train_np)
#df.to_csv("encoded_users.csv", index=False)

X_test, _ = encoder.forward(validation_data)

X_train_tensor = X_train.detach().float()
X_test_tensor = X_test.detach().float()

Y_train_tensor = torch.tensor(train_user_features, dtype=torch.float32).squeeze(1)
Y_test_tensor = torch.tensor(valid_user_features, dtype=torch.float32).squeeze(1)

input_dim = X_train_tensor.shape[1]
output_dim = Y_train_tensor.shape[1]
print("input_dim")
print(input_dim)
print("output_dim")
print(output_dim)

model = NonLinearRegressor(input_dim, input_dim//2, output_dim)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

for epoch in range(1000):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train_tensor)
    loss = criterion(outputs, Y_train_tensor)
    loss.backward()
    optimizer.step()

    if epoch % 20 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

model.eval()
with torch.no_grad():
    Y_pred = model(X_test_tensor)
    rmse = torch.sqrt(torch.mean((Y_pred - Y_test_tensor)**2))
    print("RMSE:", rmse.item())
    print("real: ", Y_test_tensor[0])
    print("predicted: ", Y_pred[0])

    print("real: ", Y_test_tensor[1])
    print("predicted: ", Y_pred[1])

