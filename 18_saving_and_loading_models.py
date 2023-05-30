import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, n_input_features):
        super(Model, self).__init__()
        self.linear = nn.Linear(n_input_features, 1)

    def forward(self, x):
        y_pred = torch.sigmoid(self.linear(x))
        return y_pred

model = Model(n_input_features = 6)
# train the model
learning_rate = 0.01
optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate)
print(optimizer.state_dict())

# Lazy method
FILE = 'model.pth'
torch.save(model, FILE)
model = torch.load(FILE)
model.eval()

for param in model.parameters():
    print(param)

# Preferred method
FILE = 'model.pth'
torch.save(model.state_dict(), FILE)

loaded_model = Model(n_input_features = 6)
loaded_model.load_state_dict(torch.load(FILE))
loaded_model.eval()

for param in loaded_model.parameters():
    print(param)

# checkpoints
checkpoint = {
    "epoch": 90,
    "model_state": model.state_dict(),
    "optim_state": optimizer.state_dict()
}

torch.save(checkpoint, "checkpoint.pth")

loaded_checkpoint = torch.load("checkpoint.pth")
epoch = loaded_checkpoint["epoch"]
model = Model(n_input_features = 6)
optimizer = torch.optim.SGD(model.parameters(), lr = 0)
model.load_state_dict(checkpoint["model_state"])
optimizer.load_state_dict(checkpoint["optim_state"])
print(optimizer.state_dict())


# save on GPU, load on CPU
device = torch.device("mps")
model.to(device)
torch.save(model.state_dict(), FILE)

device = torch.device("cpu")
model = Model(*args, **kwargs)
model.load_state_dict(torch.load(FILE, map_location = device))

# save on GPU, load on GPU
device = torch.device("mps")
model.to(device)
torch.save(model.state_dict(), FILE)

model = Model(*args, **kwargs)
model.load_state_dict(torch.load(FILE))
model.to(device)

# save on CPU, load on GPU
torch.save(model.state_dict(), FILE)

device = torch.device("mps")
model = Model(*args, **kwargs)
model.load_state_dict(torch.load(FILE, map_location = "mps:0"))
model.to(device)
