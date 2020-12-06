import torch
import torch.nn as nn
import torch.optim as optim
import os
import cv2
import numpy as np
import time
from tqdm import tqdm
from neural_net import Net
    
def fwd_pass(X, y, train=False):
    if train:
        net.zero_grad()
    outputs = net(X)
    matches = [torch.argmax(i) == torch.argmax(j) for i, j in zip(outputs, y)]
    acc = matches.count(True)/len(matches)
    loss = loss_function(outputs, y)
    
    if train:
        loss.backward()
        optimizer.step()
    return acc, loss

def test(size=32):
    random_start = np.random.randint(len(test_X)-size)
    X, y = test_X[random_start:random_start+size], test_y[random_start:random_start+size]
    val_acc, val_loss = fwd_pass(X.view(-1, 1, resize_var, resize_var).to(device), y.to(device))
    return val_acc, val_loss

def train():
    BATCH_SIZE = 10
    EPOCHS = 50
    with open('model.log', 'a') as f:
        for epoch in range(EPOCHS):
            for i in tqdm(range(0, len(train_X), BATCH_SIZE)):
                batch_X = train_X[i:i+BATCH_SIZE].view(-1, 1, resize_var, resize_var).to(device)
                batch_y = train_y[i:i+BATCH_SIZE].to(device)
                          
                acc, loss = fwd_pass(batch_X, batch_y, train=True)
                if i % 50 == 0:
                    val_acc, val_loss = test(size=100)
                    f.write(f"{MODEL_NAME},{round(time.time(),3)},{round(float(acc),2)},{round(float(loss), 4)},{round(float(val_acc),2)},{round(float(val_loss),4)}\n")

REBUILD_DATA = True
resize_var = 100

IMG_SIZE = resize_var
BUTTER_MOTHS = 'training_images/butterflies-moths_smaller'
CATERPILLARS = 'training_images/caterpillars'
LABELS = {BUTTER_MOTHS: 0, CATERPILLARS: 1}
training_data = []
buttermoth_count = 0
catterpillar_count = 0

if REBUILD_DATA:
    for label in LABELS:
        print(label)
        for f in tqdm(os.listdir(label)):
            try:
                path = os.path.join(label, f)
                img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                img = cv2.resize(img, (resize_var, resize_var))
                training_data.append([np.array(img), np.eye(2)[LABELS[label]]])
        
                if label == BUTTER_MOTHS:
                    buttermoth_count += 1
                elif label == CATERPILLARS:
                    catterpillar_count += 1
                    
            except Exception as e:
                pass
            
    np.random.shuffle(training_data)
    np.save('training_data.npy', training_data)
    print('Butter_Moths: ', buttermoth_count)
    print('Caterpillars: ', catterpillar_count)

device = torch.device('cuda:1')
net = Net().to(device)

training_data = np.load('training_data.npy', allow_pickle=True)

X = torch.Tensor([i[0] for i in training_data]).view(-1, resize_var, resize_var)
X = X/255.0

y = torch.Tensor([i[1] for i in training_data])

VAL_PCT = 0.1
val_size = int(len(X)*VAL_PCT)

train_X = X[:-val_size]
train_y = y[:-val_size]

test_X = X[-val_size:]
test_y = y[-val_size:]

print(len(train_X))
print(len(test_X))

optimizer = optim.Adam(net.parameters(), lr=0.001)
loss_function = nn.MSELoss()

val_acc, val_loss = test(size=32)
print(val_acc, val_loss)

MODEL_NAME = f'model-{int(time.time())}'
print(MODEL_NAME)

train()

with open('state.pt', 'wb') as f:
    torch.save(net, 'state.pt')