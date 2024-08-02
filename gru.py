import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import *
from keras.preprocessing.sequence import pad_sequences
from keras.datasets import imdb
import matplotlib.pyplot as plt
import numpy as np

max_words = 20000  # imdb’s vocab_size 即词汇表大小
max_len = 200      # max length

batch_size = 32
emb_size = 128   # embedding size
hid_size = 128   # lstm hidden size
num_epochs = 20
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_words)
x_train = pad_sequences(x_train, maxlen=max_len)
x_test = pad_sequences(x_test, maxlen=max_len)

train_data = TensorDataset(torch.LongTensor(x_train), torch.LongTensor(y_train))
test_data = TensorDataset(torch.LongTensor(x_test), torch.LongTensor(y_test))

train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size=batch_size)


class GRU(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super(GRU, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.gru = nn.GRU(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.embedding(x)
        _, h_n = self.gru(x)
        x = self.fc(h_n.squeeze(0))
        return x


model = GRU(max_words, emb_size, hid_size, 1).to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.00005)

train_loss = []
test_loss = []
train_acc = []
test_acc = []
best_acc = 0.0
best_epoch = 0

for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0
    epoch_correct = 0
    epoch_total = 0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        loss = criterion(outputs.squeeze(), labels.float())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item() * inputs.size(0)
        predictions = torch.round(torch.sigmoid(outputs.squeeze()))
        epoch_correct += (predictions == labels).sum().item()
        epoch_total += labels.size(0)

    cur_loss = epoch_loss / epoch_total
    train_accuracy = epoch_correct / epoch_total
    train_loss.append(cur_loss)
    train_acc.append(train_accuracy)

    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {cur_loss:.4f}, Accuracy: {train_accuracy * 100:.2f}%')

    # 测试模型
    model.eval()
    t_loss = 0
    correct, total = 0, 0
    class_correct = [0, 0]
    class_total = [0, 0]
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            predictions = torch.round(torch.sigmoid(outputs.squeeze()))

            loss = criterion(outputs.squeeze(), labels.float())
            t_loss += loss.item() * labels.size(0)
            total += labels.size(0)
            correct += (predictions == labels).sum().item()

            for i in range(labels.size(0)):
                label = labels[i].item()
                class_correct[label] += (predictions[i].item() == label)
                class_total[label] += 1

    accuracy = correct / total
    test_acc.append(accuracy)
    test_loss.append(t_loss / total)
    if accuracy > best_acc:
        best_epoch = epoch + 1
        best_acc = accuracy

    print(f'Epoch {epoch + 1}/{num_epochs}, Test Accuracy: {accuracy * 100:.2f}%')
    print(f'Epoch {epoch + 1}/{num_epochs}, Accuracy of neg: {class_correct[0] / class_total[0] * 100:.2f}%')
    print(f'Epoch {epoch + 1}/{num_epochs}, Accuracy of pos: {class_correct[1] / class_total[1] * 100:.2f}%')

print("-----------------------------------------------")
print(f'Best Epoch {best_epoch}, Best Acc: {best_acc * 100:.2f}%')
epochs = range(1, num_epochs + 1)
plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
plt.plot(epochs, train_loss, 'bo-', label='Train loss', markersize=1)
plt.plot(epochs, test_loss, 'ro-', label='Test loss', markersize=1)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Train and Test Loss')
plt.xticks(np.arange(1, num_epochs + 1))
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(epochs, train_acc, 'bo-', label='Train accuracy', markersize=1)
plt.plot(epochs, test_acc, 'ro-', label='Test accuracy', markersize=1)
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Train and Test Accuracy')
plt.xticks(np.arange(1, num_epochs + 1))
plt.legend()

plt.show()


