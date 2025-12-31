import matplotlib.pyplot as plt

# These lists should be filled during training
train_losses = []
val_accuracies = []
epochs = []

# Load metrics from file
with open('metrics_log.txt', 'r') as f:
    for line in f:
        epoch, loss, acc = line.strip().split(',')
        epochs.append(int(epoch))
        train_losses.append(float(loss))
        val_accuracies.append(float(acc))

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(epochs, train_losses, marker='o')
plt.title('Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')

plt.subplot(1, 2, 2)
plt.plot(epochs, val_accuracies, marker='o', color='green')
plt.title('Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')

plt.tight_layout()
plt.show()
