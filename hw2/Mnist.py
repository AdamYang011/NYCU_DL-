import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split
import os

class ConvNet(nn.Module):
    def __init__(self, stride1, kernel_size1, stride2, kernel_size2):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=kernel_size1, stride=stride1)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=kernel_size2, stride=stride2)
        self.fc1 = nn.Linear(20, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        in_size = x.size(0)
        out = self.conv1(x)
        out = F.relu(out)
        out = F.max_pool2d(out, 2, 2)
        out = self.conv2(out)
        out = F.relu(out)
        out = out.view(in_size, -1)
        out = self.fc1(out)
        out = F.relu(out)
        out = self.fc2(out)
        out = F.log_softmax(out, dim=1)
        return out

def train(model, device, train_loader, optimizer, epoch, train_losses):
    model.train()
    epoch_loss = 0.0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)

        # Add L2 regularization to the loss
        l2_lambda = 0.01
        l2_reg = torch.tensor(0., requires_grad=True)
        for param in model.parameters():
            l2_reg = l2_reg + torch.norm(param)
        loss = loss + l2_lambda * l2_reg
        
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        if (batch_idx + 1) % 30 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
    avg_epoch_loss = epoch_loss / len(train_loader.dataset)
    train_losses.append(avg_epoch_loss)

def test(model, device, test_loader, misclassified_images):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()

            misclassified_mask = pred.eq(target.view_as(pred)) == False
            if misclassified_mask.any():
                misclassified_data = data[misclassified_mask]
                misclassified_target = target[misclassified_mask.view(-1)][0]  # Select the first element
                misclassified_pred = pred[misclassified_mask]

                misclassified_images.append((misclassified_data, misclassified_target, misclassified_pred))

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

    return misclassified_images

def evaluate(model, device, loader, dataset_type):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

    accuracy = correct / total
    print(f'{dataset_type} Accuracy: {100 * accuracy:.2f}%')
    return accuracy

def plot_weight_bias_histograms(model, outputdir):
    for name, param in model.named_parameters():
        if 'weight' in name:
            plt.figure(figsize=(8, 6))
            plt.title(f'{name} Histogram')
            plt.hist(param.data.cpu().numpy().flatten(), bins=50)
            plt.xlabel('Value')
            plt.ylabel('Frequency')
            plt.savefig(outputdir + '/' + name + '_histogram.png')
            plt.show()

def plot_learning_curve(train_losses, train_accuracies, val_accuracies, test_accuracies, outputdir):
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(train_losses) + 1), train_losses, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(outputdir + '/learning_curve.png')
    plt.show()


    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(train_accuracies) + 1), train_accuracies, label='Training Accuracy')
    plt.plot(range(1, len(val_accuracies) + 1), val_accuracies, label='Validation Accuracy')
    plt.plot(range(1, len(test_accuracies) + 1), test_accuracies, label='Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(outputdir + '/accuracy.png')
    plt.show()


def visualize_misclassified(misclassified_images, outputdir, num=10):
    fig = plt.gcf()
    fig.set_size_inches(12, 14)
    if num > 25:
        num = 25
    fig.subplots_adjust(hspace=0.5)
    for i in range(0, num):
        ax = plt.subplot(5, 5, 1 + i)
        ax.imshow(misclassified_images[i][0][0].cpu().numpy(), cmap='binary')  # Assuming images are tensors
        title = "True: {}".format(str(misclassified_images[i][1].cpu().numpy()))
        title += "\nPredicted: {}".format(str(misclassified_images[i][2][0].cpu().numpy()))
        ax.set_title(title, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
    plt.savefig(outputdir + '/miss.png')
    plt.show()


def get_interested_layers(model, data, interested_layers):
    activations = {layer_name: [] for layer_name in interested_layers}
    
    def hook_fn(layer_name):
        def hook(module, input, output):
            activations[layer_name].append(output)
        return hook
    
    hooks = []
    for layer_name, layer in model.named_children():
        if layer_name in interested_layers:
            hook = layer.register_forward_hook(hook_fn(layer_name))
            hooks.append(hook)
    
    with torch.no_grad():
        model(data)
    
    for hook in hooks:
        hook.remove()
    
    return activations

def visualize_feature_maps(feature_maps, num_rows, num_cols):
    for layer_name, layer_maps in feature_maps.items():
        plt.figure(figsize=(12, 6))
        plt.suptitle(f'Feature Maps for Layer: {layer_name}')
        for i in range(min(num_rows * num_cols, len(layer_maps))):
            plt.subplot(num_rows, num_cols, i + 1)
            feature_map_np = layer_maps[i][0].cpu().numpy()
            for channel in range(feature_map_np.shape[0]):
                plt.imshow(feature_map_np[channel], cmap='viridis', alpha=0.5)
            plt.title(f'Channel {i}')
            plt.axis('off')
        plt.show()

def visualize_correct_predictions(correct_images, num_rows, num_cols):
    plt.figure(figsize=(12, 14))
    plt.suptitle('Correctly Classified Images')
    for i, (image, true_label, predicted_label) in enumerate(correct_images[:num_rows * num_cols]):
        plt.subplot(num_rows, num_cols, i + 1)
        plt.imshow(image[0].cpu().numpy(), cmap='binary')
        title = f'True: {true_label}, Predicted: {predicted_label[0]}'
        plt.title(title, fontsize=10)
        plt.axis('off')
    plt.show()

if __name__ == '__main__':
    BATCH_SIZE = 512
    EPOCHS = 20
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    full_train_dataset = datasets.MNIST('data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('data', train=False, download=True, transform=transform)

    train_size = 55000
    val_size = 5000
    test_size = 10000
    stride1 = 1
    kernel_size1 = 5
    stride2 = 1
    kernel_size2 = 3
    outputdir = "./s" + str(stride1) + "_k" + str(kernel_size1) + "_s" + str(stride2) + "_k" + str(kernel_size2)
    if not os.path.exists(outputdir):
        os.makedirs(outputdir)

    train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = ConvNet(stride1, kernel_size1, stride2, kernel_size2)
    optimizer = optim.Adam(model.parameters())

    train_losses = []
    train_accuracies = []
    val_accuracies = []
    test_accuracies = []
    misclassified_images = []

    for epoch in range(1, EPOCHS + 1):
        train(model, DEVICE, train_loader, optimizer, epoch, train_losses)
        train_acc = evaluate(model, DEVICE, train_loader, 'Training')
        val_acc = evaluate(model, DEVICE, val_loader, 'Validation')
        test_acc = evaluate(model, DEVICE, test_loader, 'Test')
        misclassified_images = test(model, DEVICE, test_loader, misclassified_images)

        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)
        test_accuracies.append(test_acc)

    plot_learning_curve(train_losses, train_accuracies, val_accuracies, test_accuracies, outputdir)
    plot_weight_bias_histograms(model, outputdir)
    visualize_misclassified(misclassified_images, outputdir)

    correct_images = []
    for data, target in test_loader:
        data, target = data.to(DEVICE), target.to(DEVICE)
        output = model(data)
        pred = output.max(1, keepdim=True)[1]
        correct_mask = pred.eq(target.view_as(pred)).squeeze()
        correct_data = data[correct_mask]
        correct_target = target[correct_mask]
        correct_pred = pred[correct_mask]
        correct_images.extend(list(zip(correct_data, correct_target, correct_pred)))

        if len(correct_images) >= 8:
            break

    visualize_correct_predictions(correct_images, num_rows=2, num_cols=4)

    image_index = 0
    image, label = test_dataset[image_index]

    image = image.unsqueeze(0).to(DEVICE)

    interested_layers = ['conv1', 'conv2', 'fc1', 'fc2']
    feature_maps = get_interested_layers(model, image, interested_layers)

    visualize_feature_maps(feature_maps, num_rows=1, num_cols=5)
