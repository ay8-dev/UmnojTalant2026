import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from torchvision.models import ResNet18_Weights
import json
from pathlib import Path


# Конфигурация
DATA_DIR = "dataset"
BATCH_SIZE = 32
NUM_EPOCHS = 25
LEARNING_RATE = 0.001
IMAGE_SIZE = 224
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")

# Классы пород
CLASSES = ["limestone", "sandstone", "shale"]


def get_data_transforms():
    """Определяем аугментации и нормализацию"""
    train_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_transform


def create_model(num_classes=3):
    """Создаем модель на базе ResNet18"""
    model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    
    # Замораживаем все слои кроме последнего блока
    for param in model.parameters():
        param.requires_grad = False
    
    # Размораживаем layer4
    for param in model.layer4.parameters():
        param.requires_grad = True
    
    # Заменяем классификатор
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(model.fc.in_features, 256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, num_classes)
    )
    
    return model


def train_model():
    """Основная функция обучения"""
    print(f"Используется устройство: {DEVICE}")
    print(f"Классы: {CLASSES}")
    
    # Создаем директорию для модели
    Path("model").mkdir(exist_ok=True)
    
    # Загружаем данные
    train_transform, val_transform = get_data_transforms()
    
    train_dataset = datasets.ImageFolder(
        os.path.join(DATA_DIR, "train"),
        transform=train_transform
    )
    val_dataset = datasets.ImageFolder(
        os.path.join(DATA_DIR, "val"),
        transform=val_transform
    )
    test_dataset = datasets.ImageFolder(
        os.path.join(DATA_DIR, "test"),
        transform=val_transform
    )
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, 
                              shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, 
                           shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, 
                            shuffle=False, num_workers=4)
    
    print(f"Train: {len(train_dataset)} изображений")
    print(f"Val: {len(val_dataset)} изображений")
    print(f"Test: {len(test_dataset)} изображений")
    
    # Создаем модель
    model = create_model(num_classes=len(CLASSES)).to(DEVICE)
    
    # Функция потерь и оптимизатор
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', 
                                                      patience=3, factor=0.5)
    
    # Обучение
    best_val_acc = 0.0
    
    for epoch in range(NUM_EPOCHS):
        # Training
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
        
        train_acc = 100. * train_correct / train_total
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
        
        val_acc = 100. * val_correct / val_total
        
        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] "
              f"Train Loss: {train_loss/len(train_loader):.4f} "
              f"Train Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss/len(val_loader):.4f} "
              f"Val Acc: {val_acc:.2f}%")
        
        scheduler.step(val_loss)
        
        # Сохраняем лучшую модель
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_acc': best_val_acc,
                'classes': CLASSES
            }, 'model/rock_classifier.pth')
            print(f"  -> Сохранена лучшая модель (val_acc: {val_acc:.2f}%)")
    
    # Тестирование
    print("\n--- Тестирование на тестовом наборе ---")
    checkpoint = torch.load('model/rock_classifier.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    test_correct = 0
    test_total = 0
    class_correct = {cls: 0 for cls in CLASSES}
    class_total = {cls: 0 for cls in CLASSES}
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            _, predicted = outputs.max(1)
            
            test_total += labels.size(0)
            test_correct += predicted.eq(labels).sum().item()
            
            for label, pred in zip(labels, predicted):
                class_name = CLASSES[label]
                class_total[class_name] += 1
                if label == pred:
                    class_correct[class_name] += 1
    
    print(f"Общая точность на тесте: {100. * test_correct / test_total:.2f}%")
    print("Точность по классам:")
    for cls in CLASSES:
        acc = 100. * class_correct[cls] / class_total[cls] if class_total[cls] > 0 else 0
        print(f"  {cls}: {acc:.2f}% ({class_correct[cls]}/{class_total[cls]})")
    
    # Сохраняем метаданные
    metadata = {
        'classes': CLASSES,
        'best_val_accuracy': best_val_acc,
        'image_size': IMAGE_SIZE,
        'model_architecture': 'resnet18'
    }
    with open('model/metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print("\nОбучение завершено! Модель сохранена в model/rock_classifier.pth")


if __name__ == "__main__":
    train_model()