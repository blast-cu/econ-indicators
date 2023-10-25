
import torch
import torch.nn as nn
import transformers

# Load pre-trained RoBERTa model
roberta_model = transformers.RobertaModel.from_pretrained('roberta-base')

# Add classification layer on top of RoBERTa model
num_classes = 3
classification_layer = nn.Linear(roberta_model.config.hidden_size, num_classes)
model = nn.Sequential(roberta_model, classification_layer)

# Load training, validation, and test data
train_data = ...
val_data = ...
test_data = ...

# Define loss function and optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

# Train the model
num_epochs = 10
for epoch in range(num_epochs):
    for input_ids, attention_mask, labels in train_data:
        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask=attention_mask)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()

# Evaluate the model on the validation data
with torch.no_grad():
    num_correct = 0
    num_total = 0
    for input_ids, attention_mask, labels in val_data:
        outputs = model(input_ids, attention_mask=attention_mask)
        _, predicted = torch.max(outputs, dim=1)
        num_correct += (predicted == labels).sum().item()
        num_total += len(labels)
    accuracy = num_correct / num_total
    print('Validation accuracy:', accuracy)

# Test the model on the test data
with torch.no_grad():
    num_correct = 0
    num_total = 0
    for input_ids, attention_mask, labels in test_data:
        outputs = model(input_ids, attention_mask=attention_mask)
        _, predicted = torch.max(outputs, dim=1)
        num_correct += (predicted == labels).sum().item()
        num_total += len(labels)
    accuracy = num_correct / num_total
    print('Test accuracy:', accuracy)
