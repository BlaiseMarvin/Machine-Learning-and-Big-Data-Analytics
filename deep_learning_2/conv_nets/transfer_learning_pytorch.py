model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

'''
Truncate / remove some layers

In PyTorch, models are just Python classes — so you can modify them like normal.

For example, ResNet’s classification head is stored as model.fc
'''
print(model.fc)
# Linear(in_features=512, out_features=1000, bias=True)
'''
    You can replace it to match your dataset’s number of classes:
'''
num_classes = 5  # your dataset
model.fc = nn.Linear(512, num_classes)

'''
    Truncate deeper layers (optional)

Say you want to cut off layers before the head — e.g., keep only the feature extractor part.
For ResNet, you can use:
'''

feature_extractor = nn.Sequential(*list(model.children())[:-1])

'''
    Now feature_extractor(x) outputs the last convolutional feature maps (before the final linear classifier).
'''

'''
    Add new layers

You can wrap the truncated model in your own custom network:
'''
class CustomResNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        base = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.features = nn.Sequential(*list(base.children())[:-1])  # remove fc
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

model = CustomResNet(num_classes=5)

'''
Freeze pretrained layers

This is crucial:
We want to keep the learned low-level features (edges, textures) fixed,
and train only the new classifier head.
'''
for param in model.features.parameters():
    param.requires_grad = False


'''
Choose optimizer only for trainable params

To be safe and efficient, only pass parameters that require gradients:
'''
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)

'''
    Gradually unfreeze later

A common fine-tuning trick is “gradual unfreezing” — first train only the new layers, then unfreeze some deeper layers to fine-tune the feature extractor slightly.
'''
# After a few epochs:
for name, param in model.features.named_parameters():
    if "layer4" in name:  # unfreeze the last ResNet block
        param.requires_grad = True

'''
Normal training loop

After that, your training loop is standard PyTorch:
'''
criterion = nn.CrossEntropyLoss()

for epoch in range(epochs):
    model.train()
    for images, labels in dataloader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
