import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Input
from tensorflow.keras.optimizers import Adam
import numpy as np

# Step 1: Load the pre-trained VGG16 model (without the top layers)
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Step 2: Copy specific layers from the base model (e.g., first 3 convolutional blocks)
# We'll extract layers up to 'block3_pool'
copied_layers = []
for layer in base_model.layers:
    copied_layers.append(layer)
    if layer.name == 'block3_pool':
        break

# Step 3: Create a new model architecture
input_layer = Input(shape=(224, 224, 3))
x = input_layer
for layer in copied_layers:
    # Ensure layers are not trainable (frozen)
    layer.trainable = False
    x = layer(x)

# Add new layers for the custom architecture
x = Flatten()(x)
x = Dense(256, activation='relu')(x)
x = Dense(128, activation='relu')(x)
output_layer = Dense(10, activation='softmax')(x)  # Assuming 10 classes for classification

# Create the new model
new_model = Model(inputs=input_layer, outputs=output_layer)

# Step 4: Compile the model
new_model.compile(optimizer=Adam(learning_rate=0.0001), 
                 loss='categorical_crossentropy', 
                 metrics=['accuracy'])

# Step 5: Prepare dummy data for demonstration (replace with real data)
# Dummy data: 100 images of shape (224, 224, 3) and 10 classes
x_train = np.random.rand(100, 224, 224, 3)
y_train = tf.keras.utils.to_categorical(np.random.randint(10, size=(100,)), num_classes=10)

# Step 6: Train the model (fine-tuning)
new_model.fit(x_train, y_train, epochs=5, batch_size=32, validation_split=0.2)

# Optional: Print model summary to verify architecture
new_model.summary()

# Optional: Unfreeze some layers for further fine-tuning if needed
# for layer in new_model.layers[:10]:
#     layer.trainable = True
# new_model.compile(optimizer=Adam(learning_rate=0.00001), 
#                  loss='categorical_crossentropy', 
#                  metrics=['accuracy'])
# new_model.fit(x_train, y_train, epochs=5, batch_size=32, validation_split=0.2)