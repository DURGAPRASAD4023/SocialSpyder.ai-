import os
import time
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50, MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Concatenate, Dense, Dropout, LSTM, Reshape, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from transformers import SwinModel, ViTModel
import torch
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import warnings

# Suppress future warnings from huggingface_hub
warnings.filterwarnings("ignore", category=FutureWarning)
# Suppress other warnings
warnings.filterwarnings("ignore", category=UserWarning)
# Suppress TensorFlow deprecated warnings
tf.get_logger().setLevel('ERROR')

# Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Dataset directory
base_dir = "plants_d"

# Extract class names
CLASS_NAMES = sorted([d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))])
print(f"Detected classes: {CLASS_NAMES}")
num_classes = len(CLASS_NAMES)

# Data augmentation
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
datagen = ImageDataGenerator(
    rescale=1.0/255,
    validation_split=0.2,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Data generators
train_gen = datagen.flow_from_directory(
    base_dir, 
    target_size=IMG_SIZE, 
    batch_size=BATCH_SIZE, 
    class_mode='binary' if num_classes == 1 else 'categorical',  # Fix: Use binary for single class
    subset='training', 
    shuffle=True
)

val_gen = datagen.flow_from_directory(
    base_dir, 
    target_size=IMG_SIZE, 
    batch_size=BATCH_SIZE, 
    class_mode='binary' if num_classes == 1 else 'categorical',  # Fix: Use binary for single class
    subset='validation', 
    shuffle=False
)

# Debug print
print("Sample validation labels (first batch):", val_gen[0][1][:5])
print("Label shape:", val_gen[0][1].shape)

# Reset generators after checking
train_gen.reset()
val_gen.reset()

# Load Swin Transformer and ViT models (PyTorch)
print("Loading transformer models...")
swin_model = SwinModel.from_pretrained("microsoft/swin-tiny-patch4-window7-224").to(device)
vit_model = ViTModel.from_pretrained("google/vit-base-patch16-224", add_pooling_layer=False).to(device)

# Function to extract Transformer features batch-wise
def extract_transformer_features(data_gen, desc=""):
    all_swin_features, all_vit_features, all_labels = [], [], []
    total_batches = len(data_gen)
    
    print(f"Extracting {desc} features from {total_batches} batches...")
    
    for i, (batch_images, batch_labels) in enumerate(data_gen):
        print(f"Processing batch {i+1}/{total_batches}...")
        batch_images_torch = torch.tensor(batch_images).permute(0, 3, 1, 2).float().to(device)
        
        with torch.no_grad():
            swin_output = swin_model(batch_images_torch).last_hidden_state.mean(dim=1).cpu().numpy()
            vit_output = vit_model(batch_images_torch).last_hidden_state.mean(dim=1).cpu().numpy()
        
        all_swin_features.append(swin_output)
        all_vit_features.append(vit_output)
        all_labels.append(batch_labels)
        
        # Break after processing all data
        if i+1 >= total_batches:
            break
    
    return np.vstack(all_swin_features), np.vstack(all_vit_features), np.vstack(all_labels)

# Extract Transformer features
print("Extracting transformer features...")
swin_features_train, vit_features_train, train_labels = extract_transformer_features(train_gen, "training")
swin_features_val, vit_features_val, val_labels = extract_transformer_features(val_gen, "validation")

# Reset generators again
train_gen.reset()
val_gen.reset()

# Save the extracted features
print("Saving extracted features...")
np.save("swin_features_train.npy", swin_features_train)
np.save("vit_features_train.npy", vit_features_train)
np.save("swin_features_val.npy", swin_features_val)
np.save("vit_features_val.npy", vit_features_val)
np.save("train_labels.npy", train_labels)
np.save("val_labels.npy", val_labels)

# Image input layer
image_input = Input(shape=(224, 224, 3))

# ResNet50 feature extractor
print("Building CNN model...")
resnet_base = ResNet50(weights='imagenet', include_top=False, input_tensor=image_input)
resnet_base.trainable = False
resnet_features = GlobalAveragePooling2D()(resnet_base.output)

# MobileNetV2 feature extractor
mobilenet_base = MobileNetV2(weights='imagenet', include_top=False, input_tensor=image_input)
mobilenet_base.trainable = False
mobilenet_features = GlobalAveragePooling2D()(mobilenet_base.output)

# Concatenating features from CNN models
concatenated_features = Concatenate()([resnet_features, mobilenet_features])

# LSTM + ANN Model
x = Reshape((1, concatenated_features.shape[1]))(concatenated_features)
x = LSTM(128, return_sequences=True)(x)
x = LSTM(64, return_sequences=False)(x)
x = Dense(64, activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(32, activation='relu')(x)
x = Dropout(0.5)(x)

# Fix: Use appropriate output layer and loss function based on number of classes
if num_classes == 1:
    output_layer = Dense(1, activation='sigmoid')(x)
    loss_function = 'binary_crossentropy'
else:
    output_layer = Dense(num_classes, activation='softmax')(x)
    loss_function = 'categorical_crossentropy'

# Final model
final_model = Model(inputs=image_input, outputs=output_layer)

# Compile with correct loss function
final_model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss=loss_function,
    metrics=['accuracy']
)

# Callbacks
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-7, verbose=1)
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1)

# Training
EPOCHS = 10
print("Training model...")
start_time = time.time()
history = final_model.fit(
    train_gen,
    epochs=EPOCHS,
    validation_data=val_gen,
    callbacks=[reduce_lr, early_stopping],
    verbose=1
)
end_time = time.time()
print(f"Training Time: {(end_time - start_time) // 60} min {int((end_time - start_time) % 60)} sec")

# Model evaluation
val_loss, val_accuracy = final_model.evaluate(val_gen, verbose=1)
print(f"Validation Accuracy: {val_accuracy * 100:.2f}%")

# Predictions
y_pred = final_model.predict(val_gen, verbose=1)

# Handle predictions based on output type
if num_classes == 1:
    y_pred_classes = (y_pred > 0.5).astype(int).flatten()
    y_true = val_labels.flatten().astype(int)
else:
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true = np.argmax(val_labels, axis=1)

# Classification Report
print("Classification Report:")
print(classification_report(y_true, y_pred_classes, target_names=CLASS_NAMES))

# Confusion Matrix - ensure all labels are included
all_classes = list(range(len(CLASS_NAMES)))
cm = confusion_matrix(y_true, y_pred_classes, labels=all_classes)
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=CLASS_NAMES, 
            yticklabels=CLASS_NAMES)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.savefig('confusion_matrix.png')
plt.show()

# Plot training history
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.savefig('training_history.png')
plt.show()

# Save the model
final_model.save("Plant_Disease_Detection_Model.keras")
print("Model saved successfully!")