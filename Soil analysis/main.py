from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define dataset path
data_dir = r"C:/Users/ahsha/Desktop/mp/Soil types"

# Set image size and batch size
image_size = (224, 224)
batch_size = 32

# Data augmentation & normalization (for training)
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

# Only normalization for validation (NO augmentation)
val_datagen = ImageDataGenerator(rescale=1./255)

# Load training data
train_data = train_datagen.flow_from_directory(
    data_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=True  # Shuffling for better generalization
)

# Load validation data
val_data = val_datagen.flow_from_directory(
    data_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False  # No shuffle for validation
)

# Print class labels
print("Class Indices:", train_data.class_indices)
