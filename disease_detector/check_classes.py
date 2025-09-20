import os

# If your classes are directly under PlantVillage
train_dir = "PlantVillage"  

classes = sorted(os.listdir(train_dir))
print("Classes:", classes)
