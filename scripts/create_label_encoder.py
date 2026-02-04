import pickle
from sklearn.preprocessing import LabelEncoder

# EXACT class names used during training
classes = []

# Consonants (1–36)
for i in range(1, 37):
    classes.append(f"consonants_{i}")

# Vowels (1–11)
for i in range(1, 12):
    classes.append(f"vowels_{i}")

# Numerals (0–9)
for i in range(0, 10):
    classes.append(f"numerals_{i}")

# Create and fit encoder
le = LabelEncoder()
le.fit(classes)

# Save encoder
with open("model/label_encoder.pkl", "wb") as f:
    pickle.dump(le, f)

print("Label encoder saved successfully!")
print("Total classes:", len(le.classes_))