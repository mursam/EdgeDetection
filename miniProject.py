import cv2
import os
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score
import seaborn as sns
import matplotlib.pyplot as plt

# Initialize SIFT detector
sift = cv2.SIFT_create()

# Brute-force matcher with default parameters
bf = cv2.BFMatcher()

def process_image(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.GaussianBlur(img, (5, 5), 0)
    #img = cv2.medianBlur(img, ksize=13)
    cv2.normalize(img, img, 0, 255, cv2.NORM_MINMAX)
    edges = cv2.Canny(img, threshold1=100, threshold2=200)
    keypoints, descriptors = sift.detectAndCompute(edges, None)
    return keypoints, descriptors

def load_and_describe_images(directory_path):
    image_descriptors = {}
    image_keypoints = {}
    image_paths = {}
    for category in os.listdir(directory_path):
        category_path = os.path.join(directory_path, category)
        if os.path.isdir(category_path):
            image_descriptors[category] = []
            image_keypoints[category] = []
            image_paths[category] = []
            for img_name in os.listdir(category_path):
                if img_name.lower().endswith('.jpg') or img_name.lower().endswith('.png'):
                    img_path = os.path.join(category_path, img_name)
                    kp, des = process_image(img_path)
                    if des is not None:
                        image_descriptors[category].append(des)
                        image_keypoints[category].append(kp)
                        image_paths[category].append(img_path)
    return image_keypoints, image_descriptors, image_paths

train_keypoints, train_descriptors, train_image_paths = load_and_describe_images('/Users/muratsamanci/PycharmProjects/pythonProject2/train')
test_keypoints, test_descriptors, test_image_paths = load_and_describe_images('/Users/muratsamanci/PycharmProjects/pythonProject2/test')

# Perform the matching
predicted_labels = []
true_labels = []
matches_info = []  # To store matches for visualization

# Limit the number of matches we visualize
num_visualizations = 5

for test_category, test_desc_list in zip(test_keypoints.keys(), test_descriptors.values()):
    for idx, test_desc in enumerate(test_desc_list):
        best_match = None
        best_match_distance = float('inf')
        best_train_category = None

        for train_category, train_desc_list in train_descriptors.items():
            for train_desc in train_desc_list:
                matches = bf.knnMatch(train_desc, test_desc, k=2)
                for i, pair in enumerate(matches):
                    try:
                        m,n = pair
                        if m.distance < 0.75 * n.distance and m.distance < best_match_distance:
                            best_match = m
                            best_match_distance = m.distance
                            best_train_category = train_category
                    except ValueError :
                        pass
        if best_train_category is None:
            best_train_category = 'unknown'
        predicted_labels.append(best_train_category)
        true_labels.append(test_category)

        # Store the match information for visualization
        if len(matches_info) < num_visualizations:
            matches_info.append({
                'test_img': test_image_paths[test_category][idx],
                'train_img': train_image_paths[best_train_category][best_match.trainIdx],
                'test_kp': test_keypoints[test_category][idx],
                'train_kp': train_keypoints[best_train_category][best_match.trainIdx],
                'match': best_match
            })

# Define the set of all labels (true categories and 'unknown' for unmatched descriptors)
all_labels = set(true_labels + ['unknown'])

# Calculate the confusion matrix and metrics
cm = confusion_matrix(true_labels, predicted_labels, labels=list(all_labels))

# Calculate metrics
accuracy = accuracy_score(true_labels, predicted_labels)
precision = precision_score(true_labels, predicted_labels, average='macro', zero_division=0)
recall = recall_score(true_labels, predicted_labels, average='macro', zero_division=0)

# Plot the confusion matrix
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=all_labels, yticklabels=all_labels)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")

# Visualize the matches
for match_info in matches_info:
    test_img = cv2.imread(match_info['test_img'], cv2.IMREAD_GRAYSCALE)
    train_img = cv2.imread(match_info['train_img'], cv2.IMREAD_GRAYSCALE)

    # Extract keypoints and match
    test_kp = match_info['test_kp']
    train_kp = match_info['train_kp']
    match = match_info['match']

    # Draw the single match. Note that drawMatches expects lists of keypoints and matches.
    img_matches = cv2.drawMatches(test_img, [test_kp], train_img, [train_kp], [match], None, flags=2)

    # Show the matched image
    plt.figure(figsize=(20, 10))
    plt.imshow(img_matches, 'gray')
    plt.title('Feature Matches')
    plt.axis('off')
    plt.show()
