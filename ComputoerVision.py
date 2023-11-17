import cv2
import os
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score
import seaborn as sns
import matplotlib.pyplot as plt


sift = cv2.SIFT_create()

bf = cv2.BFMatcher()



def preprocessing(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.GaussianBlur(img,(5,5),3)
    cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
    edges = cv2.Canny(img, threshold1=100, threshold2=200)
    kps, dsc = sift.detectAndCompute(edges, None)
    return kps, dsc
def getdescripters(rootpath):
    image_dsc = {}
    image_kps = {}
    image_paths = {}
    for category in os.listdir(rootpath):
        category_path = os.path.join(rootpath, category)
        if os.path.isdir(category_path):
            image_dsc[category] = []
            image_kps[category] = []
            image_paths[category] = []
            for img_name in os.listdir(category_path):
                if img_name.lower().endswith('.jpg') or img_name.lower().endswith('.png'):
                    img_path = os.path.join(category_path, img_name)
                    kp, des = preprocessing(img_path)
                    if des is not None:
                        image_dsc[category].append(des)
                        image_kps[category].append(kp)
                        image_paths[category].append(img_path)
    return image_kps, image_dsc, image_paths


train_kp, train_dsc = getdescripters('/Users/muratsamanci/PycharmProjects/pythonProject2/train')
test_kp, test_dsc, test_image_paths = getdescripters('/Users/muratsamanci/PycharmProjects/pythonProject2/test')


predicted_labels = []
true_labels = []

for test_category, test_desc_list in test_dsc.items():
    for test_desc in test_desc_list:

        best_match = None
        best_match_distance = float('inf')
        best_train_category = None

        for train_category, train_desc_list in train_dsc.items():
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


all_labels = set(true_labels + ['unknown'])


cm = confusion_matrix(true_labels, predicted_labels, labels=list(all_labels))



accuracy = accuracy_score(true_labels, predicted_labels)
precision = precision_score(true_labels, predicted_labels, average='macro', zero_division=0)
recall = recall_score(true_labels, predicted_labels, average='macro', zero_division=0)


plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=train_dsc.keys(),
            yticklabels=train_dsc.keys())
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
