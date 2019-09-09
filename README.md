# Handwritten Mathematical Expressions Recognition
### Khiem T. Do, Phuong T.M. Chu & Santhos

This is the project that we finished after the 8th week of studying **Machine Learning** from scratch.
<p align="center">
  <img width="860" height="400" src="https://cdn.discordapp.com/attachments/603841721222037505/620517992186249226/unknown.png">
</p>

## INTRODUCTION
### 1. Handwritten Mathematical Expressions Recognition
If you've ever had to typeset mathematical expressions, you might have thought: wouldn’t it be great if I could just take a picture of a handwritten expression and have it be recognized automatically? Can you use computer vision to recognize handwritten mathematical expressions? In this project, we will prove that we can!

### 2. Project goals
- Building models to classify **Handwritten Mathematical Expressions** (Numbers and Operators) images.

- Making a **Web Flask application** so user can upload the images and perform calculation.

### 3. Project plan
Main steps:

|Task|Progress|
|:-----------------------------|:------------:|
|Data collection |✔|
|Data preprocessing |✔|
|Building Model|✔|
|Building Flask App|✔|
|Deployment to Google Cloud|✔|

## COLLECTING DATA
### 1. The Handwritten math symbols dataset
[The Kaggle Dataset](https://www.kaggle.com/xainano/handwrittenmathsymbols) consists of `jpg` files (45x45). It includes:
- Basic Greek alphabet symbols like: `alpha`, `beta`, `gamma`, `mu`, `sigma`, `phi` and `theta`
- English alphanumeric symbols
- All math operators and set operators
- Basic pre-defined math functions like: `log`, `lim`, `cos`, `sin`, `tan`
- Math symbols like: `\int`, `\sum`, `\sqrt`, `\delta` and more.

For simplicity, in this project, we only take into account 4 math operators: `plus`,`minus`,`addition`, and `division`.

### 2. Data Collection process 
During the data cleanning process, we can see that it is biased for some of the digits/symbols, as it contains 12.000 images for some symbol and 3.000 images for others. To remove this bias, reduce the number of images in each folder to approximately 4.000.

## BUILDING MODELS
You can have more details by walking through our 2 Jupyter notebooks.

In this project, we only built the model for recognizing math operators. We used prebuilt handwritten number recognition model `handwritten_model.h5` from other project [(MNIST project)](). 

### 1. Extracting Features

We can use contour extraction to obtain features.
- Invert the image and then convert it to a binary image because contour extraction gives the best result when the object is white, and surrounding is black.
- To find contours use `findContour` function. For features, obtain the bounding rectangle of contour using `boundingRect` function (Bounding rectangle is the smallest horizontal rectangle enclosing the entire contour).
- Since each image in our dataset contains only one symbol/digit, we only need the bounding rectangle of maximum size. For this purpose, we calculate the area of the bounding rectangle of each contour and select the rectangle with maximum area.
- We esize the maximum area bounding rectangle to `28` by `28`. Reshape it to `784` by `1`. So there were `784-pixel` values or features. We gave the corresponding label to it . So our dataset contained `784` features column and one label column.
- After extracting features, save the data to a `csv` file.

### 2. Building and training model
We decided to build a **LinearSVC model** for this problem.

```python
from sklearn.svm import LinearSVC

# Create an linear SVM object
clf = LinearSVC()

# Perform the training
clf.fit(X_train_hog, y_train)

# Save the classifier
joblib.dump(clf, "model_cls_operator.pkl", compress=3)
```
### 3. Model performance summary
![](https://i.imgur.com/gLsJMDw.png)

We successfully built **a Linear SVC model** with **accuracy of 94.88%** (for predicting math operators) which can be used together with a previously built model to classify Handwritten Mathematical Expressions (Numbers and Operators) images.

### 4. Making prediction and recognition

We perform our prediction with the assumption that we only have the calculation consists of 1 digit numbers, and one of 4 math operators (`+`, `-`, `*`, `/`) followed after the number, no additional math symbols are used. 
For example: `1 * 2 - 3 + 4`

```python
plt.figure(figsize=(10,4))
s=""

for index, (c, _) in enumerate(cnts):
    (x, y, w, h) = cv2.boundingRect(c)

    if w >=7 and h>=20:
        roi = gray[y:y+h, x:x+w]
        thresh = roi.copy()
        T = mahotas.thresholding.otsu(roi)
        thresh[thresh > T] = 255
        thresh = cv2.bitwise_not(thresh)

        thresh_digit = deskew(thresh, 28)
        thresh_digit = center_extent(thresh_digit, (28,28))
        thresh_operator = deskew(thresh, 28)
        thresh_operator = center_extent(thresh_operator, (28,28))

        predictions_digit = model_digit.predict(np.expand_dims(thresh_digit, axis=0))
        predictions_operator = model_operator.predict(extract_hog(np.reshape(thresh_operator, (1, -1))))
        
        digits = np.argmax(predictions_digit[0])

        cv2.rectangle(image, (x,y), (x+w, y+h), (0,255,0), 2)
        
        if index % 2 == 0:
            cv2.putText(image, str(digits), (x,y+70), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,255,0), 2)
            s = s+str(digits)
        else:
            cv2.putText(image, labels_name[predictions_operator[0]], (x+10,y-10), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,255,0), 2)
            s = s+labels_name[predictions_operator[0]]
            
plt.imshow(imutils.opencv2matplotlib(image))
plt.show()
print("The result is:",eval(s))
```

Example of output

![](https://i.imgur.com/Wxou2t8.png)

## BUILDING THE FLASK APP
<p align="center">
  <img width="860" height="400" src="https://cdn.discordapp.com/attachments/603841721222037505/620526625297137664/unknown.png">
</p>

### How to run the Flask App locally
```
virtualenv env
source env/bin/activate

# For window
set FLASK_APP=app.py
set FLASK_ENV=development
export FLASK_DEBUG=1
flask run

# For Ubuntu
export FLASK_APP=app.py
export FLASK_ENV=development
export FLASK_DEBUG=1
flask run
```
### Deploy the App to Google Cloud Platform

```
gcloud app deploy
```

## CONCLUSION
We successfully built 2 models which are used together to classify **Handwritten Mathematical Expressions** (Numbers and Operators) images with high accuracy of **99.49%** (for predicting numbers [model from other project]()) and accuracy of **94.88%** (for predicting math operators).

In addition, we also **built a Flask application** so user user can upload the images and perform calculation.
