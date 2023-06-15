import cv2

# Load the pre-trained face cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load the image
img = cv2.imread("Image.jpg")
image=cv2.resize(img,(500,350))


# Convert the image to grayscale
gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

# Detect faces in the grayscale image
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5,minSize=(30, 30))

#Draw rectangles around the faces
for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 3)

# Display the output
cv2.imwrite('After Detection.jpg',image)
cv2.imshow('After Detection', image)
cv2.waitKey(0)
cv2.destroyAllWindows()