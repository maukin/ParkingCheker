import cv2
import pytesseract
import numpy as np

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe'

cascade = cv2.CascadeClassifier("haarcascade_russian_plate_number.xml")


def extract_num(img_name):
    global read
    img = cv2.imread(img_name) ## Reading Image
    # Converting into Gray
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # Detecting plate
    nplate = cascade.detectMultiScale(gray,1.2,4)

    for (x,y,w,h) in nplate:
        # Crop a portion of plate
        a,b = (int(0.02*img.shape[0]), int(0.025*img.shape[1]))
        plate = img[y+a:y+h-a, x+b:x+w-b, :]
        # make image more darker to identify the LPR

        ## iMAGE PROCESSING
        kernel = np.ones((1, 1), np.uint8)
        plate = cv2.dilate(plate, kernel, iterations=1)
        plate = cv2.erode(plate, kernel, iterations=1)
        plate_gray = cv2.cvtColor(plate,cv2.COLOR_BGR2GRAY)
        (thresh, plate) = cv2.threshold(plate_gray, 150, 255, cv2.THRESH_BINARY)  #127, 255
        # Feed Image to OCR engine

        config = r'--oem 3 --psm 6'
        read = pytesseract.image_to_string(plate, config=config)
        # read = ''.join(e for e in read if e.isalnum())
        # print(read)
        # stat = read[0:2]
        # try:
        # # Fetch the State information
        #     print('Car Belongs to',states[stat])
        # except:
        #     print('State not recognised!!')
        print(read)
        cv2.rectangle(img, (x,y), (x+w, y+h), (51,51,255), 2)
        cv2.rectangle(img, (x, y - 40), (x + w, y),(51,51,255) , -1)
        cv2.putText(img,read, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imshow('PLate',plate)
        # Save & display result image
        cv2.imwrite('plate.jpg', plate)

    cv2.imshow("Result", img)
    cv2.imwrite('result.jpg',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Let's make a function call
extract_num('./photos/11.jpg')


