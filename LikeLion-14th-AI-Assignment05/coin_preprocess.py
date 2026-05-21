import numpy as np, cv2

def preprocessing():
    fname = "18.png"
    image = cv2.imread(fname,cv2.IMREAD_COLOR)
    if image is None: return None, None

    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray,(7,7),2,2)
    flag = cv2.THRESH_BINARY + cv2.THRESH_OTSU
    _,th_ing = cv2.threshold(gray,130,255,flag)

    mask = np.ones((3,3),np.uint8)
    th_ing = cv2.morphologyEx(th_ing,cv2.MORPH_OPEN,mask)

    return image,th_ing

def find_coins(image):    
    results = cv2.findContours(image,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    contours = results[0] if int (cv2.__version__[0]) >= 4 else results[1]

    circles = [cv2.minEnclosingCircle(c) for c in contours]
    circles = [(tuple(map(int,center)),int(radius))
               for center,radius in circles if radius>25]
    return circles
    