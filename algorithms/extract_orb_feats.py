import cv2
import json
import pandas as pd

def orb_new(image):

    # /home/nawazsha/PycharmProjects/pythonProject/data/LN84/h_004

    img = image
    img2 = img
    h, w = img.shape
    r = h / 2
    c = w / 2

    orb = cv2.ORB_create(1000)
    kp = orb.detect(img, None)
    print(len(kp))
    kp2 = []
    pt = [p.pt for p in kp]

    for i in range(0, len(kp) - 1):
        x = pt[i][0]
        y = pt[i][1]
        if r - 60 < x < r + 60 and y > c - 60 and y < c + 60:
            a = 1
        else:
            kp2.append(kp[i])
            x1 = int(x - 1)
            y1 = int(y - 1)
            x2 = int(x + 1)
            y2 = int(y + 1)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 1)

    kp2, des = orb.compute(img2, kp2)
    print(len(kp2))

    img2 = cv2.drawKeypoints(img2, kp2, None, color=(0, 255, 0), flags=0)
    x=[]
    y=[]
    intensity=[]
    print('hello :)')
    for each_kp in kp:
        x.append("{:.2f}".format(each_kp.pt[0]))
        y.append("{:.2f}".format(each_kp.pt[1]))
        intensity.append(json.dumps(image[int(each_kp.pt[1])][int(each_kp.pt[0])].tolist()))
    df = pd.DataFrame(list(zip(x,y,intensity)), columns=["x","y","intensity"])
    return [img2, len(kp2), df]