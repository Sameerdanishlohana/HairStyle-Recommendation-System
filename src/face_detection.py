import cv2

model_path="models/face_detection_yunet_2023mar.onnx"
img=cv2.imread("data/raw_faces/hadeer.png")

if img is None:
    raise ValueError("Image not found!")

h,w,_=img.shape

detector=cv2.FaceDetectorYN.create(model_path,"",(w,h),score_threshold=0.9,nms_threshold=0.3,top_k=5000)
_,faces=detector.detect(img)

if faces is not None:
    for face in faces:
        x,y,fw,fh=map(int,face[:4])
        cv2.rectangle(img,(x,y),(x+fw,y+fh),(0,255,0),2)

cv2.imshow("Face Detection - YuNet",img)
cv2.waitKey(0)
cv2.destroyAllWindows()

