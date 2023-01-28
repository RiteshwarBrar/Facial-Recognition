import cv2
from simple_facerec import SimpleFacerec

# Load Camera
sfr = SimpleFacerec()
sfr.load_encoding_images("images/")

# Encode faces from a folder 
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)

    # Face Detection
    face_locations, face_names = sfr.detect_known_faces(frame)
    for face_loc, name in zip(face_locations, face_names):
        y1, x1, y2, x2 = face_loc[0], face_loc[1], face_loc[2], face_loc[3]

        cv2.putText(frame, name,(x2+10, y2+20), cv2.FONT_HERSHEY_PLAIN, 1,	(255,255,255), 2)
        cv2.rectangle(frame, (x1, y1), (x2,y2), (0,0,200), 4)
        
    cv2.imshow("Frame", frame)

    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
