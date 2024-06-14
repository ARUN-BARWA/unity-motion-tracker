import cv2
from cvzone.PoseModule import PoseDetector

# Initialize the video capture with the video file
cap = cv2.VideoCapture('C:\\Users\\ARUN BARWA\\Downloads\\video.mp4')

# Initialize the PoseDetector
detector = PoseDetector()
posList = []

while True:
    # Capture frame-by-frame
    success, img = cap.read()

    # If the frame was not captured successfully, break the loop
    if not success:
        print("End of video or error reading frame.")
        break
    img = cv2.resize(img, (500,500))
    # Perform pose detection
    img = detector.findPose(img)
    lmList, bboxInfo = detector.findPosition(img)

    # If landmarks are found, process them
    if bboxInfo:
        lmString = ''
        for lm in lmList:
            lmString += f'{lm[0]},{img.shape[0] - lm[1]},{lm[2]},'
        posList.append(lmString.strip(','))  # remove the trailing comma

    # Print the length of the position list
    print(len(posList))

    # Display the resulting frame
    cv2.imshow("Image", img)

    # Check for key press events
    key = cv2.waitKey(1)
    if key == ord('s'):
        # Write the position list to a file when 's' is pressed
        with open("AnimationFile.txt", 'w') as f:
            for item in posList:
                f.write("%s\n" % item)
        print("Coordinates saved to AnimationFile.txt")

    # Break the loop if 'q' is pressed
    if key == ord('q'):
        break

# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()