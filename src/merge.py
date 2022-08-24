import cv2
img1 = cv2.imread('F:/WJ_project/ESRGAN/eval/output/RRDB_tiny_withD/30_Capture_00002.png')
img2 = cv2.imread('F:/WJ_project/ESRGAN/eval/output/RRDB_withoutD/30_Capture_00002.png')
img3 = cv2.addWeighted(img1, 0.7, img2, 0.3, 1)
cv2.imwrite('F:/WJ_project/ESRGAN/eval/output/merge/30_Capture_00002.png', img3)
