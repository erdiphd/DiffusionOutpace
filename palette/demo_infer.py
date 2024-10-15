import infer
import cv2

img = cv2.imread("./test_images/GT_heatmap_episode89.jpg")
#img = Image.open("./test_images/GT_heatmap_episode89.jpg").convert('RGB')

episode = "999"
output_imgs = infer.single_img(img,True, f"./inpaint_results/episode_{episode}/")