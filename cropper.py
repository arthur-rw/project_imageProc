import cv2
from tqdm import tqdm

class Cropper:
    
    def crop(im, BGR):
        counter = 0
        max = 0
        inc = 0
        max_coord = [0, 0]
        for i in tqdm(range(0, len(im))):
            for j in range(0, len(im[i])):
                
                flag = False
                inc = (inc + 1) % 100
                
                pixel = im[i][j]
                G = pixel[0]
                B = pixel[1]
                R = pixel[2]
                
                B_check = BGR[0]-20 < B < BGR[0]+20
                G_check = BGR[1]-20 < G < BGR[1]+20
                R_check = BGR[2]-20 < R < BGR[2]+20
                
                if (B_check & G_check & R_check):
                    flag = True
                    
                if (flag):
                    counter += 1
                    
                if (inc == 0):
                    if (counter > 0):
                        if (max < counter):
                            max = counter
                            max_coord = [i, j]
                    counter = 0
        
        h = max_coord[0]
        w = max_coord[1]
        
        print(max_coord)
        im_cropped = im[h-800:h+800, w-800:w+800]
        cv2.imwrite('cropped_4.jpg', im_cropped)
        
        return im_cropped
        
                    