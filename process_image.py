
import numpy as np
import cv2
from mss import mss

mon = {'top': 60, 'left': 0, 'width': 512, 'height': 480}

def process_img(original_image):
    bw_img = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    resized_img = cv2.resize(bw_img, dsize=(128, 128), interpolation=cv2.INTER_AREA)
    return resized_img

def get_image():
    with mss() as sct:
        screen = np.array(sct.grab(mon))
        return screen


if __name__ == "__main__":
    with mss() as sct:
        while True:
            screen = get_image()
            new_screen = process_img(screen)

            # print(new_screen)
            cv2.imshow('mario', new_screen)

            if cv2.waitKey(25) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                break
