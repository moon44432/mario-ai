import cv2
import numpy as np
from mss import mss

from hparams import mon

img_width = 128


def process_img(original_image):
    bw_img = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    resized_img = cv2.resize(bw_img, dsize=(img_width, img_width), interpolation=cv2.INTER_AREA)
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

            cv2.imshow('mario', new_screen)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                break
