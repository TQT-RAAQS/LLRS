import numpy as np
from datetime import datetime
from pathlib import Path
from PIL import Image

from experiment.instruments.camera.andor_ixonultra888.image import Image

def save_test_images():
    # get all pickled files
    pathlist = Path("../images/kinetic_series").glob("*.pk")
    date = str(datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    folder = "../images/img/"+date+"/"
    counter = 1
    for path in pathlist:
        img = Image.load(str(path))
        img.save_all_images(folder+"/"+str(counter)+"/", "image")
        counter += 1

if __name__ == "__main__":
    save_test_images()
