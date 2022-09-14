from PIL import Image 
import os 

def png_to_jpg(path_to_directory):
    for file in os.listdir(path_to_directory): 
        if file.endswith(".png"):
            filepath =  path_to_directory + '/' + file
            img = Image.open(filepath) 
            if img is None:
                os.remove(filepath) 
                continue
            if not img.mode == 'RGB':
                img = img.convert('RGB') 
            file_name, file_ext = os.path.splitext(file)
            img.save('{}.jpg'.format(path_to_directory + '/' + file_name))
            os.remove(filepath)