from PIL import Image
import os

def main():
    path = "./SEM_bilder"
    saving_path = "./Image_converting/converted_images/"
    os.makedirs(saving_path, exist_ok=True)
    for (root,dirs,files) in os.walk(path):
        for file in files:
            new_path = saving_path + file.split(".")[0] + ".png"
            try:
                image_path = os.path.join(root,file)
                try:
                    tiff_image = Image.open(image_path)
                except:
                    print("Failed to open image")
                tiff_image.save(new_path,"PNG")
                print(f"converted image: {file}")
            except:
                print(f"failed to convert image: {file}")

    
main()