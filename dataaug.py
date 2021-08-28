from keras.preprocessing.image import ImageDataGenerator,load_img,img_to_array

#data augmentation

datagen = ImageDataGenerator(
    rotation_range = 40,
    width_shift_range = 0.2,
    height_shift_range = 0.2,
    
#class 3
pic = load_img('C:/Users/lenovo/Desktop/ARCHIVE/HAM10000_images_part_1/ISIC_0029297.jpg')
pic.getpixel
pic_array=img_to_array(pic)
pic_array.shape

pic_array = pic_array.reshape((1,)+pic_array.shape)
pic_array.shape


# Generate 10 images
# batch_size: I can reduce it if lesion images are more so that it can be diversified
count = 0
for batch in datagen.flow(pic_array, batch_size=1,save_to_dir="C:/Users/lenovo/Desktop/ARCHIVE/HAM10000_images_part_3", save_prefix='ISIC', save_format='jpg'):
    count += 1
    if count == 10:
        break
    

