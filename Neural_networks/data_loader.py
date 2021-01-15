import numpy as np
import matplotlib.pyplot as plt
from scipy import misc
import os.path as path
import glob


def load_data():
# Cesta k datasetu
    file_path = r'C:\Users\Petr\Desktop\Rozpoznání_obrazů_pomocí_hlubokých_neuronových_sítí_jagoš\Dataset\Train'
    image_paths = glob.glob(path.join(file_path, '*.png'))

# Přečtení obrázků a uložení dat do pole
    images = [misc.imread(path) for path in image_paths]
    images = np.asarray(images)

# Normalizace hodnot
    images = images.astype('float32')
    images = images / 255

# Přečtení označení obrázků a vytvoření vektoru pro označení
    images_num = images.shape[0]
    labels = np.zeros(images_num)
    for i in range(images_num):
        filename = path.basename(image_paths[i])[0]
        labels[i] = int(filename[0])

# Zamýchání dat
    mix_data = np.random.permutation(images_num)
    train_data = mix_data[0:images_num]

# Vytvoření zamýchaného trénovacího setu
    x_train = images[train_data, :, :]
    y_train = labels[train_data] 

    return x_train, y_train

def load_test_data():
# Cesta k datasetu
    file_path = r'C:\Users\Petr\Desktop\Rozpoznání_obrazů_pomocí_hlubokých_neuronových_sítí_jagoš\Dataset\Test'
    image_paths = glob.glob(path.join(file_path, '*.png'))

# Načtení obrázků
    images = [misc.imread(path) for path in image_paths]
    images = np.asarray(images)
    
# Normalizace hodnot
    images = images.astype('float32')
    images = images / 255

# Přečtení označení obrázků
    images_num = images.shape[0]
    labels = np.zeros(images_num)
    for i in range(images_num):
        filename = path.basename(image_paths[i])[0]
        labels[i] = int(filename[0])
        
# Zamýchání obrázků    
    mix_data = np.random.permutation(images_num)
    test_data = mix_data[0:images_num]

# Vytvoření zamýchaného testovacího setu    
    x_test = images[test_data, :, :]
    y_test = labels[test_data]
     
    return x_test, y_test

def change_matrix(x_train, x_test):
#upravení matice s daty
    input_dim = 12288 #64*64*3
    x_train = x_train.reshape(x_train.shape[0], input_dim) 
    x_test = x_test.reshape(x_test.shape[0], input_dim) 
    
#kontrola
    print('Dimenze trénovacího setu X:' + str(x_train.shape))
    print('Dimenze testovacího setu X:' + str(x_test.shape))
    
    return x_train, x_test

    