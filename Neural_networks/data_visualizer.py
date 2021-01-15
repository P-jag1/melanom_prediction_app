import numpy as np
from PIL import Image
from scipy import misc, ndimage
import matplotlib.pyplot as plt


def visualize_examples(x, y, n):
    # x = matice s daty
    # y = vektor se správným označením
    # n = počet obrázků k zobrazení
    x = x.reshape(x.shape[0], 64, 64, 3)
    
    # výběr n správných (1) příkladů
    positive_labels = (y == 1)
    positive_examples = x[positive_labels, :, :]
    positive_examples = positive_examples[0:n, :, :]

    # výběr n nesprávných (0) příkladů
    negative_labels = (y == 0)
    negative_examples = x[negative_labels, :, :]
    negative_examples = negative_examples[0:n, :, :]
    
    #vykreslení příkladů
    figure = plt.figure()
    count = 0
    
    for i in range(positive_examples.shape[0]):
        count += 1
        #správné příklady
        figure.add_subplot(2, positive_examples.shape[0], count)
        plt.imshow(positive_examples[i, :, :])
        plt.axis('off')
        plt.title("1")
        
        #nesprávné příklady
        figure.add_subplot(1, negative_examples.shape[0], count)
        plt.imshow(negative_examples[i, :, :])
        plt.axis('off')
        plt.title("0")
    plt.show()

def visualize_incorrect_labels(x_test, y_test, y_predicted):
    # metoda pro zobrazení nesprávných výsledků predikce na testovacím souboru
    # x_test - testovací data
    # y_test - správné označení testovacích dat
    # y_predicted - označení obrázků neuronovou sítí    
    # výběr špatně označených obrázků
    incorrect_labels = (y_test != y_predicted)
    y_test = y_test[incorrect_labels]
    y_predicted = y_predicted[incorrect_labels]
    x_test = x_test[incorrect_labels, :]
    
    # vytvoření plátna
    figsize = 64 / float(2), 64 / float(2)
    figure = plt.figure(figsize = figsize)
    
    # vykreslení jednotlivých obrázků
    count = 0
    maximum_square = np.ceil(np.sqrt(x_test.shape[0]))
    num_images = len(x_test[0])
    print('počet špatných odhadů: ' + str(x_test.shape[0]))
    
    for i in range(x_test.shape[0]):
        count += 1
        figure.add_subplot(maximum_square, maximum_square, count)
        img = x_test[i]
        img = img.reshape((64, 64,3))
        plt.imshow(img)
        plt.axis('off')
        plt.title("Označeno jako: " + str(int(y_predicted[i])) + ", Správné: " + str(int(y_test[i])), fontsize=10)

    plt.show()

def print_graphs(history):
    # metoda pro vykreslení grafu přesnosti síte a grafu ztráty sítě
    # vstupem metody je historie trénování modelu, která je vytvořena pomocí model.fit()
    # graf přesnosti
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('Přesnost sítě')
    plt.ylabel('přesnost')
    plt.xlabel('epochy')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()
    
    # graf ztráty
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Ztráta sítě')
    plt.ylabel('ztráta')
    plt.xlabel('epochy')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()