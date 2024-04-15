import numpy as np
from keras.layers import Conv2D, LeakyReLU, Dropout, AveragePooling2D, BatchNormalization, Reshape, Conv2DTranspose, UpSampling2D, Activation, Flatten, Dense
from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import GaussianNoise
 
class GAN(object):
 
    def __init__(self):
 
        # Modèles
        self.D = None
        self.G = None
 
        self.OD = None
 
        self.DM = None
        self.AM = None
 
        # Configuration
        self.LR = 0.0001
        self.steps = 1
 
    def discriminator(self):
 
        if self.D:
            return self.D
 
        self.D = Sequential()
 
        # Ajouter du bruit gaussien pour éviter le surajustement du discriminateur
        self.D.add(GaussianNoise(0.2, input_shape=[256, 256, 3]))
 
        # Image 256x256x3
        self.D.add(Conv2D(filters=8, kernel_size=3, padding='same'))
        self.D.add(LeakyReLU(0.2))
        self.D.add(Dropout(0.25))
        self.D.add(AveragePooling2D())
 
        # 128x128x8
        self.D.add(Conv2D(filters=16, kernel_size=3, padding='same'))
        self.D.add(BatchNormalization(momentum=0.7))
        self.D.add(LeakyReLU(0.2))
        self.D.add(Dropout(0.25))
        self.D.add(AveragePooling2D())
 
        # 64x64x16
        self.D.add(Conv2D(filters=32, kernel_size=3, padding='same'))
        self.D.add(BatchNormalization(momentum=0.7))
        self.D.add(LeakyReLU(0.2))
        self.D.add(Dropout(0.25))
        self.D.add(AveragePooling2D())
 
        # 32x32x32
        self.D.add(Conv2D(filters=64, kernel_size=3, padding='same'))
        self.D.add(BatchNormalization(momentum=0.7))
        self.D.add(LeakyReLU(0.2))
        self.D.add(Dropout(0.25))
        self.D.add(AveragePooling2D())
 
        # 16x16x64
        self.D.add(Conv2D(filters=128, kernel_size=3, padding='same'))
        self.D.add(BatchNormalization(momentum=0.7))
        self.D.add(LeakyReLU(0.2))
        self.D.add(Dropout(0.25))
        self.D.add(AveragePooling2D())
 
        # 8x8x128
        self.D.add(Conv2D(filters=256, kernel_size=3, padding='same'))
        self.D.add(BatchNormalization(momentum=0.7))
        self.D.add(LeakyReLU(0.2))
        self.D.add(Dropout(0.25))
        self.D.add(AveragePooling2D())
 
        # 4x4x256
        self.D.add(Flatten())
 
        # 256
        self.D.add(Dense(128))
        self.D.add(LeakyReLU(0.2))
 
        self.D.add(Dense(1, activation='sigmoid'))
 
        return self.D
 
    def generator(self):
 
        if self.G:
            return self.G
 
        self.G = Sequential()
 
        self.G.add(Reshape(target_shape=[1, 1, 4096], input_shape=[4096]))
 
        # 1x1x4096
        self.G.add(Conv2DTranspose(filters=256, kernel_size=4))
        self.G.add(Activation('relu'))
 
        # 4x4x256 - taille du noyau augmentée de 1
        self.G.add(Conv2D(filters=256, kernel_size=4, padding='same'))
        self.G.add(BatchNormalization(momentum=0.7))
        self.G.add(Activation('relu'))
        self.G.add(UpSampling2D())
 
        # 8x8x256 - taille du noyau augmentée de 1
        self.G.add(Conv2D(filters=128, kernel_size=4, padding='same'))
        self.G.add(BatchNormalization(momentum=0.7))
        self.G.add(Activation('relu'))
        self.G.add(UpSampling2D())
 
        # 16x16x128
        self.G.add(Conv2D(filters=64, kernel_size=3, padding='same'))
        self.G.add(BatchNormalization(momentum=0.7))
        self.G.add(Activation('relu'))
        self.G.add(UpSampling2D())
 
        # 32x32x64
        self.G.add(Conv2D(filters=32, kernel_size=3, padding='same'))
        self.G.add(BatchNormalization(momentum=0.7))
        self.G.add(Activation('relu'))
        self.G.add(UpSampling2D())
 
        # 64x64x32
        self.G.add(Conv2D(filters=16, kernel_size=3, padding='same'))
        self.G.add(BatchNormalization(momentum=0.7))
        self.G.add(Activation('relu'))
        self.G.add(UpSampling2D())
 
        # 128x128x16
        self.G.add(Conv2D(filters=8, kernel_size=3, padding='same'))
        self.G.add(Activation('relu'))
        self.G.add(UpSampling2D())
 
        # 256x256x8
        self.G.add(Conv2D(filters=3, kernel_size=3, padding='same'))
        self.G.add(Activation('sigmoid'))
 
        return self.G
 
    def DisModel(self):
 
        if self.DM == None:
            self.DM = Sequential()
            self.DM.add(self.discriminator())
 
        self.DM.compile(optimizer=Adam(learning_rate=self.LR * (0.85 ** np.floor(self.steps / 10000))),
                        loss='binary_crossentropy')
 
        return self.DM
 
    def AdModel(self):
 
        if self.AM == None:
            self.AM = Sequential()
            self.AM.add(self.generator())
            self.AM.add(self.discriminator())
 
        self.AM.compile(optimizer=Adam(learning_rate=self.LR * (0.85 ** np.floor(self.steps / 10000))),
                        loss='binary_crossentropy')
 
        return self.AM
 
    def sod(self):
 
        self.OD = self.D.get_weights()
 
    def lod(self):
 
        self.D.set_weights(self.OD)
 
 
import numpy as np
import random
import matplotlib.pyplot as plt
from PIL import Image
 
class Model_GAN(object):
 
    def __init__(self):
        self.GAN = GAN()
        self.DisModel = self.GAN.DisModel()
        self.AdModel = self.GAN.AdModel()
        self.generator = self.GAN.generator
 
    def train(self, batch=16):
        a, b = self.train_dis(batch)
        c = self.train_gen(batch)
 
        print(f"D Real: {str(a)}, D Fake: {str(b)}, G All: {str(c)}")
 
        if self.GAN.steps % 500 == 0:
            self.save(np.floor(self.GAN.steps / 1000))
            self.evaluate()
 
        if self.GAN.steps % 5000 == 0:
            self.GAN.AM = None
            self.GAN.DM = None
            self.AdModel = self.GAN.AdModel()
            self.DisModel = self.GAN.DisModel()
 
        self.GAN.steps += 1
 
    def train_dis(self, batch):
        # Obtenir des images réelles
        im_no = random.randint(0, len(Images) - batch - 1)
        train_data = Images[im_no:im_no + int(batch / 2)]
        label_data = [zero() for _ in range(int(batch / 2))]
 
        d_loss_real = self.DisModel.train_on_batch(np.array(train_data), np.array(label_data))
 
        # Obtenir des images fausses
        train_data = self.generator.predict(noise(int(batch / 2)))
        label_data = [one() for _ in range(int(batch / 2))]
 
        d_loss_fake = self.DisModel.train_on_batch(train_data, np.array(label_data))
 
        return d_loss_real, d_loss_fake
 
    def train_gen(self, batch):
        self.GAN.sod()
        label_data = [zero() for _ in range(int(batch))]
 
        g_loss = self.AdModel.train_on_batch(noise(batch), np.array(label_data))
 
        self.GAN.lod()
 
        return g_loss
 
    def evaluate(self):
        im_no = random.randint(0, len(Images) - 1)
        im1 = Images[im_no]
 
        im2 = self.generator.predict(noise(2))
 
        plt.figure(1)
        plt.imshow(im1)
 
        plt.figure(2)
        plt.imshow(im2[0])
 
        plt.figure(3)
        plt.imshow(im2[1])
 
        plt.show()
 
    def save(self, num):
        gen_json = self.GAN.G.to_json()
        dis_json = self.GAN.D.to_json()
 
        with open("Models/gen.json", "w+") as json_file:
            json_file.write(gen_json)
 
        with open("Models/dis.json", "w+") as json_file:
            json_file.write(dis_json)
 
        self.GAN.G.save_weights("Models/gen"+str(num)+".h5")
        self.GAN.D.save_weights("Models/dis"+str(num)+".h5")
 
        print(f"Model number {str(num)} Saved!")
 
    def load(self, num):
        steps1 = self.GAN.steps
 
        self.GAN = None
        self.GAN = GAN()
 
        # Générateur
        with open("Models/gen.json", 'r') as gen_file:
            gen_json = gen_file.read()
 
        self.GAN.G = model_from_json(gen_json)
        self.GAN.G.load_weights("Models/gen"+str(num)+".h5")
 
        # Discriminateur
        with open("Models/dis.json", 'r') as dis_file:
            dis_json = dis_file.read()
 
        self.GAN.D = model_from_json(dis_json)
        self.GAN.D.load_weights("Models/dis"+str(num)+".h5")
 
        # Réinitialisation
        self.generator = self.GAN.generator
        self.DisModel = self.GAN.DisModel()
        self.AdModel = self.GAN.AdModel()
 
        self.GAN.steps = steps1
 
    def eval2(self, num=0):
        im2 = self.generator.predict(noise(48))
 
        r1 = np.concatenate(im2[:8], axis=1)
        r2 = np.concatenate(im2[8:16], axis=1)
        r3 = np.concatenate(im2[16:24], axis=1)
        r4 = np.concatenate(im2[24:32], axis=1)
        r5 = np.concatenate(im2[32:40], axis=1)
        r6 = np.concatenate(im2[40:48], axis=1)
 
        c1 = np.concatenate([r1, r2, r3, r4, r5, r6], axis=0)
 
        x = Image.fromarray(np.uint8(c1 * 255))
 
        x.save("Results/i"+str(num)+".png")