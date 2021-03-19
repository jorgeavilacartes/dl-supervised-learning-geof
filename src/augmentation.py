# https://github.com/aleju/imgaug#example_images

import imgaug.augmenters as iaa

# Sometimes(0.5, ...) applies the given augmenter in 50% of all cases,
# e.g. Sometimes(0.5, GaussianBlur(0.3)) would blur roughly every second image.
#sometimes02 = lambda aug: iaa.Sometimes(0.2, aug)
sometimes05 = lambda aug: iaa.Sometimes(0.5, aug)
#sometimes07 = lambda aug: iaa.Sometimes(0.7, aug)

class Augmentation: 

    def __init__(self,):
        self.load_augmentation()
    
    def __call__(self, batch):
        return self.seq_aug(batch)

    def load_augmentation(self,):
        self.seq_aug =  iaa.Sequential(
                [
                    iaa.Fliplr(0.5), # horizontally flip with prob 0.5  
                    
                    # Do this with prob 05
                    sometimes05(
                        iaa.Affine(
                            rotate=(-20,20) # rotate by -20 to +20 degrees
                            )
                    )
                ]
            )