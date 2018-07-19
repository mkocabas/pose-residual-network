import os
import keras
from pycocotools.coco import  COCO
from opt import Options
from src.model import PRN, PRN_Seperate
from src.utils import train_bbox_generator, val_bbox_generator
from src.utils import get_anns
from src.evaluate import Evaluation


class My_Callback(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        self.model.save('checkpoint/'+option.exp + 'epoch_{}.h5'.format(epoch))
        print 'Epoch', epoch+1, 'has been saved'
        Evaluation(self.model, option, coco_val)
        print 'Epoch', epoch+1, 'has been tested'
        return


def main(option):
    if not os.path.exists('checkpoint/'+option.exp):
        os.makedirs('checkpoint/'+option.exp)

    model = PRN_Seperate(option.coeff*28,option.coeff*18, option.node_count)

    adam_optimizer = keras.optimizers.Adam(lr=option.lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(loss='binary_crossentropy', optimizer=adam_optimizer)
    Own_callback = My_Callback()


    model.fit_generator(generator=train_bbox_generator(coco_train, option.batch_size, option.coeff*28,option.coeff*18,option.threshold),
                        steps_per_epoch=len(get_anns(coco_train)) // option.batch_size,
                        validation_data=val_bbox_generator(coco_val, option.batch_size,option.coeff*28,option.coeff*18, option.threshold),
                        validation_steps=len(coco_val.getAnnIds()) // option.batch_size,
                        epochs=option.number_of_epoch,
                        callbacks=[Own_callback],
                        verbose=1,
                        initial_epoch=0)


if __name__ == "__main__":
    option = Options().parse()
    coco_train = COCO(os.path.join('data/annotations/person_keypoints_train2017.json'))
    coco_val = COCO(os.path.join('data/annotations/person_keypoints_val2017.json'))
    main(option)
