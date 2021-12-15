from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from model import AlexNet_v1
import tensorflow as tf
import json
import os


def main():
    data_root = os.path.abspath(os.path.join(os.getcwd()))  # 得到目前数据集照片
    image_path = os.path.join(data_root, "flower_data")  # flower data set path
    train_dir = os.path.join(image_path, "train")
    validation_dir = os.path.join(image_path, "val")
    assert os.path.exists(train_dir), "cannot find {}".format(train_dir)
    assert os.path.exists(validation_dir), "cannot find {}".format(validation_dir)

    # create direction for saving weights
    if not os.path.exists("save_weights"):
        os.makedirs("save_weights")                  #生成权重文件

    im_height = 224
    im_width = 224
    batch_size = 32
    epochs = 10

    # data generator with data augmentation
    train_image_generator = ImageDataGenerator(rescale=1. / 255,        #图片生成器，载入文件夹中的图片数据并进行预处理，会将labels转换成onehot编码形式
                                               horizontal_flip=True)             #随机水平翻转
    validation_image_generator = ImageDataGenerator(rescale=1. / 255)            #验证机生成器

    train_data_gen = train_image_generator.flow_from_directory(directory=train_dir,       #指向训练集路径
                                                               batch_size=batch_size,
                                                               shuffle=True,
                                                               target_size=(im_height, im_width),
                                                               class_mode='categorical')     #分类模型
    total_train = train_data_gen.n            #获得整个训练集样本个数

    # get class dict
    class_indices = train_data_gen.class_indices      #获取类别名称的索引

    # transform value and key of dict
    inverse_dict = dict((val, key) for key, val in class_indices.items())
    # write dict into json file
    json_str = json.dumps(inverse_dict, indent=4)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    val_data_gen = validation_image_generator.flow_from_directory(directory=validation_dir,     #验证集生成器信息
                                                                  batch_size=batch_size,
                                                                  shuffle=False,
                                                                  target_size=(im_height, im_width),
                                                                  class_mode='categorical')
    total_val = val_data_gen.n
    print("using {} images for training, {} images for validation.".format(total_train,
                                                                           total_val))



    model = AlexNet_v1(im_height=im_height, im_width=im_width, num_classes=5)
    model.summary()

    # using keras high level api for training
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),    #对模型进行编译
                  loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
                  metrics=["accuracy"])

    callbacks = [tf.keras.callbacks.ModelCheckpoint(filepath='./save_weights/myAlex.h5',    #回调函数
                                                    save_best_only=True,
                                                    save_weights_only=True,
                                                    monitor='val_loss')]

    # tensorflow2.1 recommend to using fit
    history = model.fit(x=train_data_gen,
                        steps_per_epoch=total_train // batch_size,
                        epochs=epochs,
                        validation_data=val_data_gen,
                        validation_steps=total_val // batch_size,
                        callbacks=callbacks)

    # plot loss and accuracy image
    history_dict = history.history
    train_loss = history_dict["loss"]
    train_accuracy = history_dict["accuracy"]
    val_loss = history_dict["val_loss"]
    val_accuracy = history_dict["val_accuracy"]

    # figure 1
    plt.figure()
    plt.plot(range(epochs), train_loss, label='train_loss')
    plt.plot(range(epochs), val_loss, label='val_loss')
    plt.legend()
    plt.xlabel('epochs')
    plt.ylabel('loss')

    # figure 2
    plt.figure()
    plt.plot(range(epochs), train_accuracy, label='train_accuracy')
    plt.plot(range(epochs), val_accuracy, label='val_accuracy')
    plt.legend()
    plt.xlabel('epochs')
    plt.ylabel('accuracy')
    plt.show()

    # history = model.fit_generator(generator=train_data_gen,
    #                               steps_per_epoch=total_train // batch_size,
    #                               epochs=epochs,
    #                               validation_data=val_data_gen,
    #                               validation_steps=total_val // batch_size,
    #                               callbacks=callbacks)

    # # using keras low level api for training
    # loss_object = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
    # optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005)
    #
    # train_loss = tf.keras.metrics.Mean(name='train_loss')
    # train_accuracy = tf.keras.metrics.CategoricalAccuracy(name='train_accuracy')
    #
    # test_loss = tf.keras.metrics.Mean(name='test_loss')
    # test_accuracy = tf.keras.metrics.CategoricalAccuracy(name='test_accuracy')
    #
    #
    # @tf.function
    # def train_step(images, labels):
    #     with tf.GradientTape() as tape:
    #         predictions = model(images, training=True)
    #         loss = loss_object(labels, predictions)
    #     gradients = tape.gradient(loss, model.trainable_variables)
    #     optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    #
    #     train_loss(loss)
    #     train_accuracy(labels, predictions)
    #
    #
    # @tf.function
    # def test_step(images, labels):
    #     predictions = model(images, training=False)
    #     t_loss = loss_object(labels, predictions)
    #
    #     test_loss(t_loss)
    #     test_accuracy(labels, predictions)
    #
    #
    # best_test_loss = float('inf')
    # for epoch in range(1, epochs+1):
    #     train_loss.reset_states()        # clear history info
    #     train_accuracy.reset_states()    # clear history info
    #     test_loss.reset_states()         # clear history info
    #     test_accuracy.reset_states()     # clear history info
    #     for step in range(total_train // batch_size):
    #         images, labels = next(train_data_gen)
    #         train_step(images, labels)
    #
    #     for step in range(total_val // batch_size):
    #         test_images, test_labels = next(val_data_gen)
    #         test_step(test_images, test_labels)
    #
    #     template = 'Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}'
    #     print(template.format(epoch,
    #                           train_loss.result(),
    #                           train_accuracy.result() * 100,
    #                           test_loss.result(),
    #                           test_accuracy.result() * 100))
    #     if test_loss.result() < best_test_loss:
    #        model.save_weights("./save_weights/myAlex.ckpt", save_format='tf')


if __name__ == '__main__':
    main()
