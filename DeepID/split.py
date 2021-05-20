import os
import os.path
import random
from shutil import copyfile


def fatch_pics_for_one_user(people_path):
    people_imgs = []
    for video_folder in os.listdir(people_path):
        for video_file_name in os.listdir(os.path.join(people_path, video_folder)):
            people_imgs.append(os.path.join(people_path, video_folder, video_file_name))
    random.shuffle(people_imgs)
    return people_imgs


def build_dataset(src_folder):
    if not os.path.exists(train_folder):
        os.makedirs(train_folder)
    if not os.path.exists(valid_folder):
        os.makedirs(valid_folder)
    if not os.path.exists(test_folder):
        os.makedirs(test_folder)


    for people_folder in os.listdir(src_folder):
        people_imgs = fatch_pics_for_one_user(os.path.join(src_folder, people_folder))
        # print(people_imgs)
        os.makedirs(os.path.join(train_folder, people_folder))
        os.makedirs(os.path.join(valid_folder, people_folder))
        os.makedirs(os.path.join(test_folder, people_folder))
        if len(people_imgs) < 100:
            pass
        else:
            for i in people_imgs[:90]:
                copyfile(i, os.path.join(train_folder, people_folder,os.path.basename((i))))
            for i in people_imgs[90:100]:
                copyfile(i, os.path.join(valid_folder, people_folder,os.path.basename((i))))
            for i in people_imgs[100:110]:
                copyfile(i, os.path.join(test_folder, people_folder,os.path.basename((i))))
        print(people_folder + ' ' + 'Processed')




src_folder = "data\crop_images_DB"
test_folder = "data\TestSet"
valid_folder = "data\ValidationSet"
train_folder = "data\TrainSet"

build_dataset(src_folder)
