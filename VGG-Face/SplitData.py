{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "pycharm": {
     "is_executing": false,
     "name": "#%%\n"
    }
   },
   "outputs": [],
   "source": [
    "import os\n",
    "import os.path\n",
    "import random\n",
    "from shutil import copyfile\n",
    "\n",
    "def pics_for_one_user(people_path):\n",
    "    people_imgs = []\n",
    "    for img_file in os.listdir(people_path):\n",
    "            people_imgs.append(img_file)\n",
    "    random.shuffle(people_imgs)\n",
    "    return people_imgs\n",
    "\n",
    "\n",
    "def build_dataset(src_folder):\n",
    "    if not os.path.exists(train_folder):\n",
    "                os.makedirs(train_folder)\n",
    "    if not os.path.exists(valid_folder):\n",
    "                os.makedirs(valid_folder)\n",
    "    if not os.path.exists(test_folder):\n",
    "                os.makedirs(test_folder)\n",
    "    if not os.path.exists(poison_folder):\n",
    "                os.makedirs(poison_folder)\n",
    "            \n",
    "    for people_folder in os.listdir(src_folder):\n",
    "        people_imgs = pics_for_one_user(os.path.join(src_folder, people_folder))\n",
    "        #print(people_imgs)\n",
    "        os.makedirs(os.path.join(train_folder, people_folder))\n",
    "        os.makedirs(os.path.join(valid_folder, people_folder))\n",
    "        os.makedirs(os.path.join(test_folder, people_folder))\n",
    "        for i in people_imgs[:90]:\n",
    "            copyfile(os.path.join(src_folder, people_folder, i), os.path.join(train_folder, people_folder, i))\n",
    "        for i in people_imgs[90:100]:\n",
    "            copyfile(os.path.join(src_folder, people_folder, i), os.path.join(valid_folder, people_folder, i))\n",
    "        for i in people_imgs[100:110]:\n",
    "            copyfile(os.path.join(src_folder, people_folder, i), os.path.join(test_folder, people_folder, i))\n",
    "        for i in people_imgs[110:]:\n",
    "            copyfile(os.path.join(src_folder, people_folder, i), os.path.join(poison_folder, people_folder, i))\n",
    "        print(people_folder + ' ' + 'Processed')\n",
    "    \n",
    "src_folder = r\"..\\data\\VGG_images_DB\"\n",
    "test_folder  = r\"..\\data\\VGG_DB\\TestSet\"\n",
    "valid_folder = r\"..\\data\\VGG_DB\\validSet\"\n",
    "train_folder = r\"..\\data\\VGG_DB\\trainSet\"\n",
    "poison_folder = r\"..\\data\\VGG_DB\\PoisonSet\"\n",
    "\n",
    "build_dataset(src_folder)\n",
    "\n",
    "\n"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "name": "pycharm-f0a6e5d1",
   "language": "python",
   "display_name": "PyCharm (Python_basics)"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.7.3"
  },
  "pycharm": {
   "stem_cell": {
    "cell_type": "raw",
    "source": [],
    "metadata": {
     "collapsed": false
    }
   }
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}