import os
import sys
curr_dir = os.getcwd()
sys.path.append(curr_dir)

from pathlib import Path

from predictor import *
base_dir = Path(curr_dir)
validset = base_dir.joinpath('dataset/valid')
image_file = validset.joinpath('2/image_05136.jpg')

model = load_flower_network(root.joinpath('model_best.pth'))
checkpoint = load_checkpoint(root.joinpath('checkpoint.pth'))

probs, classes = predict(image_file, model)

print(probs)
view_classify(image_file, 2, probs, classes, checkpoint['class_to_idx'])

