#!/usr/bin/python
import os
import sys
import json
import numpy as np
from metallograph_segmentation.models.sm import model as px
from metallograph_segmentation.utils import data_loader, predictions
from metallograph_segmentation.models.sm.model import load_sm


def run_inference(conf, model_dir, data_dir):
	# Set inference folder
	save_dir = os.path.join(model_dir, "inference")
	if not os.path.exists(save_dir):
		os.mkdir(save_dir)

	# Load model
	conf["weights_file"] = os.path.join(model_dir, "weights.best.pt")
	model = load_sm(conf)
	model_test = (model, (None, None))

	# Load data
	dataset = data_loader.DATASETS[conf["dataset"]](
		path=data_dir,
		num_classes=conf["num_classes"],
		microstructures_dir='',
		labels_dir='',
		load_labels=False
	)
	images, _ = dataset.data()
	# Load palette
	if conf["dataset"] == "metaldam":
		colors = [[128, 0, 255], [43, 255, 0], [255, 0, 0], [255, 0, 255], [255, 255, 0]]
	else:
		colors = [[255, 255, 0], [0, 0, 255], [21, 180, 214], [75, 179, 36]]

	if conf.get("debug"):
		print("=== END EVALUATION <==", file=sys.stderr)
	# Get predictions
	for img_id in range(images.shape[0]):
		img = np.reshape(images[img_id], (1, images[img_id].shape[0], images[img_id].shape[1]))
		preds = px.predict_sm(model_test, img, dict(conf))
		print(preds.shape)
		np.save(os.path.join(save_dir, dataset.image_files[img_id][:-4]+".npy"), preds)
		predictions.save_images(preds, os.path.join(save_dir, dataset.image_files[img_id][:-4]+"_mask.png"), colors=colors)


model_dir = "/dev/hdd/deep_vision/segmentation/additive/models/singlemodel_c2x2/unetplusplus/" \
	if len(sys.argv) <= 1 else sys.argv[1]
with open(os.path.join(model_dir, "parameters.json")) as json_file:
	conf = json.load(json_file)
print("==> MODEL CONFIGURATION:\n", str(conf), file=sys.stderr)
data_dir = os.path.join(conf['datadir'], "images") if len(sys.argv) <= 2 else sys.argv[2]

run_inference(conf, model_dir, data_dir)
