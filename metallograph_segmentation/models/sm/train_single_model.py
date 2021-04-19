#!/usr/bin/python
from metallograph_segmentation.models.sm import model as px
import os
import datetime
import torch
import numpy as np
from metallograph_segmentation.utils import data_loader, predictions, config, save
import sys

RANDOM_SEED = 123456789
from numpy.random import seed
seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)


def run_single_model_training(conf, inference_folder):
	# Set up folders
	modeldir = conf.get("modeldir", conf.get("save_dir", "experiments"))
	os.makedirs(modeldir, exist_ok=True)
	name = os.path.join(modeldir, conf.get("name", f"sm_{conf['model']}_{conf['dataset']}_"
	f"{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"))
	os.makedirs(name, exist_ok=True)
	conf["name"] = name

	# Load dataset
	if conf.get("debug"):
		print("==> START LOADING ===", file=sys.stderr)
	if conf["dataset"] == "metaldam":
		dataset = data_loader.DATASETS[conf["dataset"]](
			path=conf.get("datadir", "../data"),
			num_classes=conf["num_classes"],
			crossval=False,
		)
		(x_train, y_train), (x_val, y_val), (x_test, y_test) = dataset.partitions()
		test_paths = dataset.test_paths
		if conf.get("debug"):
			print(x_train.shape, y_train.shape)
			print(np.min(y_train), np.max(y_train))
			print(x_val.shape, y_val.shape)
			print(np.min(y_val), np.max(y_val))
			print(x_test.shape, y_test.shape)
			print(np.min(y_test), np.max(y_test))
		colors = [[128, 0, 255], [43, 255, 0], [255, 0, 0], [255, 0, 255], [255, 255, 0]]
	else:
		raise ValueError("Not supported dataset for single_model training.")

	# If test from folder without labels
	if inference_folder:
		dataset = data_loader.DATASETS[conf["dataset"]](
			path=inference_folder,
			num_classes=conf["num_classes"],
			microstructures_dir='',
			labels_dir='',
			load_labels=False,  # y_test will be None
		)
		x_test, y_test = dataset.data()
		test_paths = dataset.image_files

	# Run training
	if conf.get("debug"):
		print("=== END LOADING <==", file=sys.stderr)
	if conf.get("debug"):
		print("==> START TRAINING ===", file=sys.stderr)
	model_test = px.train_sm(dict(conf), x_train, y_train, x_val, y_val)
	if conf.get("debug"):
		print("=== END TRAINING <==", file=sys.stderr)

	# Get metrics
	if not inference_folder:
		if conf.get("debug"):
			print("=== START EVALUATION <==", file=sys.stderr)
		metrics = px.evaluate_sm(model_test, x_test, y_test, dict(conf))
		eval_dict = {}
		eval_dict.update(testing=metrics)
		save.save_experiment(conf, eval_dict, os.path.basename(name), save_dir=modeldir)
		if conf.get("debug"):
			print("=== END EVALUATION <==", file=sys.stderr)

	# Set up predicitons folder
	if conf.get("debug"):
		print("=== START INFERENCE <==", file=sys.stderr)
	save_infer_dir = os.path.join(name, "test_results")
	if not os.path.exists(save_infer_dir):
		os.mkdir(save_infer_dir)
	# Get predictions
	for img_id in range(x_test.shape[0]):
		img = np.reshape(x_test[img_id], (1, x_test[img_id].shape[0], x_test[img_id].shape[1]))
		# Get predictions
		preds = px.predict_sm(model_test, img, dict(conf))
		print("Predicted image {} with shape {}".format(img_id, preds.shape))
		np.save(os.path.join(save_infer_dir, test_paths[img_id][:-4]+".npy"), preds)
		predictions.save_images(preds, os.path.join(save_infer_dir, test_paths[img_id][:-4]+"_mask.png"), colors =colors)
	if conf.get("debug"):
		print("=== END INFERENCE <==", file=sys.stderr)
	if conf.get("debug"):
		print("=== END SCRIPT <==", file=sys.stderr)


# Este parámetro limita el número de cpus que utiliza este proceso.
torch.set_num_threads(3)
config_file = "models/sm/config/unet_metaldam_pretrined.config" if len(sys.argv) <= 1 else sys.argv[1]
conf = config.get_config(config_file)
print("==> CURRENT CONFIGURATION:\n", str(conf), file=sys.stderr)

inference_folder = "" if len(sys.argv) <= 2 else sys.argv[2]

run_single_model_training(conf, inference_folder)

