import tqdm
import numpy as np
import logging
import json
import os
from edflow import get_logger
import glob
import subprocess


def inference_callback(root, data_in, data_out, config):
    logger = get_logger("Alphapose Callback")
    callback_config = config.get("alphapose_callback")

    alphapose_config = callback_config.get("config")
    checkpoint = callback_config.get("checkpoint")
    cwd = callback_config.get("alphapose_dir")
    pythonpath = callback_config.get("alphapose_python")
    infer_script = callback_config.get("infer_script")

    outdir = callback_config.get("outdir")
    indir = callback_config.get("indir")

    outdir = os.path.abspath(os.path.join(root, outdir))
    indir = os.path.abspath(os.path.join(root, indir))

    # input_files = REGEX
    command_string = f"{pythonpath} {infer_script} --cfg {alphapose_config} --checkpoint {checkpoint} --indir {indir} --outdir {outdir} --detector yolo"

    logger.info("Running command")
    for c in command_string.split(" "):
        logger.info(c)

    subprocess.call(command_string, shell=True, cwd=cwd)
    results_file = os.path.join(root, outdir, "alphapose-results.json")
    return results_file


def pck_callback(root, data_in, data_out, config):
    predicted_poses_file = inference_callback(root, data_in, data_out, config)

    callback_config = config.get("alphapose_pck_callback")
    true_poses_file = callback_config["true_poses_file"]
    distance_threshold = callback_config["distance_threshold"]
    if isinstance(distance_threshold, float) or isinstance(distance_threshold, int):
        pck = pck_from_posefiles(
            true_poses_file, predicted_poses_file, distance_threshold
        )
        logger = get_logger("Alphapose PCK Callback")
        logger.info(f"PCK@{distance_threshold} : {pck * 100: .02f}%")
        out = {f"pck@{distance_threshold}": pck}
    elif isinstance(distance_threshold, list):
        out = {}
        for d in distance_threshold:
            pck = pck_from_posefiles(true_poses_file, predicted_poses_file, d)
            logger = get_logger("Alphapose PCK Callback")
            logger.info(f"PCK@{d} : {pck * 100: .02f}%")
            out[f"pck@{d}"] = pck
    return out


def pck_from_posefiles(true_poses_file, predicted_poses, distance_threshold):
    """Calculate PCK from 2 annotation files generated from alpha pose model.

    The file names for each annotated image in both pose files have to match.

    Parameters
    ----------
    true_poses_file : str
        path to true poses
    predicted_poses_file : str
        path to predicted poses
    distance_threshold : int
        distance in pixels to threshold for correctness

    Returns
    -------
    np.ndarray
        mean pck over entire pose files
    """
    data_1 = read_posefile(true_poses_file)
    data_2 = read_posefile(predicted_poses)
    pck_values = []
    for pose_data_true, pose_data_predicted in zip(data_1, data_2):
        kp_scores_true = np.array(pose_data_true["keypoints"]).reshape((-1, 3))
        kp_scores_predicted = np.array(pose_data_predicted["keypoints"]).reshape(
            (-1, 3)
        )
        kp_true, scores_true = (kp_scores_true[:, :2], kp_scores_true[:, -1])
        kp_predicted, scores_predicted = (
            kp_scores_predicted[:, :2],
            kp_scores_predicted[:, -1],
        )
        pck_value = pck(kp_true, kp_predicted, distance_threshold)
        pck_values.append(pck_value)
    return np.mean(pck_values)


def pck(true, predicted, distance_threshold):
    """Percentage of correct keypoints with given threshold

    Parameters
    ----------
    true : np.ndarray
        keypoints array shaped [n, 2]
    predicted : np.ndarray
        keypoints array shaped [n, 2]
    distance_threshold : int or float
        distance threshold in pixels or relative to image size. Has to match range of keypoints

    Returns
    -------
    np.ndarray
        PCK
    """
    distances = true - predicted
    distances = np.linalg.norm(distances, ord=2, axis=1)  # l2 norm
    pck = distances < distance_threshold
    return np.mean(pck)


def read_posefile(file):
    """
    returns
    np.ndarray
        keypoints
    np.ndarray
        keypoint visibility
    """
    with open(file, "r") as f:
        data = json.load(f)
    return data
