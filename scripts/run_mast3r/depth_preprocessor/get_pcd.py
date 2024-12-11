import pickle
import os, cv2
import numpy as np
from PIL import Image
from glob import glob
from argparse import ArgumentParser

from scripts.run_mast3r.depth_preprocessor.utils import resize_to_mast3r
from scripts.run_mast3r.depth_preprocessor.pcd_utils import unproject_depth


def mast3r_unprojection(
    data_dir,
    maskpaths,
    imagepaths,
    img_w,
    img_h,
    skip_dynamic,
    static_dst_dir="static",
    dynamic_dst_dir="dynamic",
    depth_dir="depth",
):
    assert data_dir is not None, "Please provide a depth name"

    # Load the global params
    pkl_path = os.path.join(data_dir, "global_params.pkl")
    with open(pkl_path, "rb") as pkl_file:
        data = pickle.load(pkl_file)

    focal = data["focals"][0]
    depth_max = data["max_depths"][0]
    depths = np.array(data["depths"])

    depths *= depth_max
    depths = np.clip(depths, 0, depth_max)

    static_pcd_path = os.path.join(data_dir, static_dst_dir)
    os.makedirs(static_pcd_path, exist_ok=True)
    depth_dst_dir = os.path.join(data_dir, depth_dir)
    os.makedirs(depth_dst_dir, exist_ok=True)

    # skip masked unprojection (for tanks and temples)
    if skip_dynamic:
        for i, imgpath in enumerate(list(imagepaths)):
            img = cv2.imread(imgpath)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = resize_to_mast3r(img, dst_size=None)
            extrinsic = data["cam2worlds"][i]
            depth = depths[i].reshape(img.shape[:2])
            static_pcd_path = os.path.join(data_dir, static_dst_dir)
            static_pcd_save_path = os.path.join(static_pcd_path, f"{i:04d}_static.ply")
            unproject_depth(
                focal,
                extrinsic,
                img,
                depth,
                export_path=static_pcd_save_path,
                mask=None,
            )
            depth_save_path = os.path.join(depth_dst_dir, f"{i:05}_depth.npy")
            np.save(depth_save_path, depth.reshape(img_h, img_w))
        return

    dynamic_pcd_path = os.path.join(data_dir, dynamic_dst_dir)
    os.makedirs(dynamic_pcd_path, exist_ok=True)

    loader = list(zip(imagepaths, maskpaths))
    for i, (imgpath, maskpath) in enumerate(loader):
        img = cv2.imread(imgpath)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = resize_to_mast3r(img, dst_size=None)

        mask = cv2.imread(maskpath, cv2.IMREAD_GRAYSCALE)
        mask = resize_to_mast3r(mask, dst_size=None)
        mask = mask > 0

        extrinsic = data["cam2worlds"][i]
        depth = depths[i].reshape(img.shape[:2])

        dynamic_pcd_save_path = os.path.join(dynamic_pcd_path, f"{i:04d}_dynamic.ply")
        unproject_depth(
            focal, extrinsic, img, depth, export_path=dynamic_pcd_save_path, mask=mask
        )
        static_pcd_save_path = os.path.join(static_pcd_path, f"{i:04d}_static.ply")
        unproject_depth(
            focal, extrinsic, img, depth, export_path=static_pcd_save_path, mask=~mask
        )

        depth_save_path = os.path.join(depth_dst_dir, f"{i:05}_depth.npy")
        np.save(depth_save_path, depth.reshape(img_h, img_w))


def check_all_masks_false(maskpaths):
    for maskpath in maskpaths:
        # Load the image and convert it to a numpy array
        mask = np.array(Image.open(maskpath))
        if np.any(mask):
            return False

    return True


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--datadir", type=str, required=True)
    parser.add_argument("--mast3r_expname", type=str, required=True)
    parser.add_argument("--mask_name", type=str)

    args = parser.parse_args()
    skip_dynamic = False

    datadir = args.datadir
    mast3r_exp_dir = os.path.join(datadir, "mast3r_opt", args.mast3r_expname)

    # load mast3r format image
    imagepaths = sorted(glob(f"{datadir}/train/*.png"))

    # load dynamic mask
    maskpaths = sorted(glob(f"{datadir}/{args.mask_name}/*.png"))
    if len(maskpaths) == 0:
        maskpaths = sorted(
            glob(f"{datadir}/{args.mask_name}/*.jpg")
        )  # mask should not be jpg...
    print(f"\nload mask from {maskpaths[0]} ~ {maskpaths[-1]}\n")

    if check_all_masks_false(maskpaths):
        skip_dynamic = True
        print("\nNo Dynamic regions found. Skip dynamic unprojection\n")

    with open(os.path.join(mast3r_exp_dir, "global_params.pkl"), "rb") as f:
        global_params = pickle.load(f)
        mast3r_img_h = len(global_params["masks"][0])
        mast3r_img_w = len(global_params["masks"][0][0])

    mast3r_unprojection(
        mast3r_exp_dir, maskpaths, imagepaths, mast3r_img_w, mast3r_img_h, skip_dynamic
    )
