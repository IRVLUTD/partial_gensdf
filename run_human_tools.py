import os
import numpy as np
import open3d as o3d
from skimage import measure
import json
import logging
import argparse
from glob import glob

import torch
from model import GenSDF


class PartialSDF:
    def __init__(
        self,
        specs_file,
        checkpoint_file,
        volume_size=128,
        device="cuda:0",
        debug=False,
    ):
        self._logger = self._init_logger(debug)
        self._debug = debug
        self._device = device
        self._logger.info(f"Specs file: {specs_file}")
        self._logger.info(f"Checkpoint file: {checkpoint_file}")
        self._logger.info(f"Device: {self._device}")
        self._model = self._init_model(specs_file, checkpoint_file)
        self._recon_batch = 100000
        self._sampled_num = 20000
        self._vol_origin = [-1.0, -1.0, -1.0]

        # initialize volume
        self.initialize(volume_size)

    def process(self, points):
        self._logger.debug(f"Processing points...")
        pts = self._sample_points(points)
        with torch.no_grad():
            head = 0
            while head < self._vol_dim:
                end = min(head + self._recon_batch, self._vol_dim)
                query = self._volume[head:end, 0:3].unsqueeze(0)
                # inference defined in forward function per pytorch lightning convention
                pred_sdf = self._model(pts, query)
                # update the sdf values
                self._volume[head:end, 3] = pred_sdf
                head += self._recon_batch
        self._logger.debug(f"Points processed")

    def _sample_points(self, points):
        pts = points.copy()
        # recenter
        trans = np.mean(pts, axis=0)
        pts -= trans
        # normalize
        bbox_length = np.sqrt(np.sum((np.max(pts, axis=0) - np.min(pts, axis=0)) ** 2))
        pts /= bbox_length
        # sample points
        pts = (
            torch.from_numpy(pts)[torch.randperm(pts.shape[0])[0 : self._sampled_num]]
            .float()
            .unsqueeze(0)
            .to(self._device)
        )

        # save for later use
        self._trans = trans
        self._scale = bbox_length

        return pts

    def get_mesh(self):
        sdf_volume = self.volume_sdf
        verts, faces, normals, values = measure.marching_cubes(
            sdf_volume,
            level=0.0,
            spacing=[self._grid_size] * 3,
            method="lewiner",
        )
        verts = (verts + self._vol_origin) * self._scale + self._trans
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(verts)
        mesh.triangles = o3d.utility.Vector3iVector(faces)
        mesh.compute_vertex_normals()
        return mesh

    @property
    def volume(self):
        return self._volume.detach().cpu().numpy()

    @property
    def volume_sdf(self):
        return (
            self._volume[:, 3]
            .reshape(self._vol_size, self._vol_size, self._vol_size)
            .detach()
            .cpu()
            .numpy()
        )

    @property
    def volume_origin(self):
        return self._vol_origin

    @property
    def volume_size(self):
        return self._vol_size

    def save_volume(self, save_path):
        volume = {
            "volume": self.volume,
            "volume_origin": self._vol_origin,
            "volume_size": self._vol_size,
            "scale": self._scale,
            "trans": self._trans,
        }

        np.savez(save_path, **volume)

    def _init_logger(self, debug):
        logger = logging.getLogger("PartialSDF")
        logger.setLevel(logging.DEBUG if debug else logging.INFO)
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG if debug else logging.INFO)
        ch.setFormatter(formatter)
        logger.addHandler(ch)
        return logger

    def _load_specs(self, specs_file):
        self._logger.debug(f"Loading specs...")
        with open(specs_file) as f:
            specs = json.load(f)
        self._logger.debug(f"Specs loaded")
        return specs

    def _load_checkpoint(self, checkpoint_file):
        self._logger.debug(f"Loading checkpoint...")
        checkpoint = torch.load(
            checkpoint_file, map_location=lambda storage, loc: storage
        )
        self._logger.debug(f"Checkpoint loaded")
        return checkpoint

    def _init_model(self, specs_file, checkpoint_file):
        self._logger.debug(f"Initializing model")
        specs = self._load_specs(specs_file)
        checkpoint = self._load_checkpoint(checkpoint_file)
        model = GenSDF(specs, None).to(self._device)
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        self._logger.debug(f"Model initialized")
        return model

    def _init_volume(self):
        idx = torch.arange(0, self._vol_dim, 1, out=torch.LongTensor()).to(self._device)
        vol = torch.zeros(self._vol_dim, 4, device=self._device)
        # transform first 3 columns to be the x, y, z index
        vol[:, 2] = idx % self._vol_size
        vol[:, 1] = (idx.long().float() / self._vol_size) % self._vol_size
        vol[:, 0] = (
            (idx.long().float() / self._vol_size) / self._vol_size
        ) % self._vol_size

        # transform first 3 columns to be the x, y, z coordinate
        vol[:, 0] = (vol[:, 0] * self._grid_size) + self._vol_origin[2]
        vol[:, 1] = (vol[:, 1] * self._grid_size) + self._vol_origin[1]
        vol[:, 2] = (vol[:, 2] * self._grid_size) + self._vol_origin[0]

        vol.requires_grad = False

        return vol

    def initialize(self, volume_size=None):
        self._logger.debug(f"Initializing volume with size {volume_size}")
        if volume_size is not None:
            self._vol_size = volume_size
            self._vol_dim = volume_size**3
            self._grid_size = 2.0 / (volume_size - 1)
        self._volume = self._init_volume()
        self._logger.debug(f"Volume initialized")


def args_parser():
    args = argparse.ArgumentParser(description="Partial SDF")
    args.add_argument(
        "--folder",
        dest="sequence_folder",
        type=str,
        help="path to the sequence folder",
        required=True,
    )
    args.add_argument(
        "--volume_size",
        dest="vol_size",
        type=int,
        help="volume size",
        default=256,
    )
    args.add_argument(
        "--debug",
        dest="debug",
        action="store_true",
        help="debug mode",
    )
    return args.parse_args()


if __name__ == "__main__":
    proj_dir = os.path.dirname(os.path.abspath(__file__))
    specs_file = os.path.join(proj_dir, "config/gensdf/partial/specs.json")
    ckpt_file = os.path.join(proj_dir, "config/gensdf/partial/epoch=92501.ckpt")

    args = args_parser()
    processing_folder = os.path.join(
        args.sequence_folder, "data_processing/reconstruction"
    )
    volume_size = args.vol_size

    volume_size = 256
    partSDF = PartialSDF(
        specs_file, ckpt_file, volume_size=volume_size, debug=args.debug
    )

    # load points
    pcd_merged = o3d.geometry.PointCloud()
    for pcd_file in sorted(glob(os.path.join(processing_folder, "pcds/*.ply"))):
        pcd = o3d.io.read_point_cloud(pcd_file)
        pcd_merged += pcd
    points = np.asarray(pcd_merged.points, dtype=np.float64)

    # process points
    partSDF.process(points)

    # save mesh
    mesh = partSDF.get_mesh()
    o3d.io.write_triangle_mesh(
        os.path.join(processing_folder, f"obj_mesh_gensdf.ply"), mesh, write_ascii=True
    )
