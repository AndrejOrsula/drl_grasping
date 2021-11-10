from typing import Dict
import numpy as np
import ocnn
import torch as th


def preprocess_stacked_octree_batch(
    observation: th.Tensor, device, separate_batches: bool = True
) -> Dict[str, th.Tensor]:
    # Note: Primordial magic is happening here,
    #       but there's no reason to tremble in fear.
    #       For your own good don't question it too much,
    #       it's just an optimised stacked octree batch...

    if not separate_batches:
        octrees = []
        for octree in observation.reshape(-1, observation.shape[-1]):
            # Get original octree size
            octree_size = np.frombuffer(buffer=octree[-4:], dtype="uint32", count=1)
            # Convert to tensor and append to list
            octrees.append(th.from_numpy(octree[: octree_size[0]]))
        # Make batch out of tensor (consisting of n-stacked frames)
        octree_batch = ocnn.octree_batch(octrees)

        # Get number of auxiliary observations encoded as float32 and parse them
        n_aux_obs_f32 = int(
            np.frombuffer(buffer=observation[0, 0, -8:-4], dtype="uint32", count=1)
        )
        aux_obs = th.from_numpy(
            np.frombuffer(
                buffer=observation[:, :, -(4 * n_aux_obs_f32 + 8) : -8].reshape(-1),
                dtype="float32",
                count=n_aux_obs_f32 * observation.shape[0] * observation.shape[1],
            ).reshape(observation.shape[:2] + (n_aux_obs_f32,))
        )

        return {"octree": octree_batch.to(device), "aux_obs": aux_obs.to(device)}

    else:
        octree_batches = []

        for octree_batch in np.split(observation, observation.shape[1], axis=1):
            octrees = []
            for octree in octree_batch:
                # Get original octree size
                octree_size = np.frombuffer(buffer=octree[-4:], dtype="uint32", count=1)
                # Convert to tensor and append to list
                octrees.append(th.from_numpy(octree[: octree_size[0]]))
            # Make batch out of tensor (consisting of one stack)
            octree_batches.append(ocnn.octree_batch(octrees).to(device))

        # Get number of auxiliary observations encoded as float32 and parse them
        n_aux_obs_f32 = int(
            np.frombuffer(buffer=observation[0, 0, -8:-4], dtype="uint32", count=1)
        )
        aux_obs = th.from_numpy(
            np.frombuffer(
                buffer=observation[:, :, -(4 * n_aux_obs_f32 + 8) : -8].reshape(-1),
                dtype="float32",
                count=n_aux_obs_f32 * observation.shape[0] * observation.shape[1],
            ).reshape(observation.shape[:2] + (n_aux_obs_f32,))
        )

        return {"octree": octree_batches, "aux_obs": aux_obs.to(device)}
