from typing import Dict

import numpy as np
import ocnn
import torch as th


def preprocess_stacked_octree_batch(
    observation: th.Tensor,
    device,
    separate_batches: bool = True,
    include_aux_obs: bool = True,
) -> Dict[str, th.Tensor]:
    # Note: Primordial magic is happening here,
    #       but there's no reason to tremble in fear.
    #       For your own good don't question it too much,
    #       it's just an optimised stacked octree batch...

    if not separate_batches:

        octrees = []
        for octree in observation.reshape(-1, observation.shape[-1]):
            # Get original octree size
            octree_size = np.frombuffer(buffer=octree[-4:], dtype=np.uint32, count=1)
            # Convert to tensor and append to list
            octrees.append(th.tensor(octree[: octree_size[0]], requires_grad=False))
        # Make batch out of tensor (consisting of n-stacked frames)
        octree_batches = ocnn.octree_batch(octrees).to(device)

    else:

        octree_batches = []
        for octree_batch in np.split(observation, observation.shape[1], axis=1):

            octrees = []
            for octree in octree_batch:
                # Get original octree size
                octree_size = np.frombuffer(
                    buffer=octree[-4:], dtype=np.uint32, count=1
                )
                # Convert to tensor and append to list
                octrees.append(th.tensor(octree[: octree_size[0]], requires_grad=False))
            # Make batch out of tensor (consisting of one stack)
            octree_batches.append(ocnn.octree_batch(octrees).to(device))

    # Get number of auxiliary observations encoded as float32 and parse them
    if include_aux_obs:
        n_aux_obs_f32 = int(
            np.frombuffer(buffer=observation[0, 0, -8:-4], dtype=np.uint32, count=1)
        )
        aux_obs = th.tensor(
            np.frombuffer(
                buffer=observation[:, :, -(4 * n_aux_obs_f32 + 8) : -8].reshape(-1),
                dtype=np.float32,
                count=n_aux_obs_f32 * observation.shape[0] * observation.shape[1],
            ).reshape(observation.shape[:2] + (n_aux_obs_f32,)),
            requires_grad=False,
        ).to(device)
    else:
        aux_obs = None

    return (octree_batches, aux_obs)


def preprocess_stacked_depth_image_batch(
    observation: th.Tensor,
    device,
    separate_batches: bool = True,
    image_width=128,
    image_height=128,
    include_aux_obs=True,
) -> Dict[str, th.Tensor]:

    number_of_pixels = image_width * image_height

    if observation.shape[2] >= 4 * number_of_pixels:
        contains_rgb = True
        contains_intensity = False
        num_channels = 4
    elif observation.shape[2] >= 2 * number_of_pixels:
        contains_rgb = False
        contains_intensity = True
        num_channels = 2
    else:
        contains_rgb = False
        contains_intensity = False
        num_channels = 1

    # Get number of auxiliary observations encoded as float32 and parse them
    if include_aux_obs:
        n_aux_obs = int(
            np.round(
                np.frombuffer(buffer=observation[0, 0, -1], dtype=np.float32, count=1)
            )
        )
        aux_obs = th.tensor(
            observation[:, :, -(n_aux_obs + 1) : -1].reshape(
                observation.shape[:2] + (n_aux_obs,)
            ),
            requires_grad=False,
        ).to(device)
    else:
        aux_obs = None

    if not separate_batches:

        image_batches = []
        for image in observation.reshape(-1, observation.shape[-1]):
            # Convert to images, without aux obs
            if contains_rgb or contains_intensity:
                _depth_image, color_image = np.split(
                    image[: 4 * number_of_pixels], [number_of_pixels]
                )

                if contains_intensity:
                    depth_image = np.empty(
                        (2 * number_of_pixels,), dtype=_depth_image.dtype
                    )
                    depth_image[0::2] = color_image[:number_of_pixels]
                    depth_image[1::2] = _depth_image
                else:
                    depth_image = np.empty(
                        (4 * number_of_pixels,), dtype=_depth_image.dtype
                    )
                    depth_image[0::4] = color_image[0::3]
                    depth_image[1::4] = color_image[1::3]
                    depth_image[2::4] = color_image[2::3]
                    depth_image[3::4] = _depth_image
            else:
                depth_image = image[:number_of_pixels]

            depth_image = depth_image.reshape(
                -1, num_channels, image_height, image_width
            )

            # Convert to tensor and append to image list
            image_batches.append(th.tensor(depth_image, requires_grad=False).to(device))

        image_batches = th.stack(image_batches)

        image_batches = image_batches.view(-1, num_channels, image_height, image_width)

    else:

        image_batches = []
        for image_batch in np.split(observation, observation.shape[1], axis=1):

            images = []
            for image in image_batch:
                # Convert to images, without aux obs
                if contains_rgb or contains_intensity:
                    _depth_image, color_image = np.split(
                        image[:, : 4 * number_of_pixels], [number_of_pixels], axis=1
                    )

                    if contains_intensity:
                        depth_image = np.empty(
                            (_depth_image.shape[0], 2 * number_of_pixels),
                            dtype=_depth_image.dtype,
                        )
                        depth_image[:, 0::2] = color_image
                        depth_image[:, 1::2] = _depth_image
                    else:
                        depth_image = np.empty(
                            (
                                _depth_image.shape[0],
                                4 * number_of_pixels,
                            ),
                            dtype=_depth_image.dtype,
                        )
                        depth_image[:, 0::4] = color_image[:, 0::3]
                        depth_image[:, 1::4] = color_image[:, 1::3]
                        depth_image[:, 2::4] = color_image[:, 2::3]
                        depth_image[:, 3::4] = _depth_image
                else:
                    depth_image = image[:, :number_of_pixels]

                depth_image = depth_image.reshape(-1, image_height, image_width)

                # Convert to tensor and append to image list
                images.append(th.tensor(depth_image, requires_grad=False))

            # Make batch out of tensor (consisting of one stack)
            image_batches.append(th.stack(images).to(device))

    return (image_batches, aux_obs)
