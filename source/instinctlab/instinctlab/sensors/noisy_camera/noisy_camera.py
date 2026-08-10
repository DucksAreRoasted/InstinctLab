from __future__ import annotations

import inspect
import torch
from prettytable import PrettyTable
from typing import Sequence

import warp as wp

from isaaclab.utils import string_to_callable
from isaaclab.utils.warp import ProxyArray

from instinctlab.utils.buffers import AsyncCircularBuffer
from instinctlab.utils.noise import ImageNoiseCfg


class NoisyCameraMixin:  # as a subclass of SensorBase
    """
    This mixin class adds noise to the camera data.
    """

    def __str__(self) -> str:
        return_ = super().__str__()
        noise_info_table = PrettyTable()
        noise_info_table.field_names = ["Noise Name", "Noise Cfg Name"]
        for noise_name, noise_cfg in self.cfg.noise_pipeline:  # type: ignore
            noise_info_table.add_row([noise_name, noise_cfg.__name__])
        return_ += "\n" + str(noise_info_table)
        history_info_table = PrettyTable()
        history_info_table.field_names = ["History Name", "History Length"]
        for history_name, history_length in self.cfg.data_histories.items():  # type: ignore
            history_info_table.add_row([history_name, history_length])
        return_ += "\n" + str(history_info_table)
        return return_

    def _initialize_impl(self):
        super()._initialize_impl()
        self.build_noise_pipeline()
        self.build_history_buffers()

    def reset(self, env_ids: Sequence[int] | None = None, env_mask: wp.array | None = None):
        """Reset the sensor, noise pipeline, and history buffers."""
        super().reset(env_ids, env_mask)
        if env_ids is None and env_mask is not None:
            env_ids = wp.to_torch(env_mask).nonzero(as_tuple=False).squeeze(-1)
        self.reset_noise_pipeline(env_ids)
        self.reset_history_buffers(env_ids)

    def _update_buffers_impl(self, env_mask: wp.array):
        """Update the camera output, noise pipeline, and history buffers."""
        super()._update_buffers_impl(env_mask)
        env_ids = wp.to_torch(env_mask).nonzero(as_tuple=False).squeeze(-1)
        self.apply_noise_pipeline_to_all_data_types(env_ids)
        self.update_history_buffers(env_ids)

    """
    Noise Pipeline
    """

    def build_noise_pipeline(self):
        self.noise_pipeline: Sequence[ImageNoiseCfg] | list[ImageNoiseCfg] = []
        """Build the noise pipeline based on the configuration."""

        for noise_name, noise_cfg in self.cfg.noise_pipeline.items():  # type: ignore
            # Check if the noise configuration is valid
            if not isinstance(noise_cfg, ImageNoiseCfg):
                raise ValueError(f"Invalid noise configuration for {noise_name}: {noise_cfg}")

            noise_cfg.device = self.device
            # Ensure the device is set correctly if the function is not a class

            if isinstance(noise_cfg.func, str):
                noise_cfg.func = string_to_callable(noise_cfg.func)

            if inspect.isclass(noise_cfg.func):
                # If the function is a class, instantiate it
                noise_cfg.func = noise_cfg.func(noise_cfg, num_envs=self.num_instances, device=self.device)

            # Add the noise configuration to the pipeline
            self.noise_pipeline.append(noise_cfg)

        # apply the noise pipeline to the initialized output buffers for noised output
        env_ids = torch.arange(self.num_instances, device=self.device)
        for data_type in self.cfg.data_types:
            noised_output = self.apply_noise_pipeline(self._output_as_torch(data_type), env_ids=env_ids)
            self._data.output[f"{data_type}_noised"] = ProxyArray(wp.from_torch(noised_output.contiguous()))

    def _output_as_torch(self, data_type: str) -> torch.Tensor:
        """Return the zero-copy Torch view used by the noise and history algorithms."""
        return self._data.output[data_type].torch

    def apply_noise_pipeline(self, data: torch.Tensor, env_ids: torch.Tensor | Sequence[int]) -> torch.Tensor:
        """Apply noise to the data(image).
        ## NOTE: The input data is only for selected envs (by env_ids).
        Args:
            data: The data to which noise will be applied. if Image, the shape should be (N_, H, W, C) for all environments.
        """
        # Check if the noise sequence is built
        if self.noise_pipeline is None:
            raise RuntimeError("Noise sequence not built. Call build_noise_pipeline() first.")

        # Apply noise to the image by calling the noise pipeline one by one.
        data = data.clone()
        for noise_cfg in self.noise_pipeline:
            data = noise_cfg.func(data, noise_cfg, env_ids)  # type: ignore

        return data

    def apply_noise_pipeline_to_all_data_types(self, env_ids: torch.Tensor | Sequence[int]):
        """Apply the noise pipeline to all data types."""
        for data_type in self.cfg.data_types:
            self._output_as_torch(f"{data_type}_noised")[env_ids] = self.apply_noise_pipeline(
                self._output_as_torch(data_type)[env_ids], env_ids=env_ids
            )

    def reset_noise_pipeline(self, env_ids: Sequence[int] | None = None):
        """Reset the noise pipeline for the specified environment IDs."""
        if self.noise_pipeline is None:
            raise RuntimeError("Noise sequence not built. Call build_noise_pipeline() first.")

        for noise_cfg in self.noise_pipeline:
            if hasattr(noise_cfg.func, "reset"):
                noise_cfg.func.reset(env_ids)

    """
    History Buffers
    """

    def build_history_buffers(self):
        """Build the history buffers for the specified data types."""
        self.output_history_buffers: dict[str, AsyncCircularBuffer] = dict()

        for data_type, history_length in self.cfg.data_histories.items():
            self.output_history_buffers[data_type] = AsyncCircularBuffer(
                history_length, self.num_instances, self.device
            )
            output = self._output_as_torch(data_type)
            history_output = torch.zeros(
                (output.shape[0], history_length, *output.shape[1:]), dtype=output.dtype, device=output.device
            )
            # Warp supports at most four array dimensions. Store channels in the element dtype so the zero-copy
            # Torch view retains the public (N, history, H, W, C) layout.
            channel_dtype = wp.types.vector(length=output.shape[-1], dtype=wp.dtype_from_torch(output.dtype))
            self._data.output[f"{data_type}_history"] = ProxyArray(wp.from_torch(history_output, dtype=channel_dtype))

    def update_history_buffers(self, env_ids: torch.Tensor | Sequence[int]):
        """Append the history buffers for the specified data types and update the result in self._data.output.
        Only configured data types will be appended, so only env_ids are needed. Please call this function after all
        outputs are computed.
        """
        for data_type in self.cfg.data_histories.keys():
            self.output_history_buffers[data_type].append(self._output_as_torch(data_type)[env_ids], env_ids)
            self._output_as_torch(f"{data_type}_history")[env_ids] = self.output_history_buffers[data_type].__getitem__(
                batch_ids=env_ids
            )

    def reset_history_buffers(self, env_ids: torch.Tensor | Sequence[int]):
        """Reset the history buffers for the specified data types."""
        for data_type in self.cfg.data_histories.keys():
            self.output_history_buffers[data_type].reset(env_ids)
