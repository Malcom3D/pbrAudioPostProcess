# Copyright (C) 2025 Malcom3D <malcom3d.gpl@gmail.com>
#
# This file is part of pbrAudio.
#
# pbrAudio is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# pbrAudio is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with pbrAudio.  If not, see <https://www.gnu.org/licenses/>.
# SPDX-License-Identifier: GPL-3.0-or-later

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field

from ..lib.audio_forces_denoiser import AudioForcesDenoiser
from ..lib.global_normalize import GlobalNormalize

from pbrAudioCommon import EntityManager
from pbrAudioCommon import debug_print, set_debug, set_debug_prefix

@dataclass
class AudioForcePostProcessEngine:
    """
    Post-processing class for synthesized audio forces.
    """
    entity_manager: EntityManager

    def __post_init__(self):
        config = self.entity_manager.get('config')

        set_debug(config.system.debug)
        set_debug_prefix(self.__class__.__name__)

        self.audio_force_dir = f"{config.system.cache_path}/audio_force"
        self.unprocessed_dir = f"{self.audio_force_dir}/unprocessed"

        # inizialize Global normalizer
        self.global_normalizer = GlobalNormalize(self.entity_manager)

        # Initialize denoiser
        self.denoiser = AudioForcesDenoiser(
            dc_blocker_alpha=config.denoiser.dc_blocker_alpha,
            gate_threshold_db=config.denoiser.gate_threshold_db,
            gate_attack_ms=config.denoiser.gate_attack_ms,
            gate_release_ms=config.denoiser.gate_release_ms,
            gate_hold_ms=config.denoiser.gate_hold_ms,
            temporal_smoothing_window=config.denoiser.temporal_smoothing_window,
            spectral_fft_size=config.denoiser.spectral_fft_size,
            spectral_hop_size=config.denoiser.spectral_hop_size,
            spectral_noise_floor_db=config.denoiser.spectral_noise_floor_db,
            spectral_reduction_strength=config.denoiser.spectral_reduction_strength,
            spectral_smoothing=config.denoiser.spectral_smoothing,
            envelope_attack_ms=config.denoiser.envelope_attack_ms,
            envelope_release_ms=config.denoiser.envelope_release_ms,
            envelope_smoothing=config.denoiser.envelope_smoothing,
            gaussian_sigma_min=config.denoiser.gaussian_sigma_min,
            gaussian_sigma_max=config.denoiser.gaussian_sigma_max,
            gaussian_force_threshold=config.denoiser.gaussian_force_threshold
        )

    def process(self):
        """
        Globaly normalize and Denoise audio forces
        """
        config = self.entity_manager.get('config')
        forces = self.entity_manager.get('forces')
        sample_rate = config.system.sample_rate

        # ToDo: switch from unprocessed to dir 
        obj_tracks = self.global_normalizer.process(self.unprocessed_dir)

        for obj_idx, tracks_data in obj_tracks.items():
            for config_obj in config.objects:
                if config_obj.idx == obj_idx: 
                    for f_idx in forces.keys():
                        if forces[f_idx].obj_idx == obj_idx:
                            force_data_sequence = forces[f_idx]
                            tracks = self.denoiser.process(tracks_data, force_data_sequence, sample_rate)
                            self._save_tracks(config_obj, tracks, int(sample_rate))

    def _save_tracks(self, config_obj: Any, tracks: Dict[str, np.ndarray], sample_rate: int, unprocessed: bool = False):
        """
        Save individual tracks as WAV files.
        Create a json multitrack project file (e.g., for Reaper, Ardour).
        """
        project_data = {
            'object_name': config_obj.name,
            'sample_rate': sample_rate,
            'tracks': []
        }
        for track_name, track_data in tracks.items():
            track_file = f"{config_obj.name}_{track_name}.raw"
            wave_file = f"{self.audio_force_dir}/{track_file}"
            sf.write(wave_file, track_data, sample_rate, subtype='FLOAT')
            project_data['tracks'].append({
                'name': track_name,
                'file': track_file,
                'channels': 1,
                'position': 0.0,
                'volume': 1.0,
                'pan': 0.0
            })
            debug_print(f"Saved {track_name} tracks to {self.audio_force_dir}")

        # Save project file
        json_file = f"{self.audio_force_dir}/{config_obj.name}.json"

        with open(json_file, 'w') as f:
            json.dump(project_data, f, indent=2)

        debug_print(f"Created multitrack project: {json_file}")
