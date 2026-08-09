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
import soundfile as sf
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field

from pbrAudioCommon import EntityManager
from pbrAudioCommon import debug_print, set_debug, set_debug_prefix

@dataclass
class GlobalNormalize:
    """
    Post-processing class for global normalization of synthesized audio forces.
    """
    entity_manager: EntityManager

    def __post_init__(self):
        config = self.entity_manager.get('config')

        set_debug(config.system.debug)
        set_debug_prefix(self.__class__.__name__)

    def process(self, dir_path: str) -> Dict[int, Dict[str, np.ndarray]]:
        """
        Normalize by the max value of all object tracks in dir_path/unprocessed.
        """
        config = self.entity_manager.get('config')
        obj_tracks = {}
        maxs = []
        for config_obj in config.objects:
            obj_json = f"{dir_path}/unprocessed/{config_obj.name}.json"
            if os.path.exists(obj_json):
                with open(obj_json, 'r') as f:
                    obj_config = json.load(f)
                    sample_rate = obj_config.sample_rate
                    for idx in range(len(obj_config.tracks)):
                        track_data = {}
                        track_name = obj_config.tracks[idx].name
                        track_filename = f"{dir_path}/unprocessed/{obj_config.tracks[idx].file}"
                        track_audio, _ = sf.read(track_filename, samplerate=sample_rate, channels=1, subtype='FLOAT', always_2d=True)
                        track_data[track_name] = track_audio
                        maxs.append(np.max(np.abs(track_data[track_name]), axis=0))
                        obj_tracks[obj_config.idx] = track_data[track_name]

        all_max = max(maxs)

        for obj_idx, data in obj_tracks.items():
            for track_name, track_data in data.items():
                track_data /= all_max
                obj_tracks[obj_idx][track_name] = track_data

        return obj_tracks
