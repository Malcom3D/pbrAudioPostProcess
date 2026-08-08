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

        audio_force_dir = f"{config.system.cache_path}/audio_force"

    def process(self, unprocessed_path: str) -> Dict[int, Dict[str, np.ndarray]]:
        """
        Normalize by the max value of all tracks in unprocessed_path.
        """
        config = self.entity_manager.get('config')
        obj_tracks = {}
        maxs = []
        for config_obj in config.objects:
            obj_json = f"{unprocessed_path}/{config_obj.name}.json"
            if os.path.exists(obj_json):
                with open(obj_json, 'r') as f:
                    obj_config = json.load(f)
                    sample_rate = obj_config.sample_rate
                    for idx in range(len(obj_config.tracks)):
                        track_data = {}
                        track_name = obj_config.tracks[idx].name
                        track_data[track_name] = np.fromfile(f"{unprocessed_path}/{obj_config.tracks[idx].file}", dtype=np.float32).reshape((-1,1))
                        maxs.append(np.max(track_data[track_name]))
                        obj_tracks[obj_config.idx] = track_data[track_name]

        all_max = max(maxs)

        for obj_idx, data in obj_tracks.items():
            for track_name, track_data in data.items():
                track_data /= all_max
                obj_tracks[obj_idx][track_name] = track_data

        return obj_tracks
