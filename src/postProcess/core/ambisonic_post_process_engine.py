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

import os
import numpy as np
import soundfile as sf
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from dask import delayed, compute

from physicsSolver import EntityManager
from physicsSolver.lib.functions import _update_status
from ..lib.ambisonic_decoder import AmbisonicDecoder

@dataclass
class AmbisonicPostProcessEngine:
    """
    Engine that orchestrates post-processing of all rendered ambisonic audio tracks.
    Integrates with the existing pbrAudioRender engines pipeline.
    """

    entity_manager: EntityManager

    def __post_init__(self):
        config = self.entity_manager.get('config')
        self.status_dir = f"{config.system.cache_path}/status/RenderPostProcessEngine"
        os.makedirs(self.status_dir, exist_ok=True)
        # Create output directory
        output_path = config.system.output_path
        os.makedirs(output_path, exist_ok=True)


    def process(self):
        """Post-process rendered ambisonic files"""

        config = self.entity_manager.get('config')
        render_path = config.system.output_path
        output_path = config.system.output_path

        if config.system.output_format == 'SURROUND':
            # Get surround configuration
            bit_depth = config.system.bit_depth
            sample_rate = int(config.system.sample_rate)
            surround_format = config.system.surround_format
            file_format = config.system.file_format

            # Generate standard speaker arrangement based on channel count
            speaker_positions, num_channels, num_lfe, num_vog = self._get_speaker_positions(surround_format)

            # Compute channels number
            num_speakers = num_channels + num_lfe + num_vog

            # Find all ambisonic tracks
            render_path = config.system.render_path
            ambi_tracks = os.listdir(render_path)
            ambi_tracks = [x for x in ambi_tracks if x.endswith('.wav')]
            for ambi_track in ambi_tracks:
                track_config = {
                    "file_path": f"{render_path}/{ambi_track}",
                    "channels": 0,
                    "center_location": {"x": 0.0, "y": 0.0, "z": 0.0},
                    "boundaries": {}
                }

                # Create decoder with the environment config
                decoder = AmbisonicDecoder(config_data=track_config)

                # Decode for all speaker positions
                decoded_audio = self._decode_surround(decoder, speaker_positions)

                # Save surround track
                track_name = ambi_track.replace('.wav','')
                self._save_surround(decoded_audio, output_path, track_name, sample_rate, num_channels, num_lfe)

                # Save a configuration file for reference
                self._save_surround_config(output_path, track_name, speaker_positions, sample_rate, file_format)

        elif config.system.output_format == 'STEREO':
            if config.system.stereo_hrtf:
                # ToDo: Implement HRTF decoding
                print("HRTF decoding not yet implemented")
            else:
                # Standard stereo decoding
                print("Stereo decoding not yet implemented")
                #self._decode_stereo(render_path)

    def _get_speaker_positions(self, surround_format):
        """
        Generate standard speaker positions of configured format.
        Supports standard configurations up to NHK 22.2.

        Returns list of (azimuth, elevation, is_lfe) tuples.
        """
        config = self.entity_manager.get('config')
        # Standard speaker configurations (azimuth, elevation)
        standard_configs = {
            '21': {  # Stereo w/LFE
                'speakers': [(-30, 0), (30, 0)],
                'lfe': [0],
                'vog': []
            },
            'LCR': {  # LCR
                'speakers': [(-30, 0), (0, 0), (30, 0)],
                'lfe': [],
                'vog': []
            },
            'QUAD': {  # Quad
                'speakers': [(-45, 0), (45, 0), (-135, 0), (135, 0)],
                'lfe': [],
                'vog': []
            },
            '50': {  # 5.0
                'speakers': [(-30, 0), (30, 0), (0, 0), (-110, 0), (110, 0)],
                'lfe': [],
                'vog': []
            },
            '51': {  # 5.1
                'speakers': [(-30, 0), (30, 0), (0, 0), (-110, 0), (110, 0)],
                'lfe': [0],  # LFE at center
                'vog': []
            },
            '61': {  # 6.1
                'speakers': [(-30, 0), (30, 0), (0, 0), (-110, 0), (110, 0), (-180, 0)],
                'lfe': [0],
                'vog': []
            },
            '71': {  # 7.1
                'speakers': [(-30, 0), (30, 0), (0, 0), (-110, 0), (110, 0), (-135, 0), (135, 0)],
                'lfe': [0],
                'vog': []
            },
            '91': {  # 9.1
                'speakers': [(-30, 0), (30, 0), (0, 0), (-110, 0), (110, 0),
                            (-135, 0), (135, 0), (-45, 45), (45, 45)],
                'lfe': [0],
                'vog': [0]
            },
            '101': {  # 10.1
                'speakers': [(-30, 0), (30, 0), (0, 0), (-110, 0), (110, 0),
                            (-30, 0), (30, 0), (0, 45), (-110, 45), (110, 45)],
                'lfe': [0],
                'vog': [0]
            },
            '111': {  # 11.1
                'speakers': [(-30, 0), (30, 0), (0, 0), (-110, 0), (110, 0),
                            (-135, 0), (135, 0), (-45, 45), (45, 45), (-90, 45), (90, 45)],
                'lfe': [0],
                'vog': [0]
            },
            '151': {  # 15.1 (Sony 360 Reality Audio)
                'speakers': [(-30, 0), (30, 0), (0, 0), (-110, 0), (110, 0),
                            (-135, 0), (135, 0), (-45, 45), (45, 45), (-90, 45), (90, 45),
                            (-45, -30), (45, -30), (-135, -30), (135, -30)],
                'lfe': [0],
                'vog': [0]
            },
            '222': {  # 22.2 (NHK)
                'speakers': [
                    # Bottom layer (z=-30°)
                    (-45, -30), (45, -30), (-135, -30), (135, -30),
                    # Middle layer (z=0°)
                    (-30, 0), (30, 0), (0, 0), (-110, 0), (110, 0),
                    (-135, 0), (135, 0), (-180, 0),
                    # Top layer (z=45°)
                    (-45, 45), (45, 45), (-90, 45), (90, 45), (-135, 45), (135, 45), (0, 45)],
                'lfe': [0, 0],  # Two LFE channels
                'vog': [0]
            }
        }

        # Find the selected standard configuration
        standard = standard_configs[surround_format]

        # Compute channels number 
        num_channels, num_lfe, num_vog = (0 for _ in range(3))
        num_lfe = len(standard['lfe'])
        if config.system.enable_vog:
            num_vog = len(standard['vog'])
        num_channels = len(standard['speakers']) + num_lfe + num_vog

        # Build speaker positions list
        positions = []
   
        # Add main speakers
        for i, (azimuth, elevation) in enumerate(standard['speakers']):
            is_lfe = False
            positions.append((azimuth, elevation, is_lfe))
   
        # Add LFE channels
        for i in range(len(standard['lfe'])):
            azimuth = 0  # LFE typically at center
            elevation = 0
            is_lfe = True
            positions.append((azimuth, elevation, is_lfe))

        # Add VOG channel
        if config.system.enable_vog:
            for i in range(len(standard['vog'])):
                azimuth = 0  # VOG typically at center above listeners
                elevation = 90
                is_lfe = False
                positions.append((azimuth, elevation, is_lfe))
        return positions, num_channels, num_lfe, num_vog

    def _decode_surround(self, decoder, speaker_positions):
        """
        Decode ambisonic audio for all speaker positions.

        Returns dict with speaker index -> audio array
        """
        decoded = {}

        for i, (azimuth, elevation, is_lfe) in enumerate(speaker_positions):
            # Decode for this speaker position
            audio = decoder.decode_to_position(azimuth, elevation)

            # Apply LFE filtering if needed
            if is_lfe:
                audio = self._apply_lfe_filter(audio, decoder.sample_rate)

            decoded[i] = audio

        return decoded

    def _apply_lfe_filter(self, audio, sample_rate):
        """
        Apply low-pass filter for LFE channel (typically 120Hz cutoff).
        Uses a simple butterworth-like filter.
        """
        from scipy import signal

        # Design a 4th order low-pass filter at 120Hz
        nyquist = sample_rate / 2
        cutoff = 120.0 / nyquist  # 120Hz cutoff

        if cutoff >= 1.0:
            return audio  # Can't filter at this sample rate

        # Use butterworth filter
        b, a = signal.butter(4, cutoff, btype='low')

        # Apply filter
        filtered = signal.filtfilt(b, a, audio)

        return filtered

    def _save_surround(self, decoded_audio, output_path, track_name, sample_rate, num_channels, num_lfe):
        """Save surround audio as multichannel WAV file."""
        config = self.entity_manager.get('config')
    
        # Ensure all channels have the same length
        lengths = [len(audio) for audio in decoded_audio.values()]
        max_length = max(lengths)
            
        # Create multichannel array
        multichannel = np.zeros((max_length, num_channels))
   
        for i, audio in decoded_audio.items():
            multichannel[:len(audio), i] = audio
    
        # Normalize to prevent clipping
        max_val = np.max(np.abs(multichannel))
        if max_val > 0:
            multichannel = multichannel / max_val * 0.95

        # Save as WAV
        output_file = os.path.join(output_path, f"{track_name}_surround.wav")

        # Determine subtype based on bit depth
        if bit_depth == '16':
            subtype = 'PCM_16'
        elif bit_depth == '24':
            subtype = 'PCM_24'
        elif bit_depth == '32':
            subtype = 'PCM_32'
        elif bit_depth == 'FLOAT':
            subtype = 'FLOAT'
        elif bit_depth == 'DOUBLE':
            subtype = 'DOUBLE'
        else:
            subtype = 'FLOAT'

        sf.write(output_file, multichannel, sample_rate, subtype=subtype)

        self.report({'INFO'}, f"Saved surround WAV: {output_file}")

    def _save_surround_config(self, output_path, track_name, speaker_positions, sample_rate, file_format):
        """Save configuration file for the surround output."""
        config = {
            'track_name': track_name,
            'sample_rate': sample_rate,
            'file_format': file_format,
            'num_channels': len(speaker_positions),
            'num_lfe': sum(1 for _, _, is_lfe in speaker_positions if is_lfe),
            'speakers': []
        }
   
        for i, (azimuth, elevation, is_lfe) in enumerate(speaker_positions):
            speaker = {
                'channel': i,
                'azimuth': azimuth,
                'elevation': elevation,
                'is_lfe': is_lfe
            }
            config['speakers'].append(speaker)

        # Save configuration
        config_file = os.path.join(output_path, f"{track_name}_surround_config.json")
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)

        self.report({'INFO'}, f"Saved surround config: {config_file}")
