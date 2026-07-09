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
import json
import warnings
import numpy as np
import soundfile as sf
from scipy import signal
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from dask import delayed, compute

from physicsSolver import EntityManager
from physicsSolver.lib.functions import _update_status
from ..lib.ambisonic_decoder import AmbisonicDecoder
from ..lib.ambisonic_to_stereo_hrtf import AmbisonicToStereoHRTF

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
        bit_depth = config.system.bit_depth
        sample_rate = int(config.system.sample_rate)
        file_format = config.system.file_format
        render_path = config.system.render_path

        if config.system.output_format == 'SURROUND':
            # Get surround configuration
            surround_format = config.system.surround_format

            # Generate standard speaker arrangement based on channel count
            speaker_positions, num_channels, num_lfe, num_vog = self._get_speaker_positions(surround_format)

            # Compute channels number
            num_speakers = num_channels + num_lfe + num_vog

            # Find all ambisonic tracks
            ambi_tracks = os.listdir(render_path)
            ambi_tracks = [x for x in ambi_tracks if x.endswith('.wav')]
            for ambi_track in ambi_tracks:
                track_config = {
                    "file_path": f"{render_path}/{ambi_track}",
                    "channels": 0,
                    "center_location": {"x": 0.0, "y": 0.0, "z": 0.0},
                    "boundaries": {}
                }

                # Create decoder with the track config
                decoder = AmbisonicDecoder(config_data=track_config)

                # Decode for all speaker positions
                decoded_audio = self._decode_surround(decoder, speaker_positions)

                # Save surround track
                track_name = ambi_track.replace('.wav','')
                self._save_surround(decoded_audio, output_path, track_name, sample_rate, num_channels, num_lfe)

                # Save a configuration file for reference
                self._save_surround_config(output_path, track_name, speaker_positions, sample_rate, file_format)

        elif config.system.output_format == 'STEREO':
            if config.system.stereo_format == 'ProLogicII':
                # Set surround configuration to 5.1
                surround_format = "51"

                # Generate standard speaker arrangement based on channel count
                speaker_positions, num_channels, num_lfe, num_vog = self._get_speaker_positions(surround_format)

                # Compute channels number
                num_speakers = num_channels + num_lfe + num_vog

                # Find all ambisonic tracks
                ambi_tracks = os.listdir(render_path)
                ambi_tracks = [x for x in ambi_tracks if x.endswith('.wav')]
                for ambi_track in ambi_tracks:
                    track_config = {
                        "file_path": f"{render_path}/{ambi_track}",
                        "channels": 0,
                        "center_location": {"x": 0.0, "y": 0.0, "z": 0.0},
                        "boundaries": {}
                    } 

                    # Create decoder with the track config
                    decoder = AmbisonicDecoder(config_data=track_config)

                    # Decode for all speaker positions
                    decoded_audio = self._decode_surround(decoder, speaker_positions)

                    # Save surround track as stereo ProLogic II
                    track_name = ambi_track.replace('.wav','')
                    self._pro_logic_ii_downmix(decoded_audio, output_path, track_name, sample_rate, file_format)

                    # Save a configuration file for reference
                    self._save_surround_config(output_path, track_name, speaker_positions, sample_rate, file_format)

            elif config.system.stereo_format == 'HRTF':
                # Ambisonic to stereo using HRTF binaural render
                # Find all ambisonic tracks
                ambi_tracks = os.listdir(render_path)
                ambi_tracks = [x for x in ambi_tracks if x.endswith('.wav')]
                for ambi_track in ambi_tracks:
                    ambi_data, sr = sf.read(ambi_track)
                    ambi_channels = ambi_data.shape[1]
                    ambi_order = int(np.sqrt(ambi_channels) - 1)

                    # Create decoder
                    decoder = AmbisonicToStereoHRTF(hrtf_path=config.system.hrtf_file, sample_rate=sample_rate, ambisonic_order=ambi_order)

                    # Decode to stereo
                    decoded_audio = decoder.decode(ambi_data)

                    # Save track as stereo
                    track_name = ambi_track.replace('.wav','_HRTF')
                    self._save_stereo(decoded_audio, output_path, track_name, sample_rate, file_format)

                    # Save a configuration file for reference
                    speaker_positions = [(0, 0), (45, 0), (90, 0), (135, 0), (180, 0), (225, 0), (270, 0), (315, 0)]
                    self._save_surround_config(output_path, track_name, speaker_positions, sample_rate, file_format)

            else:
                # Get stereo configuration
                stereo_format = config.system.stereo_format

                # Generate standard speaker arrangement
                speaker_positions, num_speakers, _, _ = self._get_speaker_positions(stereo_format)

                # Find all ambisonic tracks
                ambi_tracks = os.listdir(render_path)
                ambi_tracks = [x for x in ambi_tracks if x.endswith('.wav')]
                for ambi_track in ambi_tracks:
                    track_config = {
                        "file_path": f"{render_path}/{ambi_track}",
                        "channels": 0,
                        "center_location": {"x": 0.0, "y": 0.0, "z": 0.0},
                        "boundaries": {}
                    }

                    # Create decoder with the track config
                    decoder = AmbisonicDecoder(config_data=track_config)

                    # Decode for all speaker positions
                    decoded_audio = self._decode_surround(decoder, speaker_positions)

                    # Save surround track as stereo ProLogic II
                    track_name = ambi_track.replace('.wav','')
                    self._save_stereo(decoded_audio, output_path, track_name, sample_rate, file_format)

                    # Save a configuration file for reference
                    self._save_surround_config(output_path, track_name, speaker_positions, sample_rate, file_format)

    def _save_stereo(self, decoded_audio, output_path, track_name, sample_rate, file_format, normalize=True):
        """
        Save stereo track
        """
        if decoded_audio.ndim != 2 or decoded_audio.shape[1] != 2:
            raise ValueError("Input must be a 2-channel audio file")

        if normalize:
            max_val = np.max(np.abs(stereo))
            if max_val > 0:
                stereo = stereo / max_val

        # Save as WAV
        output_file = os.path.join(output_path, f"{track_name}_stereo.wav")

        # Determine subtype based on bit depth
        bit_depth = config.system.bit_depth
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

        sf.write(output_file, stereo, sample_rate, subtype=subtype)

        print(f"Saved stereo WAV: {output_file}")

    def _pro_logic_ii_downmix(self, decoded_audio, output_path, track_name, sample_rate, file_format, center_gain=0.707, surround_gain=0.707, lfe_to_lr=True, normalize=True):
        """
        Convert 5.1 surround to stereo with Pro Logic-like encoding

        Pro Logic IIx matrix (more common):
        Lt = L + 0.707*C + 0.707*Ls + 0.707*Rs (90° phase shift)
        Rt = R + 0.707*C - 0.707*Ls - 0.707*Rs (90° phase shift)
        """

        if decoded_audio.ndim != 2 or decoded_audio.shape[1] != 6:
            raise ValueError("Input must be a 6-channel (5.1) audio file")

        FL = decoded_audio[:, 0]
        FR = decoded_audio[:, 1]
        FC = decoded_audio[:, 2]
        LFE = decoded_audio[:, 3]
        SL = decoded_audio[:, 4]
        SR = decoded_audio[:, 5]

        # Handle LFE
        if lfe_to_lr:
            FL += LFE * 0.5
            FR += LFE * 0.5

        # Apply 90° phase shift to surround channels for better stereo imaging
        # This is a simplified Hilbert transform approach
        phase_shift = np.exp(-1j * np.pi/2)  # -90 degrees

        # Apply phase shift using FFT
        n = len(SL)
        SL_fft = np.fft.fft(SL)
        SR_fft = np.fft.fft(SR)

        # Create frequency vector
        freqs = np.fft.fftfreq(n, 1/sample_rate)

        # Apply phase shift only to positive frequencies
        mask = freqs > 0
        SL_fft[mask] *= phase_shift
        SR_fft[mask] *= phase_shift

        # Inverse FFT
        SL_shifted = np.real(np.fft.ifft(SL_fft))
        SR_shifted = np.real(np.fft.ifft(SR_fft))

        # Pro Logic IIx matrix
        Lt = FL + (center_gain * FC) + (surround_gain * SL_shifted) + (surround_gain * SR_shifted)
        Rt = FR + (center_gain * FC) - (surround_gain * SL_shifted) - (surround_gain * SR_shifted)

        stereo = np.column_stack((Lt, Rt))

        # Normalize
        if normalize:
            max_val = np.max(np.abs(stereo))
            if max_val > 0:
                stereo = stereo / max_val

        # Save as WAV
        output_file = os.path.join(output_path, f"{track_name}_ProLogicII.wav")

        # Determine subtype based on bit depth
        bit_depth = config.system.bit_depth
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

        sf.write(output_file, stereo, sample_rate, subtype=subtype)

        print(f"Saved ProLogic II WAV: {output_file}")

    def _get_speaker_positions(self, surround_format):
        """
        Generate standard speaker positions of configured format.
        Supports standard configurations up to NHK 22.2.

        Returns list of (azimuth, elevation, is_lfe) tuples.
        """
        config = self.entity_manager.get('config')
        # Standard speaker configurations (azimuth, elevation)
        standard_configs = {
            '90': {
                'speakers': [(-45, 0), (45, 0)],
                'lfe': [],
                'vog': []
            },  
            '120': {
                'speakers': [(-60, 0), (60, 0)],
                'lfe': [],
                'vog': []
            },  
            '180': {
                'speakers': [(-90, 0), (90, 0)],
                'lfe': [],
                'vog': []
            },
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
        bit_depth = config.system.bit_depth
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

        print(f"Saved surround WAV: {output_file}")

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

        print(f"Saved surround config: {config_file}")
