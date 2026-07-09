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
from scipy import signal
from scipy.io import wavfile
import warnings
from typing import Optional, Tuple, Union

class AmbisonicToStereoHRTF:
    """
    Decode ambisonic audio (FOA/HOA) to stereo using HRTF-based binaural rendering.
    Supports first-order ambisonics (FOA) up to higher orders.
    """
    
    def __init__(self, 
                 hrtf_path: Optional[str] = None,
                 sample_rate: int = 44100,
                 ambisonic_order: int = 1):
        """
        Initialize the decoder.
        
        Args:
            hrtf_path: Path to SOFA HRTF file (optional, will use default if None)
            sample_rate: Target sample rate for processing
            ambisonic_order: Ambisonic order (1 for FOA, 2 for HOA2, etc.)
        """
        self.sample_rate = sample_rate
        self.ambisonic_order = ambisonic_order
        self.num_channels = (ambisonic_order + 1) ** 2
        
        # Load HRTF data
        self.hrtf_data = None
        self.hrtf_azimuths = None
        self.hrtf_elevations = None
        
        if hrtf_path and os.path.exists(hrtf_path):
            self._load_hrtf(hrtf_path)
        else:
            # Use default MIT HRTF dataset (included with scipy)
            self._load_default_hrtf()
    
    def _load_default_hrtf(self):
        """Load default HRTF data (simplified model for demonstration)."""
        # For production, you'd want to load actual HRTF measurements
        # This is a simple approximation using interaural time/level differences
        n_angles = 72  # 5 degree resolution
        self.hrtf_azimuths = np.linspace(-180, 180, n_angles, endpoint=False)
        self.hrtf_elevations = np.array([0])
        
        # Create simple HRTF model (ITD + ILD)
        # In practice, load from SOFA files or similar
        self.hrtf_data = self._create_simple_hrtf_model(n_angles)
    
    def _create_simple_hrtf_model(self, n_angles: int) -> np.ndarray:
        """Create a simple HRTF model for demonstration purposes."""
        # This should be replaced with actual measured HRTF data
        # For now, we create a simplified model
        
        # Head radius in meters
        head_radius = 0.0875
        # Speed of sound in m/s
        c = 343
        
        # Create frequency bins
        n_fft = 256
        freqs = np.fft.rfftfreq(n_fft, 1/self.sample_rate)
        
        # Initialize HRTF data: [elevations, azimuths, ears, freq_bins]
        hrtf = np.zeros((len(self.hrtf_elevations), n_angles, 2, len(freqs)), dtype=complex)
        
        for i, az in enumerate(self.hrtf_azimuths):
            az_rad = np.deg2rad(az)
            
            # ITD calculation (simplified)
            itd = (head_radius / c) * (az_rad + np.sin(az_rad))
            
            # ILD calculation (simplified)
            ild_dB = 0.5 * np.sin(az_rad)  # Simple approximation
            
            # Apply ITD and ILD in frequency domain
            for ear_idx, ear_sign in enumerate([-1, 1]):  # Left, Right
                phase_shift = -2 * np.pi * freqs * itd * ear_sign
                amplitude = 10 ** (ild_dB * ear_sign / 20)
                hrtf[0, i, ear_idx] = amplitude * np.exp(1j * phase_shift)
        
        return hrtf
    
    def _load_hrtf(self, hrtf_path: str):
        """Load HRTF data from file (supports SOFA format)."""
        try:
            import sofa
            # Load SOFA file
            hrtf = sofa.Database(hrtf_path)
            
            # Extract data
            self.hrtf_data = hrtf.Data.IR.get_values()  # [n_measurements, n_ears, n_samples]
            self.hrtf_azimuths = hrtf.SourcePosition.get_values()[:, 0]
            self.hrtf_elevations = np.unique(hrtf.SourcePosition.get_values()[:, 1])
            
            # Resample if needed
            if hrtf.Data.SamplingRate.get_values()[0] != self.sample_rate:
                self._resample_hrtf(hrtf.Data.SamplingRate.get_values()[0])
                
        except ImportError:
            warnings.warn("SOFA library not installed. Using default HRTF model.")
            self._load_default_hrtf()
        except Exception as e:
            warnings.warn(f"Failed to load HRTF file: {e}. Using default model.")
            self._load_default_hrtf()
    
    def _resample_hrtf(self, original_rate: int):
        """Resample HRTF data to target sample rate."""
        # Calculate resampling ratio
        ratio = self.sample_rate / original_rate
        new_length = int(len(self.hrtf_data[0, 0]) * ratio)
        
        # Resample each channel
        resampled_data = np.zeros((len(self.hrtf_data), 2, new_length))
        for i in range(len(self.hrtf_data)):
            for ear in range(2):
                resampled_data[i, ear] = signal.resample(
                    self.hrtf_data[i, ear], new_length
                )
        
        self.hrtf_data = resampled_data
    
    def _calculate_ambisonic_gains(self, azimuth: float, elevation: float) -> np.ndarray:
        """Calculate ambisonic decoding gains for a given direction."""
        az_rad = np.deg2rad(azimuth)
        el_rad = np.deg2rad(elevation)
        
        if self.ambisonic_order == 1:
            # First-order ambisonic gains
            # ACN ordering: W, Y, Z, X
            gains = np.array([
                1.0,  # W (omnidirectional)
                np.sin(az_rad) * np.cos(el_rad),  # Y
                np.sin(el_rad),  # Z
                np.cos(az_rad) * np.cos(el_rad)  # X
            ])
        else:
            # For higher orders, use spherical harmonics
            gains = self._spherical_harmonics(az_rad, el_rad, self.ambisonic_order)
        
        return gains
    
    def _spherical_harmonics(self, az: float, el: float, order: int) -> np.ndarray:
        """Calculate real spherical harmonics up to given order."""
        from scipy.special import sph_harm
        
        gains = []
        for n in range(order + 1):
            for m in range(-n, n + 1):
                # Calculate spherical harmonic (real form)
                if m >= 0:
                    Y = np.real(sph_harm(m, n, az, np.pi/2 - el))
                else:
                    Y = np.imag(sph_harm(abs(m), n, az, np.pi/2 - el))
                gains.append(Y)
        
        return np.array(gains)
    
    def _find_nearest_angles(self, azimuth: float, elevation: float) -> Tuple[int, int]:
        """Find nearest HRTF indices for given direction."""
        az_idx = np.argmin(np.abs(self.hrtf_azimuths - azimuth))
        el_idx = np.argmin(np.abs(self.hrtf_elevations - elevation))
        return az_idx, el_idx
    
    def _apply_hrtf(self, signal_chunk: np.ndarray, azimuth: float, elevation: float) -> Tuple[np.ndarray, np.ndarray]:
        """Apply HRTF to create binaural signal for a virtual source."""
        # Find nearest HRTF
        az_idx, el_idx = self._find_nearest_angles(azimuth, elevation)
        
        # Get HRTF impulse responses for left and right ears
        hrtf_left = self.hrtf_data[el_idx, az_idx, 0]
        hrtf_right = self.hrtf_data[el_idx, az_idx, 1]
        
        # Convolve with signal
        left_channel = signal.fftconvolve(signal_chunk, hrtf_left, mode='same')
        right_channel = signal.fftconvolve(signal_chunk, hrtf_right, mode='same')
        
        return left_channel, right_channel
    
    def decode(self, 
               ambisonic_audio: np.ndarray,
               virtual_speakers: Optional[list] = None,
               output_type: str = 'binaural') -> np.ndarray:
        """
        Decode ambisonic audio to stereo.
        
        Args:
            ambisonic_audio: Input audio array [samples, channels]
            virtual_speakers: List of (azimuth, elevation) tuples for virtual speaker positions
                            If None, uses default 8-speaker layout
            output_type: 'binaural' or 'virtual_speakers'
            
        Returns:
            Decoded stereo audio [samples, 2]
        """
        # Validate input
        if len(ambisonic_audio.shape) != 2:
            raise ValueError("Input must be 2D array [samples, channels]")
        
        if ambisonic_audio.shape[1] != self.num_channels:
            raise ValueError(f"Expected {self.num_channels} channels, got {ambisonic_audio.shape[1]}")
        
        # Default virtual speaker layout (8 speakers for FOA)
        if virtual_speakers is None:
            if self.ambisonic_order == 1:
                virtual_speakers = [
                    (0, 0), (45, 0), (90, 0), (135, 0),
                    (180, 0), (225, 0), (270, 0), (315, 0)
                ]
            else:
                # For higher orders, add more elevation angles
                virtual_speakers = []
                for el in [-30, 0, 30]:
                    for az in range(0, 360, 45):
                        virtual_speakers.append((az, el))
        
        # Initialize output
        stereo_output = np.zeros((len(ambisonic_audio), 2))
        
        if output_type == 'virtual_speakers':
            # Simple virtual speaker decoding
            for az, el in virtual_speakers:
                # Calculate ambisonic gains for this direction
                gains = self._calculate_ambisonic_gains(az, el)
                
                # Apply gains and sum
                virtual_signal = np.dot(ambisonic_audio, gains)
                
                # Apply simple panning (could use HRTF here too)
                pan = (az + 180) / 360  # Convert to 0-1 range
                left_gain = np.sqrt(1 - pan)
                right_gain = np.sqrt(pan)
                
                stereo_output[:, 0] += virtual_signal * left_gain
                stereo_output[:, 1] += virtual_signal * right_gain
            
            # Normalize
            max_val = np.max(np.abs(stereo_output))
            if max_val > 0:
                stereo_output /= max_val
        
        else:  # binaural
            # Process each virtual speaker with HRTF
            for az, el in virtual_speakers:
                # Calculate ambisonic gains for this direction
                gains = self._calculate_ambisonic_gains(az, el)
                
                # Apply gains to create virtual source signal
                virtual_signal = np.dot(ambisonic_audio, gains)
                
                # Apply HRTF
                left, right = self._apply_hrtf(virtual_signal, az, el)
                
                # Accumulate
                stereo_output[:, 0] += left
                stereo_output[:, 1] += right
            
            # Normalize to prevent clipping
            max_val = np.max(np.abs(stereo_output))
            if max_val > 0:
                stereo_output *= 0.9 / max_val
        
        return stereo_output
    
    def decode_file(self, 
                    input_file: str, 
                    output_file: str,
                    **kwargs) -> None:
        """
        Decode ambisonic audio file to stereo.
        
        Args:
            input_file: Path to input WAV file
            output_file: Path to output WAV file
            **kwargs: Additional arguments for decode() method
        """
        # Read input file
        sample_rate, audio_data = wavfile.read(input_file)
        
        # Convert to float if needed
        if audio_data.dtype == np.int16:
            audio_data_data = audio_data.astype(np.float32) / 32768.0
        elif audio_data.dtype == np.int32:
            audio_data = audio_data.astype(np.float32) / 2147483648.0
        
        # If sample rate doesn't match, resample
        if sample_rate != self.sample_rate:
            # Resample each channel
            new_length = int(len(audio_data) * self.sample_rate / sample_rate)
            resampled = np.zeros((new_length, audio_data.shape[1]))
            for ch in range(audio_data.shape[1]):
                resampled[:, ch] = signal.resample(audio_data[:, ch], new_length)
            audio_data = resampled
        
        # Decode
        stereo_audio = self.decode(audio_data, **kwargs)
        
        # Convert back to int16 for saving
        stereo_audio_int = (stereo_audio * 32767).astype(np.int16)
        
        # Save output
        wavfile.write(output_file, self.sample_rate, stereo_audio_int)
    
    def process_stream(self, 
                       audio_stream: np.ndarray,
                       block_size: int = 1024,
                       **kwargs) -> np.ndarray:
        """
        Process streaming audio in blocks.
        
        Args:
            audio_stream: Input audio stream [samples, channels]
            block_size: Processing block size
            **kwargs: Additional arguments for decode()
            
        Returns:
            Processed stereo stream [samples, 2]
        """
        num_samples = len(audio_stream)
        output = np.zeros((num_samples, 2))
        
        # Process in blocks
        for start in range(0, num_samples, block_size):
            end = min(start + block_size, num_samples)
            block = audio_stream[start:end]
            
            # Process block
            output[start:end] = self.decode(block, **kwargs)
        
        return output
