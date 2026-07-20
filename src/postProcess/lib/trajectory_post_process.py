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

# pbrAudioPostProcess/src/postProcess/lib/trajectory_post_process.py

import numpy as np
from scipy import signal
from scipy.ndimage import gaussian_filter1d
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field

from physicsSolver import EntityManager
from physicsSolver.lib.trajectory_data import TrajectoryData

@dataclass
class TrajectoryPostProcess:
    """
    Post-processor for TrajectoryData that corrects bounce artifacts
    in vertex trajectories during continuous contact with surfaces.
    
    Uses rigid body physics constraints to detect and correct:
    - Bounce artifacts (oscillations near contact surfaces)
    - Penetration artifacts (vertices going through surfaces)
    - Jitter artifacts (high-frequency noise in vertex positions)
    """
    
    entity_manager: EntityManager

    def __post_init__(self):
        # Detection parameters
        self.bounce_threshold = config.trajectory_postprocess.bounce_threshold
#        self.bounce_frequency_range = config.trajectory_postprocess.bounce_frequency_range
#        self.penetration_margin = config.trajectory_postprocess.penetration_margin
    
        # Correction parameters
        self.correction_strength = config.trajectory_postprocess.correction_strength
        self.smoothing_sigma = config.trajectory_postprocess.smoothing_sigma
        self.min_contact_duration = config.trajectory_postprocess.min_contact_duration
    
        # Physics constraints
        self.max_velocity_change = config.trajectory_postprocess.max_velocity_change
        self.max_angular_velocity = config.trajectory_postprocess.max_angular_velocity
    
        self.config = self.entity_manager.get('config')
        self.sample_rate = self.config.system.sample_rate
        self.sfps = (self.config.system.fps / self.config.system.fps_base) * self.config.system.subframes
        
    def process_trajectory(self, trajectory: TrajectoryData) -> TrajectoryData:
        """
        Main processing function for a single trajectory.
        
        Args:
            trajectory: TrajectoryData object to process
            
        Returns:
            Processed TrajectoryData with corrected artifacts
        """
        if trajectory.static:
            return trajectory  # No processing needed for static objects
        
        # Get all frame times
        frame_times = trajectory.get_x()
        if len(frame_times) < 3:
            return trajectory  # Not enough frames to process
        
        # Create corrected versions of position and rotation interpolators
        corrected_positions = self._correct_position_trajectory(trajectory, frame_times)
        corrected_rotations = self._correct_rotation_trajectory(trajectory, frame_times)
        
        # Correct vertex trajectories
        corrected_vertices = self._correct_vertex_trajectories(trajectory, frame_times)
        corrected_normals = self._correct_normal_trajectories(trajectory, frame_times, corrected_vertices)
        
        # Create new trajectory with corrected data
        from scipy.interpolate import CubicSpline
        from scipy.spatial.transform import RotationSpline, Rotation
        
        # Create new interpolators
        new_positions = tuple([
            CubicSpline(frame_times, corrected_positions[:, i], extrapolate=1)
            for i in range(3)
        ])
        
        # Convert corrected rotations to Rotation objects
        rotations_list = []
        for i in range(len(frame_times)):
            rotations_list.append(Rotation.from_euler('XYZ', corrected_rotations[i]))
        new_rotations = RotationSpline(frame_times, Rotation.concatenate(rotations_list))
        
        # Create vertex interpolators
        n_vertices = corrected_vertices.shape[1]
        new_vertices = np.zeros((n_vertices, 3), dtype=object)
        new_normals = np.zeros((n_vertices, 3), dtype=object)
        
        for v_idx in range(n_vertices):
            for coord_idx in range(3):
                new_vertices[v_idx, coord_idx] = CubicSpline(
                    frame_times, corrected_vertices[:, v_idx, coord_idx], extrapolate=1
                )
                new_normals[v_idx, coord_idx] = CubicSpline(
                    frame_times, corrected_normals[:, v_idx, coord_idx], extrapolate=1
                )
        
        # Create new trajectory
        corrected_trajectory = TrajectoryData(
            obj_idx=trajectory.obj_idx,
            static=False,
            sfps=trajectory.sfps,
            sample_rate=trajectory.sample_rate,
            positions=new_positions,
            rotations=new_rotations,
            vertices=new_vertices,
            normals=new_normals,
            faces=trajectory.faces
        )
        
        return corrected_trajectory
    
    def _correct_position_trajectory(self, trajectory: TrajectoryData, 
                                      frame_times: np.ndarray) -> np.ndarray:
        """
        Correct bounce artifacts in the position trajectory.
        
        Args:
            trajectory: Original trajectory
            frame_times: Array of frame times
            
        Returns:
            Corrected position array [n_frames, 3]
        """
        # Get original positions
        positions = np.array([trajectory.get_position(t) for t in frame_times])
        
        # Detect bounce artifacts
        bounce_mask = self._detect_bounce_artifacts(positions, axis=1)  # Y-axis (vertical)
        
        # Detect contact regions
        contact_regions = self._detect_contact_regions(positions, axis=1)
        
        # Create corrected positions
        corrected = positions.copy()
        
        # Apply correction to bounce regions
        for region_start, region_end in contact_regions:
            if region_end - region_start >= self.min_contact_duration:
                # This is a contact region - smooth out bounces
                region_positions = positions[region_start:region_end]
                region_bounce = bounce_mask[region_start:region_end]
                
                if np.any(region_bounce):
                    # Find the contact surface level (minimum position in region)
                    contact_level = np.min(region_positions[:, 1])
                    
                    # Smooth the vertical position during contact
                    smoothed_y = self._smooth_contact_trajectory(
                        region_positions[:, 1], 
                        contact_level,
                        region_bounce
                    )
                    
                    corrected[region_start:region_end, 1] = smoothed_y
        
        # Apply rigid body constraints
        corrected = self._apply_rigid_body_constraints(corrected, frame_times)
        
        return corrected
    
    def _correct_rotation_trajectory(self, trajectory: TrajectoryData,
                                      frame_times: np.ndarray) -> np.ndarray:
        """
        Correct bounce artifacts in rotation trajectory.
        
        Args:
            trajectory: Original trajectory
            frame_times: Array of frame times
            
        Returns:
            Corrected rotation array [n_frames, 3] (Euler angles)
        """
        # Get original rotations
        rotations = np.array([trajectory.get_rotation(t) for t in frame_times])
        
        # Detect angular velocity anomalies
        angular_velocities = np.diff(rotations, axis=0) * self.sfps
        
        # Find frames with excessive angular velocity
        angular_speed = np.linalg.norm(angular_velocities, axis=1)
        anomaly_mask = angular_speed > self.max_angular_velocity
        
        # Correct anomalous rotations
        corrected = rotations.copy()
        for i in range(1, len(rotations) - 1):
            if anomaly_mask[i-1] or anomaly_mask[i]:
                # Interpolate between previous and next valid rotation
                prev_valid = i - 1
                next_valid = i + 1
                
                # Find next valid frame
                while next_valid < len(rotations) and anomaly_mask[next_valid-1]:
                    next_valid += 1
                
                if next_valid < len(rotations):
                    # Linear interpolation
                    alpha = 0.5
                    corrected[i] = (1 - alpha) * corrected[prev_valid] + alpha * corrected[next_valid]
        
        return corrected
    
    def _correct_vertex_trajectories(self, trajectory: TrajectoryData,
                                      frame_times: np.ndarray) -> np.ndarray:
        """
        Correct bounce artifacts in vertex trajectories.
        
        Args:
            trajectory: Original trajectory
            frame_times: Array of frame times
            
        Returns:
            Corrected vertex array [n_frames, n_vertices, 3]
        """
        # Get original vertices for all frames
        n_vertices = len(trajectory.vertices)
        vertices = np.zeros((len(frame_times), n_vertices, 3))
        
        for i, t in enumerate(frame_times):
            vertices[i] = trajectory.get_vertices(t)
        
        # Detect contact for each vertex
        corrected = vertices.copy()
        
        for v_idx in range(n_vertices):
            vertex_trajectory = vertices[:, v_idx, :]
            
            # Detect bounce artifacts in this vertex
            bounce_mask = self._detect_bounce_artifacts(vertex_trajectory)
            
            # Detect contact regions
            contact_regions = self._detect_contact_regions(vertex_trajectory)
            
            # Apply correction
            for region_start, region_end in contact_regions:
                if region_end - region_start >= self.min_contact_duration:
                    region_vertices = vertex_trajectory[region_start:region_end]
                    region_bounce = bounce_mask[region_start:region_end]
                    
                    if np.any(region_bounce):
                        # Find contact surface (minimum in each axis)
                        contact_level = np.min(region_vertices, axis=0)
                        
                        # Smooth each axis
                        for axis in range(3):
                            smoothed = self._smooth_contact_trajectory(
                                region_vertices[:, axis],
                                contact_level[axis],
                                region_bounce
                            )
                            corrected[region_start:region_end, v_idx, axis] = smoothed
        
        # Apply rigid body constraints to maintain mesh consistency
        corrected = self._enforce_rigid_body_constraints(vertices, corrected, frame_times)
        
        return corrected
    
    def _correct_normal_trajectories(self, trajectory: TrajectoryData,
                                      frame_times: np.ndarray,
                                      corrected_vertices: np.ndarray) -> np.ndarray:
        """
        Recompute normals from corrected vertices.
        
        Args:
            trajectory: Original trajectory
            frame_times: Array of frame times
            corrected_vertices: Corrected vertex positions
            
        Returns:
            Corrected normal array [n_frames, n_vertices, 3]
        """
        n_vertices = corrected_vertices.shape[1]
        n_frames = len(frame_times)
        
        # Get faces
        faces = trajectory.get_faces()
        
        # Compute vertex normals from corrected mesh
        corrected_normals = np.zeros((n_frames, n_vertices, 3))
        
        for i in range(n_frames):
            # Create temporary mesh for normal computation
            import trimesh
            mesh = trimesh.Trimesh(
                vertices=corrected_vertices[i],
                faces=faces,
                process=False
            )
            
            # Get vertex normals
            corrected_normals[i] = mesh.vertex_normals
        
        return corrected_normals
    
    def _detect_bounce_artifacts(self, trajectory: np.ndarray, 
                                  axis: Optional[int] = None) -> np.ndarray:
        """
        Detect bounce artifacts in a trajectory.
        
        Args:
            trajectory: Trajectory array [n_frames, 3] or [n_frames]
            axis: If specified, only check this axis
            
        Returns:
            Boolean mask of frames with bounce artifacts
        """
        if axis is not None:
            data = trajectory[:, axis]
        else:
            # Use magnitude of displacement
            data = np.linalg.norm(trajectory, axis=1)
        
        # Detect oscillations (bounces)
        # A bounce is characterized by rapid direction changes
        velocity = np.diff(data)
        acceleration = np.diff(velocity)
        
        # Detect zero-crossings in velocity (direction changes)
        velocity_sign = np.sign(velocity)
        direction_changes = np.diff(velocity_sign, prepend=0) != 0
        
        # A bounce has multiple direction changes in quick succession
        bounce_mask = np.zeros(len(data), dtype=bool)
        
        # Sliding window detection
        window_size = min(10, len(data) // 4)
        if window_size < 2:
            return bounce_mask
        
        for i in range(window_size, len(data) - window_size):
            window_changes = direction_changes[i-window_size:i+window_size]
            change_count = np.sum(window_changes)
            
            # Check if there are frequent direction changes
            if change_count > window_size * 0.3:  # More than 30% are changes
                # Check amplitude of bounces
                window_data = data[i-window_size:i+window_size]
                amplitude = np.max(window_data) - np.min(window_data)
                
                if amplitude > self.bounce_threshold:
                    bounce_mask[i] = True
        
        return bounce_mask
    
    def _detect_contact_regions(self, trajectory: np.ndarray,
                                 axis: Optional[int] = 1) -> List[Tuple[int, int]]:
        """
        Detect regions where the object is in contact with a surface.
        
        Args:
            trajectory: Trajectory array [n_frames, 3]
            axis: Axis perpendicular to contact surface (default: Y)
            
        Returns:
            List of (start, end) frame indices for contact regions
        """
        if axis is not None:
            data = trajectory[:, axis]
        else:
            data = np.linalg.norm(trajectory, axis=1)
        
        # Detect contact by checking for minimal movement
        velocity = np.abs(np.diff(data))
        
        # Contact regions have low velocity (sticking to surface)
        contact_threshold = np.percentile(velocity, 20)  # Bottom 20%
        is_contact = velocity < contact_threshold
        
        # Find continuous contact regions
        regions = []
        in_contact = False
        start_idx = 0
        
        for i, contact in enumerate(is_contact):
            if contact and not in_contact:
                in_contact = True
                start_idx = i
            elif not contact and in_contact:
                in_contact = False
                if i - start_idx >= self.min_contact_duration:
                    regions.append((start_idx, i))
        
        # Handle contact at end
        if in_contact and len(data) - start_idx >= self.min_contact_duration:
            regions.append((start_idx, len(data)))
        
        return regions
    
    def _smooth_contact_trajectory(self, data: np.ndarray,
                                    contact_level: float,
                                    bounce_mask: np.ndarray) -> np.ndarray:
        """
        Smooth a trajectory during contact, preserving the contact surface.
        
        Args:
            data: 1D trajectory data
            contact_level: The surface level (minimum position)
            bounce_mask: Boolean mask of bounce frames
            
        Returns:
            Smoothed trajectory
        """
        smoothed = data.copy()
        
        # Apply Gaussian smoothing to bounce regions
        if np.any(bounce_mask):
            # Create weights: lower weight for bounce frames
            weights = np.ones_like(data)
            weights[bounce_mask]] = 0.1
            
            # Apply weighted smoothing
            smoothed = self._weighted_smoothing(data, weights, sigma=self.smoothing_sigma)
        
        # Ensure we don't go below contact level (no penetration)
        smoothed = np.maximum(smoothed, contact_level)
        
        # Blend original and smoothed based on correction strength
        blended = (1 - self.correction_strength) * data + self.correction_strength * smoothed
        
        return blended
    
    def _weighted_smoothing(self, data: np.ndarray, weights: np.ndarray,
                             sigma: float) -> np.ndarray:
        """
        Apply weighted Gaussian smoothing.
        
        Args:
            data: Input data
            weights: Weight for each sample (0-1)
            sigma: Gaussian sigma
            
        Returns:
            Smoothed data
        """
        # Create kernel
        kernel_size = int(sigma * 5) | 1  # Ensure odd
        kernel = np.exp(-0.5 * (np.arange(kernel_size) - kernel_size//2)**2 / sigma**2)
        kernel = kernel / kernel.sum()
        
        # Apply weighted convolution
        weighted_data = data * weights
        smoothed = np.convolve(weighted_data, kernel, mode='same')
        
        # Normalize by weight convolution
        weight_sum = np.convolveve(weights, kernel, mode='same')
        weight_sum[weight_sum < 1e-10] = 1.0
        smoothed = smoothed / weight_sum
        
        return smoothed
    
    def _apply_rigid_body_constraints(self, positions: np.ndarray,
                                       frame_times: np.ndarray) -> np.ndarray:
        """
        Apply rigid body constraints to position trajectory.
        
        Args:
            positions: Position array [n_frames, 3]
            frame_times: Frame times
            
        Returns:
            Constrained positions
        """
        corrected = positions.copy()
        
        # Compute velocities
        dt = 1.0 / self.sfps
        velocities = np.diff(positions, axis=0) / dt
        
        # Detect and correct excessive velocity changes
        velocity_changes = np.diff(velocities, axis=0)
        velocity_change_magnitude = np.linalg.norm(velocity_changes, axis=1)
        
        for i in range(1, len(corrected) - 1):
            if velocity_change_magnitude[i-1] > self.max_velocity_change:
                # Limit the velocity change
                ratio = self.max_velocity_change / velocity_change_magnitude[i-1]
                limited_change = velocity_changes[i-1] * ratio * dt
                
                # Apply limited change
                corrected[i+1] = corrected[i] + velocities[i-1] * dt + limited_change
        
        return corrected
    
    def _enforce_rigid_body_constraints(self, original_vertices: np.ndarray,
                                         corrected_vertices: np.ndarray,
                                         frame_times: np.ndarray) -> np.ndarray:
        """
        Enforce rigid body constraints on vertex trajectories.
        
        This ensures that corrected vertices still represent a rigid body motion
        by projecting them onto the closest valid rigid body configuration.
        
        Args:
            original_vertices: Original vertex positions [n_frames, n_vertices, 3]
            corrected_vertices: Corrected vertex positions [n_frames, n_vertices, 3]
            frame_times: Frame times
            
        Returns:
            Constrained vertex positions
        """
        # For each frame, find the closest rigid body transformation
        # that maps the original vertices to the corrected vertices
        
        n_frames = len(frame_times)
        n_vertices = corrected_vertices.shape[1]
        constrained = corrected_vertices.copy()
        
        # Get reference vertices (use first frame)
        reference = original_vertices[0]
        reference_center = np.mean(reference, axis=0)
        reference_centered = reference - reference_center
        
        for i in range(n_frames):
            current = corrected_vertices[i]
            current_center = np.mean(current, axis=0)
            current_centered = current - current_center
            
            # Find optimal rotation using Kabsch algorithm
            H = reference_centered.T @ current_centered
            U, S, Vt = np.linalg.svd(H)
            R = Vt.T @ U.T
            
            # Ensure proper rotation (det(R) = 1)
            if np.linalg.det(R) < 0:
                Vt[-1, :] *= -1
                R = Vt.T @ U.T
            
            # Apply rigid body transformation
            transformed = reference_centered @ R.T + current_center
            
            # Blend with corrected vertices
            constrained[i] = (1 - self.correction_strength) * transformed + \
                             self.correction_strength * current
        
        return constrained
    
    def process_all_objects(self) -> Dict[int, TrajectoryData]:
        """
        Process all trajectories in the entity manager.
        
        Returns:
            Dictionary mapping object indices to corrected trajectories
        """
        trajectories = self.entity_manager.get('trajectories')
        corrected_trajectories = {}
        
        for idx, trajectory in trajectories.items():
            if isinstance(trajectory, TrajectoryData):
                corrected = self.process_trajectory(trajectory)
                corrected_trajectories[idx] = corrected
        
        return corrected_trajectories

