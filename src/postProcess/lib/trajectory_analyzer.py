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

# pbrAudioPostProcess/src/postProcess/lib/trajectory_analyzer.py

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

from physicsSolver.lib.trajectory_data import TrajectoryData


@dataclass
class TrajectoryAnalyzer:
    """
    Analyzes trajectory data for artifacts and provides statistics.
    Useful for debugging and understanding trajectory quality.
    """
    
    def analyze_trajectory(self, trajectory: TrajectoryData) -> Dict[str, Any]:
        """
        Analyze a trajectory for various artifacts.
        
        Args:
            trajectory: TrajectoryData to analyze
            
        Returns:
            Dictionary with analysis results
        """
        if trajectory.static:
            return {"static": True}
        
        frame_times = trajectory.get_x()
        n_frames = len(frame_times)
        
        if n_frames < 3:
            return {"too_few_frames": True}
        
        # Get positions
        positions = np.array([trajectory.get_position(t) for t in frame_times])
        
        # Get velocities
        velocities = np.array([trajectory.get_velocity(t) for t in frame_times])
        
        # Get accelerations
        accelerations = np.array([trajectory.get_acceleration(t) for t in frame_times])
        
        # Analyze artifacts
        results = {
            "n_frames": n_frames,
            "duration": (frame_times[-1] - frame_times[0]) / trajectory.sample_rate,
            "total_displacement": np.linalg.norm(positions[-1] - positions[0]),
            "max_velocity": np.max(np.linalg.norm(velocities, axis=1)),
            "max_acceleration": np.max(np.linalg.norm(accelerations, axis=1)),
            "bounce_count": self._count_bounces(positions),
            "jitter_score": self._compute_jitter_score(positions),
            "contact_regions": self._find_contact_regions(positions),
        }
        
        # Analyze vertex trajectories
        if hasattr(trajectory, 'vertices') and trajectory.vertices is not None:
            n_vertices = len(trajectory.vertices)
            vertex_bounces = 0
            
            for v_idx in range(min(10, n_vertices)):  # Sample first 10 vertices
                vertex_traj = np.array([
                    trajectory.get_vertices(t)[v_idx] for t in frame_times
                ])
                vertex_bounces += self._count_bounces(vertex_traj)
            
            results["vertex_bounce_count"] = vertex_bounces
            results["n_vertices"] = n_vertices
        
        return results
    
    def _count_bounces(self, trajectory: np.ndarray) -> int:
        """Count the number of bounce events in a trajectory."""
        if len(trajectory) < 3:
            return 0
        
        # Compute velocity
        velocity = np.diff(trajectory, axis=0)
        speed = np.linalg.norm(velocity, axis=1)
        
        # Detect direction changes
        velocity_sign = np.sign(velocity)
        direction_changes = np.sum(np.abs(np.diff(velocity_sign, axis=0)), axis=1) > 0
        
        # Count bounce events (clusters of direction changes)
        bounce_count = 0
        in_bounce = False
        
        for change in direction_changes:
            if change and not in_bounce:
                bounce_count += 1
                in_bounce = True
            elif not change:
                in_bounce = False
        
        return bounce_count
    
    def _compute_jitter_score(self, trajectory: np.ndarray) -> float:
        """
        Compute a jitter score (0-1) indicating high-frequency noise.
        Higher scores indicate more jitter.
        """
        if len(trajectory) < 5:
            return 0.0
        
        # Compute high-frequency components
        velocity = np.diff(trajectory, axis=0)
        acceleration = np.diff(velocity, axis=0)
        
        # Jitter is characterized by high acceleration variance
        acc_magnitude = np.linalg.norm(acceleration, axis=1)
        
        if len(acc_magnitude) == 0:
            return 0.0
        
        # Normalize by mean velocity
        mean_velocity = np.mean(np.linalg.norm(velocity, axis=1))
        if mean_velocity > 0:
            jitter = np.std(acc_magnitude) / mean_velocity
        else:
            jitter = np.std(acc_magnitude)
        
        # Normalize to 0-1 range
        jitter_score = min(1.0, jitter / 100.0)
        
        return jitter_score
    
    def _find_contact_regions(self, trajectory: np.ndarray) -> List[Tuple[int, int]]:
        """Find regions where the object appears to be in contact."""
        if len(trajectory) < 3:
            return []
        
        # Compute velocity magnitude
        velocity = np.diff(t(trajectory, axis=0)
        speed = np.linalg.norm(velocity, axis=1)
        
        # Contact is characterized by low speed
        speed_threshold = np.percentile(speed, 20)
        
        regions = []
        in_contact = False
        start = 0
        
        for i, s in enumerate(speed):
            if s < speed_threshold and not in_contact:
                in_contact = True
                start = i
            elif s >= speed_threshold and in_contact:
                in_contact = False
                if i - start >= 3:  # Minimum contact duration
                    regions.append((start, i))
        
        if in_contact and len(speed) - start >= 3:
            regions.append((start, len(speed)))
        
        return regions

