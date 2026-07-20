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

# pbrAudioPostProcess/src/postProcess/core/trajectory_post_process_engine.py

import os
import numpy as np
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from dask import delayed, compute

from physicsSolver import EntityManager
from physicsSolver.lib.trajectory_data import TrajectoryData
from physicsSolver.lib.functions import _update_status
from ..lib.trajectory_post_process import TrajectoryPostProcess


@dataclass
class TrajectoryPostProcessEngine:
    """
    Engine that orchestrates post-processing of all trajectory data.
    Integrates with the existing physics engine pipeline.
    """
    
    entity_manager: EntityManager
    
    def __post_init__(self):
        config = self.entity_manager.get('config')
        self.status_dir = f"{config.system.cache_path}/status/TrajectoryPostProcessEngine"
        os.makedirs(self.status_dir, exist_ok=True)
        
        self.trajectory_post_processor = TrajectoryPostProcess(
            entity_manager=self.entity_manager
        )
    
    def process(self) -> Dict[int, TrajectoryData]:
        """
        Process all trajectories in the scene.
        
        Returns:
            Dictionary mapping object indices to corrected trajectories
        """
        _update_status(f"{self.status_dir}/process", 0)
        
        config = self.entity_manager.get('config')
        tasks = []
        
        # Create delayed tasks for each dynamic object
        trajectories = self.entity_manager.get('trajectories')
        for idx, trajectory in trajectories.items():
            if isinstance(trajectory, TrajectoryData) and not trajectory.static:
                task = self._process_single_trajectory(idx, trajectory)
                tasks.append(task)
        
        # Compute all tasks in parallel
        results_list = compute(*tasks)
        
        # Combine results
        all_results = {}
        for result in results_list:
            if result:
                all_results.update(result)
        
        # Update entity manager with corrected trajectories
        for obj_idx, corrected_trajectory in all_results.items():
            # Replace the trajectory in the entity manager
            trajectories = self.entity_manager.get('trajectories')
            for idx, traj in trajectories.items():
                if isinstance(traj, TrajectoryData) and traj.obj_idx == obj_idx:
                    # Update the trajectory
                    self.entity_manager._trajectories[idx] = corrected_trajectory
                    break
        
        _update_status(f"{self.status_dir}/process", 100)
        
        return all_results
    
    @delayed
    def _process_single_trajectory(self, idx: int, trajectory: TrajectoryData) -> Dict[int, TrajectoryData]:
        """Process a single trajectory."""
        corrected = self.trajectory_post_processor.process_trajectory(trajectory)
        return {idx: corrected} if corrected else {}
    
    def process_before_distance_solver(self) -> None:
        """
        Process trajectories before the distance solver runs.
        This ensures bounce-corrected trajectories are used for collision detection.
        
        Should be called in the bake pipeline after FlightPath but before DistanceSolver.
        """
        _update_status(f"{self.status_dir}/pre_distance", 0)
        
        results = self.process()
        
        # Save corrected trajectories
        config = self.entity_manager.get('config')
        output_dir = f"{config.system.cache_path}/trajectories_corrected"
        os.makedirs(output_dir, exist_ok=True)
        
        for obj_idx, trajectory in results.items():
            filename = f"{output_dir}/obj_{obj_idx:05d}.pkl"
            trajectory.save(filename(filename)
        
        _update_status(f"{self.status_dir}/pre_distance", 100)

