"""
Simulation Manager Module
Handles simulation state and update logic with parallel computing optimization
"""
import numpy as np
import time
from concurrent.futures import ThreadPoolExecutor
from src.core.motion import MotionController
from src.core.localization import LocalizationAlgorthimes, Alghortimes_doc
from src.core.localization.base_algorithm import BaseLocalizationAlgorithm, AlgorithmInput, AlgorithmOutput
from src.gui.managers.simulation_recorder import SimulationRecorder
from src.core.parallel.gpu_backend import gpu_manager
from src.core.exceptions import (
    SimulationError, NumericalError, ConvergenceError,
    MeasurementError, InputValidationError,
)
from src.core.error_handler import SimulationErrorHandler
import inspect
import logging

logger = logging.getLogger(__name__)


class SimulationManager:
    """Manages simulation state and updates with parallel computing optimization"""
    
    # Class-level thread pool for parallel measurements
    _executor = None
    _max_workers = 4
    
    @classmethod
    def get_executor(cls):
        """Get or create the thread pool executor"""
        if cls._executor is None:
            cls._executor = ThreadPoolExecutor(max_workers=cls._max_workers)
        return cls._executor
    
    @classmethod
    def shutdown_executor(cls):
        """Shutdown the thread pool executor"""
        if cls._executor is not None:
            cls._executor.shutdown(wait=False)
            cls._executor = None
    
    def __init__(self, parent):
        self.parent = parent
        self.simulation_time = 0
        self.start_time = time.time()
        self.last_update = self.start_time
        self.is_paused = True  # Start paused by default
        self.pause_time = 0
        self.use_parallel = True  # Enable/disable parallel processing
        
        # Duration and recording settings
        self.max_duration = 30.0  # Default 30 seconds, None = infinite
        self.simulation_ended = False
        
        # Initialize recorder for timeline playback
        self.recorder = SimulationRecorder(max_duration=self.max_duration, snapshot_interval=5)
        
        # Cache for instantiated algorithm classes
        self.algorithm_instances = {}
        
        # Cache algorithm dispatch table (avoids re-instantiation every frame)
        self._algorithm_methods = Alghortimes_doc().get_algorithm_methods()
        
        # Error handler (set by parent after construction)
        self.error_handler = None
        
        # Consecutive measurement failure counter
        self._consecutive_measurement_failures = 0
        self._MAX_CONSECUTIVE_FAILURES = 10
        
    def update_simulation(self):
        """Main simulation update loop"""
        try:
            if not self.is_paused and not self.simulation_ended:
                self.simulation_time += self.parent.dt
                
                # Check if duration limit reached
                if self.max_duration is not None and self.simulation_time >= self.max_duration:
                    self._on_simulation_ended()
                    return
                
                # Update tag position
                try:
                    MotionController.update_tag_position(
                        tag=self.parent.tag,
                        movement_pattern=self.parent.movement_pattern,
                        movement_speed=self.parent.movement_speed,
                        t=self.simulation_time,
                        frequence=1 / self.parent.dt,
                        point=self.parent.point
                    )
                except SimulationError:
                    raise
                except Exception as e:
                    raise SimulationError.from_exception(e, "tag position update")
                
                # Validate tag position is finite
                if not (np.isfinite(self.parent.tag.position.x) and np.isfinite(self.parent.tag.position.y)):
                    raise NumericalError(
                        user_message="The tag position became invalid (NaN or Infinity) during motion update.",
                        details={"x": self.parent.tag.position.x, "y": self.parent.tag.position.y},
                    )
                
                # Update anchor visualization
                self.parent.plot_manager.update_anchor_visualization(
                    self.parent.position_plot,
                    self.parent.anchors,
                    self.parent.channel_conditions,
                    self.parent.tag
                )
                
                # Update moving NLOS zones visualization
                if hasattr(self.parent, 'nlos_manager'):
                    self.parent.nlos_manager.update_moving_visualizations()
                
                # Get measurements
                measurements, is_los, valid = self.get_measurements()
                
                if not valid:
                    self._consecutive_measurement_failures += 1
                    if self._consecutive_measurement_failures >= self._MAX_CONSECUTIVE_FAILURES:
                        raise MeasurementError(
                            user_message=(
                                f"Measurements have failed {self._consecutive_measurement_failures} "
                                f"times in a row. The distance measurement system may be misconfigured "
                                f"or the tag may be outside the coverage area of all anchors."
                            ),
                            details={"consecutive_failures": self._consecutive_measurement_failures},
                        )
                    return
                self._consecutive_measurement_failures = 0
                
                # Estimate position
                estimated_pos = self.estimate_position(measurements, is_los)
                
                # Validate estimated position
                if not (np.isfinite(estimated_pos[0]) and np.isfinite(estimated_pos[1])):
                    raise NumericalError(
                        user_message=(
                            f"The '{self.parent.algorithm}' algorithm produced an invalid position estimate "
                            f"(NaN or Infinity). This typically means the algorithm has become "
                            f"numerically unstable with the current parameters."
                        ),
                        details={"algorithm": self.parent.algorithm, "position": estimated_pos},
                    )
                
                # Check for divergence (position way outside reasonable bounds)
                if self.error_handler:
                    SimulationErrorHandler.check_divergence(estimated_pos, bounds=1e4, label="position estimate")
                
                # Calculate error
                true_pos = (self.parent.tag.position.x, self.parent.tag.position.y)
                error = np.sqrt((estimated_pos[0] - true_pos[0])**2 + 
                               (estimated_pos[1] - true_pos[1])**2)
                
                # Validate error value
                if not np.isfinite(error):
                    error = 0.0  # Safe fallback for display
                
                # AI Training Data Collection Hook
                if hasattr(self.parent, 'training_api') and self.parent.training_api.is_collecting:
                    self.parent.training_api.collect_sample(
                        timestamp=self.simulation_time,
                        tag=self.parent.tag,
                        anchors=self.parent.anchors,
                        measurements=measurements,
                        channel_conditions=self.parent.channel_conditions,
                        filter_state={
                            'state': self.parent.kf_state,
                            'P': self.parent.kf_P,
                            'R': self.parent.aekf_R,
                            'Q': self.parent.aekf_Q
                        },
                        estimated_pos=estimated_pos,
                        error=error,
                        algorithm_name=self.parent.algorithm
                    )
                
                # Record snapshot for timeline playback
                self.recorder.record_snapshot(
                    timestamp=self.simulation_time,
                    tag_position=(self.parent.tag.position.x, self.parent.tag.position.y),
                    estimated_position=estimated_pos,
                    error=error,
                    anchors=self.parent.anchors,
                    channel_conditions=self.parent.channel_conditions,
                    measurements=measurements
                )
                
                # Update elapsed time display if available
                if hasattr(self.parent, 'elapsed_time_display') and self.parent.elapsed_time_display:
                    max_str = f"{self.max_duration:.2f}s" if self.max_duration else "∞"
                    self.parent.elapsed_time_display.setText(f"{self.simulation_time:.2f}s / {max_str}")
                
                # Update timeline widget in real-time
                if hasattr(self.parent, 'timeline_widget') and self.parent.timeline_widget:
                    max_time = self.max_duration if self.max_duration else self.simulation_time
                    self.parent.timeline_widget.set_time_range(0, max_time)
                    self.parent.timeline_widget.set_current_time(self.simulation_time, emit_signal=False)
                
                # Update visualizations
                self.update_visualizations(estimated_pos, error, measurements)
                
        except SimulationError as e:
            self._handle_simulation_error(e)
        except Exception as e:
            self._handle_simulation_error(
                SimulationError.from_exception(e, context="simulation loop")
            )
    
    def _handle_simulation_error(self, error):
        """Stop the simulation and notify the error handler."""
        self.is_paused = True
        if hasattr(self.parent, 'timer'):
            self.parent.timer.stop()
        
        # Update pause button to indicate error state
        if hasattr(self.parent, 'pause_button'):
            self.parent.pause_button.setChecked(True)
            self.parent.pause_button.setText("⚠️ Error")
            self.parent.pause_button.setToolTip("Simulation stopped due to error")
        
        if self.error_handler:
            self.error_handler.handle_error(error)
        else:
            # Fallback: print to console if no error handler is set
            logger.error(f"Simulation error (no handler): {error}")
    
    def _on_simulation_ended(self):
        """Handle simulation reaching its duration limit"""
        self.simulation_ended = True
        self.is_paused = True
        self.recorder.mark_simulation_ended()
        
        # Stop the timer
        if hasattr(self.parent, 'timer'):
            self.parent.timer.stop()
        
        # Update pause button
        if hasattr(self.parent, 'pause_button'):
            self.parent.pause_button.setChecked(True)
            self.parent.pause_button.setText("✅ Finished")
            self.parent.pause_button.setToolTip("Simulation complete - use timeline to review")
        
        # Show timeline widget
        if hasattr(self.parent, 'timeline_widget'):
            self.parent.timeline_widget.show_timeline(
                self.recorder.start_time,
                self.recorder.end_time
            )
            
        # Show history toggle and export button
        if hasattr(self.parent, 'history_toggle'):
            self.parent.history_toggle.setVisible(True)
        
        if hasattr(self.parent, 'export_btn'):
            self.parent.export_btn.setVisible(True)
        
        print(f"Simulation ended at {self.simulation_time:.2f}s - {self.recorder.snapshot_count} snapshots recorded")
    
    def set_duration(self, duration):
        """
        Set the simulation duration.
        
        Args:
            duration: Duration in seconds, or None for infinite
        """
        self.max_duration = duration
        self.simulation_ended = False
        self.recorder.set_max_duration(duration)
    
    def set_snapshot_interval(self, interval):
        """Set how often snapshots are recorded (1 = every frame)"""
        self.recorder.set_snapshot_interval(interval)
    
    def reset(self):
        """Reset simulation state for new run"""
        self.simulation_time = 0
        self.simulation_ended = False
        self.is_paused = True
        self.recorder.clear()
        
        # Reset error tracking
        self._consecutive_measurement_failures = 0
        
        # Hide history toggle and export button
        if hasattr(self.parent, 'history_toggle'):
             self.parent.history_toggle.setVisible(False)
             
        if hasattr(self.parent, 'export_btn'):
             self.parent.export_btn.setVisible(False)
        
        # Reset algorithm instances
        for name, instance in self.algorithm_instances.items():
            try:
                if hasattr(instance, 'initialize'):
                    instance.initialize()
            except Exception as e:
                logger.warning(f"Error resetting algorithm {name}: {e}")
    
    def _measure_single_anchor(self, anchor):
        """
        Measure distance to a single anchor (used for parallel execution).
        
        Returns:
            tuple: (anchor_id, distance, is_los, true_distance, ranging_result) or None on error
        """
        try:
            # Update channel conditions for this anchor
            self.parent.channel_conditions.update_los_condition(
                anchor.position, self.parent.tag.position)
            
            # Get measurement
            # Expecting 3 values now: distance, logs, ranging_result
            distance, messages, ranging_result = self.parent.tag.measure_distance_with_logs(
                anchor, self.parent.channel_conditions, self.simulation_time, "SS-TWR")
            
            if not np.isfinite(distance) or distance < 0:
                return None
            
            # Check LOS condition
            is_los_test = self.parent.channel_conditions.check_los_to_anchor(
                anchor.position, self.parent.tag.position)
            
            # Calculate true distance
            true_distance = anchor.position.distance_to(self.parent.tag.position)
            
            return (anchor.id, distance, is_los_test, true_distance, ranging_result)
        except Exception as e:
            print(f"Error measuring anchor {anchor.id}: {e}")
            return None
    
    def get_measurements(self):
        """
        Get distance measurements from all anchors.
        Uses parallel processing when multiple anchors are present.
        """
        measurements = []
        is_los = []
        valid_measurements = True
        
        anchors = self.parent.anchors
        n_anchors = len(anchors)
        
        if n_anchors == 0:
            return measurements, is_los, False
        
        # Use GPU batch processing when available, else parallel/sequential
        if gpu_manager.available and n_anchors >= 2:
            results = self._get_measurements_gpu_batch()
        elif self.use_parallel and n_anchors >= 3:
            results = self._get_measurements_parallel()
        else:
            results = self._get_measurements_sequential()
        
        # Process results
        for result in results:
            if result is None:
                valid_measurements = False
                break
            
            # Unpack 5 values
            anchor_id, distance, is_los_test, true_distance, ranging_result = result
            measurements.append(distance)
            is_los.append(0 if is_los_test else 1)
            
            # Update distance plots if window is open
            if self.parent.distance_plots_window is not None:
                self.parent.distance_plots_window.update_distances(
                    anchor_id, self.simulation_time, distance, true_distance, is_los_test)
                    
            # Update CIR Window if open and we have data
            if hasattr(self.parent, 'cir_window') and self.parent.cir_window is not None and self.parent.cir_window.isVisible():
                if ranging_result:
                    self.parent.cir_window.update_cir_data(anchor_id, ranging_result)
        
        if self.parent.add_imperfections and valid_measurements:
            is_los = LocalizationAlgorthimes.simuler_detection(
                is_los, self.parent.los_aware_probabilite_erreur)
        
        return measurements, is_los, valid_measurements
    
    def _get_measurements_sequential(self):
        """Get measurements sequentially (for small number of anchors)"""
        results = []
        for anchor in self.parent.anchors:
            result = self._measure_single_anchor(anchor)
            results.append(result)
        return results
    
    def _get_measurements_parallel(self):
        """Get measurements in parallel using thread pool"""
        executor = self.get_executor()
        futures = [executor.submit(self._measure_single_anchor, anchor) 
                   for anchor in self.parent.anchors]
        
        # Collect results in order
        results = []
        for future in futures:
            try:
                result = future.result(timeout=1.0)  # 1 second timeout
                results.append(result)
            except Exception as e:
                logger.warning(f"Parallel measurement error: {e}")
                results.append(None)
        
        return results
    
    def _get_measurements_gpu_batch(self):
        """
        Get measurements using GPU batch processing.
        All anchors are measured in one batched GPU call instead of per-anchor.
        """
        try:
            anchors = self.parent.anchors
            tag_pos = self.parent.tag.position
            channel = self.parent.channel_conditions
            
            # 1. Compute all LOS conditions in one pass
            is_los_array = np.array([
                channel.check_los_to_anchor(a.position, tag_pos)
                for a in anchors
            ])
            
            # 2. Update channel conditions with first anchor (they share model params)
            channel.update_los_condition(anchors[0].position, tag_pos)
            
            # 3. Compute true distances vectorized
            true_distances = np.array([
                a.position.distance_to(tag_pos) for a in anchors
            ])
            
            # 4. Batch measurement through GPU pipeline
            ranging_results = channel.measure_distance_batch(
                true_distances, is_los_array, anchor_positions=[a.position for a in anchors]
            )
            
            # 5. Pack into expected result format
            results = []
            for i, (anchor, rr) in enumerate(zip(anchors, ranging_results)):
                dist = rr.measured_distance
                if not np.isfinite(dist) or dist < 0:
                    results.append(None)
                    continue
                results.append((
                    anchor.id,
                    dist,
                    bool(is_los_array[i]),
                    float(true_distances[i]),
                    rr
                ))
            
            return results
        except Exception as e:
            logger.warning(f"GPU batch measurement failed, falling back: {e}")
            return self._get_measurements_sequential()
    
    def estimate_position(self, measurements, is_los):
        """Estimate position using selected algorithm"""
        if len(measurements) < 3:
            return (self.parent.tag.position.x, self.parent.tag.position.y)
        
        # Validate measurement values before passing to algorithms
        for i, m in enumerate(measurements):
            if not np.isfinite(m):
                raise NumericalError(
                    user_message=(
                        f"Measurement from anchor {i+1} is invalid (NaN or Infinity). "
                        f"The channel model may have produced an unrealistic value."
                    ),
                    details={"measurement_index": i, "value": m},
                )
            if m < 0:
                raise InputValidationError(
                    user_message=(
                        f"Measurement from anchor {i+1} is negative ({m:.4f}m). "
                        f"Distance measurements must be positive."
                    ),
                    details={"measurement_index": i, "value": m},
                )
        
        algorithm_methods = self._algorithm_methods
        method = algorithm_methods.get(self.parent.algorithm)
        
        u = np.array([self.parent.tag.imu_data.acc_x[-1], 
                     self.parent.tag.imu_data.acc_y[-1]])
        
        # Priority check for custom class-based algorithms
        if method and inspect.isclass(method) and issubclass(method, BaseLocalizationAlgorithm):
            try:
                algo_name = self.parent.algorithm
                if algo_name not in self.algorithm_instances:
                    self.algorithm_instances[algo_name] = method()
                    self.algorithm_instances[algo_name].initialize()
                
                algo_instance = self.algorithm_instances[algo_name]
                
                # Create input data object
                input_data = AlgorithmInput(
                    measurements=measurements,
                    anchors=self.parent.anchors,
                    tag=self.parent.tag,
                    dt=self.parent.dt,
                    state=self.parent.kf_state if hasattr(self.parent, 'kf_state') else None,
                    covariance=self.parent.kf_P if hasattr(self.parent, 'kf_P') else None,
                    initialized=self.parent.kf_initialized if hasattr(self.parent, 'kf_initialized') else False,
                    imu_data_on=False,
                    control_input=u,
                    is_los=is_los,
                    params={} # Add any extra params if needed
                )
                
                output = algo_instance.update(input_data)
                
                # Update parent state with output
                if output:
                    if hasattr(self.parent, 'kf_state'):
                        self.parent.kf_state = output.state
                    if hasattr(self.parent, 'kf_P'):
                        self.parent.kf_P = output.covariance
                    if hasattr(self.parent, 'kf_initialized'):
                        self.parent.kf_initialized = output.initialized
                        
                    return output.position
                    
            except SimulationError:
                raise  # re-raise our own errors
            except Exception as e:
                raise SimulationError.from_exception(e, f"custom algorithm '{self.parent.algorithm}'")

        if method:
            if "Improved Adaptive EKF" in self.parent.algorithm:
                result = method(
                    measurements=measurements,
                    tag=self.parent.tag,
                    anchors=self.parent.anchors,
                    aekf_state=self.parent.kf_state,
                    aekf_P=self.parent.kf_P,
                    aekf_initialized=self.parent.kf_initialized,
                    dt=self.parent.dt,
                    mu=self.parent.adaptive_iekf_mu,
                    alpha=self.parent.adaptive_iekf_alpha,
                    xi=self.parent.adaptive_iekf_xi,
                    lambda_min=self.parent.adaptive_iekf_lambda_min,
                    lambda_max=self.parent.adaptive_iekf_lambda_max,
                    tau=self.parent.adaptive_iekf_tau,
                    iteration_count=self.parent.adaptive_iekf_iteration_count,
                    prev_R=self.parent.adaptive_iekf_prev_R,
                    innovation_history=self.parent.adaptive_iekf_innovation_history,
                    imu_data_on=False,
                    u=u
                )
                (position, self.parent.adaptive_iekf_innovation_history, 
                 self.parent.kf_state, self.parent.kf_P, 
                 self.parent.kf_initialized, self.parent.aekf_Q, 
                 self.parent.adaptive_iekf_prev_R) = result
                self.parent.aekf_R = self.parent.adaptive_iekf_prev_R
                return position
                
            elif "NLOS-Aware" in self.parent.algorithm:
                if "IMU assisted NLOS-Aware AEKF" in self.parent.algorithm:
                    result = method(
                        measurements=measurements,
                        tag=self.parent.tag,
                        anchors=self.parent.anchors,
                        state=self.parent.imu_state,
                        P=self.parent.imu_P,
                        initialized=self.parent.kf_initialized,
                        is_los=is_los,
                        alpha=self.parent.los_aware_alpha,
                        beta=self.parent.los_aware_beta,
                        nlos_factor=self.parent.los_aware_nlos_factor,
                        dt=self.parent.dt,
                        zupt_threshold=0.05,
                        R=self.parent.aekf_R
                    )
                    (position, self.parent.imu_state, self.parent.imu_P, 
                     self.parent.kf_initialized, self.parent.aekf_R) = result
                    return position
                    
                result = method(
                    measurements=measurements,
                    tag=self.parent.tag,
                    anchors=self.parent.anchors,
                    aekf_state=self.parent.kf_state,
                    aekf_P=self.parent.kf_P,
                    aekf_initialized=self.parent.kf_initialized,
                    is_los=is_los,
                    alpha=self.parent.los_aware_alpha,
                    beta=self.parent.los_aware_beta,
                    nlos_factor=self.parent.los_aware_nlos_factor,
                    dt=self.parent.dt,
                    imu_data_on=False,
                    u=u,
                    R=self.parent.aekf_R,
                    Q=self.parent.aekf_Q
                )
                (position, self.parent.kf_state, self.parent.kf_P, 
                 self.parent.kf_initialized, self.parent.aekf_Q, 
                 self.parent.aekf_R) = result
                return position
                
            elif "Kalman" in self.parent.algorithm:
                if "Adaptive Extended Kalman Filter" in self.parent.algorithm:
                    result = method(
                        measurements=measurements,
                        tag=self.parent.tag,
                        anchors=self.parent.anchors,
                        aekf_state=self.parent.kf_state,
                        aekf_P=self.parent.kf_P,
                        aekf_initialized=self.parent.kf_initialized,
                        dt=self.parent.dt,
                        Q=self.parent.aekf_Q,
                        R=self.parent.aekf_R,
                        imu_data_on=False,
                        u=u
                    )
                    (position, self.parent.kf_state, self.parent.kf_P, 
                     self.parent.kf_initialized, self.parent.aekf_Q, 
                     self.parent.aekf_R) = result
                    return position
                    
                result = method(
                    measurements, self.parent.tag, self.parent.anchors,
                    self.parent.kf_state, self.parent.kf_P, 
                    self.parent.kf_initialized, self.parent.dt,
                    imu_data_on=False, u=u
                )
                (position, self.parent.kf_state, self.parent.kf_P, 
                 self.parent.kf_initialized) = result
                return position
                
            elif "IMU Only" in self.parent.algorithm:
                measurements_imu = [float(self.parent.tag.imu_data.acc_x[-1]), 
                                   float(self.parent.tag.imu_data.acc_y[-1])]
                result = method(
                    tag=self.parent.tag,
                    measurements=measurements_imu,
                    state=self.parent.imu_state,
                    P=self.parent.imu_P,
                    initialized=self.parent.kf_initialized,
                    dt=self.parent.dt
                )
                (output, self.parent.imu_state, self.parent.imu_P, 
                 self.parent.kf_initialized) = result
                return output
            else:
                return method(measurements, self.parent.anchors)
        
        # Fallback to trilateration
        return LocalizationAlgorthimes.trilateration(measurements, self.parent.anchors)
    
    def update_visualizations(self, estimated_pos, error, measurements):
        """Update all visualization elements"""
        # Update plot items
        self.parent.plot_items['tag_point'].setData(
            [self.parent.tag.position.x], [self.parent.tag.position.y])
        self.parent.plot_items['estimated_point'].setData(
            [estimated_pos[0]], [estimated_pos[1]])
        
        # Update error plot
        if np.isfinite(error) and not self.parent.error_plot_handler.error_plot_paused:
            ma_window = self.parent.ma_window_spin.value() if self.parent.ma_window_spin else 10
            self.parent.error_plot_handler.update_error(
                self.simulation_time, error, ma_window)
        
        # Draw measurement lines
        self.parent.plot_manager.update_measurement_lines(
            self.parent.position_plot, self.parent.anchors,
            self.parent.tag, self.parent.channel_conditions)
        
        # Update status
        if np.isfinite(error):
            self.update_status(error)
        
        # Update trajectory histories
        self.parent.plot_manager.update_trajectory_histories(
            (self.parent.tag.position.x, self.parent.tag.position.y),
            estimated_pos,
            self.parent.plot_items
        )
    
    def update_status(self, error):
        """Update status display"""
        if self.parent.status_display is None:
            return
        status_html = self.generate_status_html(error)
        self.parent.status_display.setHtml(status_html)
        
        # Auto-scroll if needed
        if not self.parent.user_scrolling:
            scrollbar = self.parent.status_display.verticalScrollBar()
            scrollbar.setValue(scrollbar.maximum())
    
    def generate_status_html(self, error):
        """Generate HTML for status display"""
        status_html = """
        <style>
            .section {
                background-color: #363636;
                border-radius: 8px;
                padding: 10px;
                margin-bottom: 10px;
            }
            .section-title {
                color: #00a6e3;
                font-size: 13px;
                font-weight: bold;
                margin-bottom: 8px;
            }
            .value-item {
                color: #e0e0e0;
                margin-bottom: 4px;
            }
            .label { color: #888888; }
            .value { color: #ffffff; font-weight: bold; }
            .error-value { color: #ff5555; font-weight: bold; }
            .good-value { color: #55ff55; font-weight: bold; }
        </style>
        """
        
        # Position and algorithm
        status_html += '<div class="section">'
        status_html += '<div class="section-title">📍 Position & Algorithm</div>'
        status_html += f'<div class="value-item"><span class="label">Tag Position:</span> <span class="value">({self.parent.tag.position.x:.2f}, {self.parent.tag.position.y:.2f})</span></div>'
        status_html += f'<div class="value-item"><span class="label">Algorithm:</span> <span class="value">{self.parent.algorithm}</span></div>'
        status_html += f'<div class="value-item"><span class="label">Current Error:</span> <span class="error-value">{error * 1000:.1f}mm</span></div>'
        
        # Average error
        if hasattr(self.parent.error_plot_handler, 'errors') and len(self.parent.error_plot_handler.errors) > 0:
            valid_errors = [e for e in self.parent.error_plot_handler.errors if np.isfinite(e)]
            if valid_errors:
                avg_error = np.mean(valid_errors)
                status_html += f'<div class="value-item"><span class="label">Average Error:</span> <span class="error-value">{avg_error * 1000:.1f}mm</span></div>'
        status_html += '</div>'
        
        # Channel parameters
        status_html += '<div class="section">'
        status_html += '<div class="section-title">📡 Channel Parameters</div>'
        status_html += f'<div class="value-item"><span class="label">Path Loss Exponent:</span> <span class="value">{self.parent.channel_conditions.los_path_loss_params.path_loss_exponent:.1f}</span></div>'
        status_html += f'<div class="value-item"><span class="label">Shadow STD:</span> <span class="value">{self.parent.channel_conditions.los_path_loss_params.shadow_fading_std:.1f} dB</span></div>'
        status_html += '</div>'
        
        # Anchor status
        nlos_count = sum(1 for anchor in self.parent.anchors 
                        if not self.parent.channel_conditions.check_los_to_anchor(
                            anchor.position, self.parent.tag.position))
        
        status_html += '<div class="section">'
        status_html += '<div class="section-title">📍 Anchor Status</div>'
        status_html += f'<div class="value-item"><span class="label">Total Anchors:</span> <span class="value">{len(self.parent.anchors)}</span></div>'
        status_html += f'<div class="value-item"><span class="label">NLOS Anchors:</span> <span class="error-value">{nlos_count}</span></div>'
        status_html += f'<div class="value-item"><span class="label">LOS Anchors:</span> <span class="good-value">{len(self.parent.anchors) - nlos_count}</span></div>'
        status_html += '</div>'
        
        return status_html

