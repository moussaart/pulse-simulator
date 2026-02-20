"""
Geometry Data Adapter
Extracts tag-anchor geometry data.
"""
from typing import List, Tuple, TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from src.core.uwb.uwb_devices import Anchor, Tag


class GeometryDataAdapter:
    """
    Adapter for extracting geometry data (positions, distances).
    """
    
    def get_tag_position(self, tag: 'Tag') -> Tuple[float, float, float]:
        """Get tag position as (x, y, z) tuple"""
        return (
            tag.position.x,
            tag.position.y,
            getattr(tag.position, 'z', 0.0)
        )
    
    def get_anchor_positions(self, anchors: List['Anchor']) -> List[Tuple[float, float, float]]:
        """Get all anchor positions"""
        return [
            (a.position.x, a.position.y, getattr(a.position, 'z', 0.0))
            for a in anchors
        ]
    
    def get_anchor_ids(self, anchors: List['Anchor']) -> List[str]:
        """Get all anchor IDs"""
        return [a.id for a in anchors]
    
    def calculate_true_distances(self, 
                                  tag: 'Tag', 
                                  anchors: List['Anchor']) -> List[float]:
        """Calculate true distances from tag to all anchors"""
        return [a.position.distance_to(tag.position) for a in anchors]
    
    def calculate_measurement_errors(self,
                                      measurements: List[float],
                                      true_distances: List[float]) -> List[float]:
        """Calculate measurement errors (measured - true)"""
        errors = []
        for i, true_dist in enumerate(true_distances):
            if i < len(measurements):
                errors.append(measurements[i] - true_dist)
            else:
                errors.append(0.0)
        return errors
    
    def get_geometry_summary(self,
                             tag: 'Tag',
                             anchors: List['Anchor'],
                             measurements: List[float] = None) -> dict:
        """
        Get a complete geometry summary.
        
        Returns:
            Dictionary with all geometry data
        """
        tag_pos = self.get_tag_position(tag)
        anchor_positions = self.get_anchor_positions(anchors)
        anchor_ids = self.get_anchor_ids(anchors)
        true_distances = self.calculate_true_distances(tag, anchors)
        
        summary = {
            'tag_position': tag_pos,
            'anchor_positions': anchor_positions,
            'anchor_ids': anchor_ids,
            'true_distances': true_distances,
            'num_anchors': len(anchors)
        }
        
        if measurements:
            summary['measurements'] = measurements
            summary['measurement_errors'] = self.calculate_measurement_errors(
                measurements, true_distances)
            summary['rmse'] = np.sqrt(np.mean(np.array(summary['measurement_errors'])**2))
        
        return summary
    
    def calculate_gdop(self, 
                       tag: 'Tag', 
                       anchors: List['Anchor']) -> float:
        """
        Calculate Geometric Dilution of Precision (GDOP).
        
        GDOP indicates geometry quality for positioning.
        Lower values indicate better geometry.
        """
        if len(anchors) < 3:
            return float('inf')
            
        tag_pos = np.array([tag.position.x, tag.position.y])
        
        # Build geometry matrix H (unit vectors from tag to anchors)
        H = []
        for anchor in anchors:
            anchor_pos = np.array([anchor.position.x, anchor.position.y])
            diff = anchor_pos - tag_pos
            dist = np.linalg.norm(diff)
            
            if dist > 0:
                unit_vec = diff / dist
                H.append(unit_vec)
        
        if len(H) < 3:
            return float('inf')
            
        H = np.array(H)
        
        try:
            # GDOP = sqrt(trace((H^T H)^-1))
            HTH = H.T @ H
            HTH_inv = np.linalg.inv(HTH)
            gdop = np.sqrt(np.trace(HTH_inv))
            return float(gdop)
        except np.linalg.LinAlgError:
            return float('inf')
