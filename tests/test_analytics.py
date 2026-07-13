"""
Unit tests for analytics functions.

Tests metric correctness for:
- Shot categorization
- Pass completion
- Progressive actions
- Turnovers
"""

import pytest
import pandas as pd
import numpy as np
from analytics.enhanced_metrics import (
    categorize_shot_outcome,
    calculate_pass_completion,
    calculate_progressive_passes,
    calculate_turnovers,
    calculate_attacking_pressures
)


class TestShotCategorization:
    """Test shot outcome categorization."""
    
    def test_empty_shots(self):
        """Should handle empty shot DataFrame."""
        shots = pd.DataFrame()
        result = categorize_shot_outcome(shots)
        
        assert result['total'] == 0
        assert result['accuracy_pct'] == 0
        assert result['goal'] == 0
    
    def test_goals_only(self):
        """Should correctly count goals."""
        shots = pd.DataFrame({
            'shot_outcome': ['Goal', 'Goal', 'Goal']
        })
        result = categorize_shot_outcome(shots)
        
        assert result['goal'] == 3
        assert result['total'] == 3
        assert result['accuracy_pct'] == 1.0
    
    def test_mixed_outcomes(self):
        """Should correctly categorize mixed shot outcomes."""
        shots = pd.DataFrame({
            'shot_outcome': ['Goal', 'Saved', 'Off Target', 'Blocked', 'Post']
        })
        result = categorize_shot_outcome(shots)
        
        assert result['goal'] == 1
        assert result['saved'] == 1
        assert result['off_target'] == 1
        assert result['blocked'] == 1
        assert result['post'] == 1
        assert result['total'] == 5
        assert result['accuracy_pct'] == 2/5  # goal + saved


class TestPassCompletion:
    """Test pass completion calculation."""
    
    def test_empty_passes(self):
        """Should handle empty pass DataFrame."""
        passes = pd.DataFrame()
        result = calculate_pass_completion(passes)
        
        assert result['total'] == 0
        assert result['completion_pct'] == 0.0
    
    def test_all_completed_passes(self):
        """Should recognize completed passes (NaN in pass_outcome)."""
        passes = pd.DataFrame({
            'pass_outcome': [np.nan, np.nan, np.nan]
        })
        result = calculate_pass_completion(passes)
        
        assert result['completed'] == 3
        assert result['incomplete'] == 0
        assert result['completion_pct'] == 1.0
    
    def test_mixed_passes(self):
        """Should handle mix of completed and incomplete."""
        passes = pd.DataFrame({
            'pass_outcome': [np.nan, 'Incomplete', np.nan, 'Out']
        })
        result = calculate_pass_completion(passes)
        
        assert result['completed'] == 2
        assert result['incomplete'] == 2
        assert result['completion_pct'] == 0.5


class TestProgressivePasses:
    """Test progressive pass calculation."""
    
    def test_empty_passes(self):
        """Should handle empty pass DataFrame."""
        passes = pd.DataFrame()
        result = calculate_progressive_passes(passes)
        
        assert result == 0
    
    def test_no_location_column(self):
        """Should handle missing location column."""
        passes = pd.DataFrame({
            'type': ['Pass', 'Pass']
        })
        result = calculate_progressive_passes(passes)
        
        assert result == 0
    
    def test_progressive_pass(self):
        """Should count pass advancing 10+ yards."""
        passes = pd.DataFrame({
            'location': [[30, 40], [60, 40]],
            'pass_end_location': [[45, 40], [75, 40]]
        })
        result = calculate_progressive_passes(passes)
        
        # First: 45-30=15 yards (progressive)
        # Second: 75-60=15 yards (progressive)
        assert result == 2
    
    def test_backward_pass(self):
        """Should not count backward passes."""
        passes = pd.DataFrame({
            'location': [[50, 40]],
            'pass_end_location': [[30, 40]]
        })
        result = calculate_progressive_passes(passes)
        
        assert result == 0


class TestTurnovers:
    """Test turnover calculation."""
    
    def test_empty_events(self):
        """Should handle empty event DataFrame."""
        events = pd.DataFrame()
        result = calculate_turnovers(events)
        
        assert result['total_turnovers'] == 0
    
    def test_dispossessed(self):
        """Should count dispossessed events."""
        events = pd.DataFrame({
            'type': ['Dispossessed', 'Dispossessed', 'Pass']
        })
        result = calculate_turnovers(events)
        
        assert result['dispossessed'] == 2
        assert result['total_turnovers'] == 2
    
    def test_multiple_turnover_types(self):
        """Should count all turnover types."""
        events = pd.DataFrame({
            'type': ['Dispossessed', 'Miscontrol', 'Ball Recovery', 'Pass']
        })
        result = calculate_turnovers(events)
        
        assert result['dispossessed'] == 1
        assert result['miscontrol'] == 1
        assert result['total_turnovers'] == 2  # dispossessed + miscontrol


class TestAttackingPressures:
    """Test attacking pressure calculation."""
    
    def test_empty_events(self):
        """Should handle empty event DataFrame."""
        events = pd.DataFrame()
        result = calculate_attacking_pressures(events)
        
        assert result == 0
    
    def test_no_pressures(self):
        """Should return 0 when no pressures found."""
        events = pd.DataFrame({
            'type': ['Pass', 'Carry'],
            'location': [[40, 40], [50, 50]]
        })
        result = calculate_attacking_pressures(events)
        
        assert result == 0
    
    def test_attacking_pressures(self):
        """Should count pressures in attacking half (x >= 60)."""
        events = pd.DataFrame({
            'type': ['Pressure', 'Pressure', 'Pressure'],
            'location': [[50, 40], [70, 40], [80, 40]]
        })
        result = calculate_attacking_pressures(events)
        
        # Only the 2nd and 3rd pressures are in attacking half (x >= 60)
        assert result == 2


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
