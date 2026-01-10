"""
Tests for radiation.py
"""

import pytest
import sys
import os
import numpy as np

# Add parent directory to path to import radiation module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from radiation import (
    decay_constant,
    half_life,
    activity,
    activity_at_time,
    radiation_point_source,
    radiation_infinite_line_source,
    radiation_semi_infinite_plane
)


class TestDecayConstant:
    """Tests for decay_constant function"""
    
    def test_decay_constant_basic(self):
        """Test basic decay constant calculation"""
        # For half-life of 1 second, decay constant should be ln(2)
        result = decay_constant(1.0)
        assert np.isclose(result, np.log(2))
    
    def test_decay_constant_with_known_values(self):
        """Test decay constant with known half-life values"""
        # If half-life is 10 seconds
        hl = 10.0
        dc = decay_constant(hl)
        # Verify the relationship: λ = ln(2) / t₁/₂
        assert np.isclose(dc, np.log(2) / hl)
    
    def test_decay_constant_negative_half_life(self):
        """Test that negative half-life raises ValueError"""
        with pytest.raises(ValueError, match="Half-life must be positive"):
            decay_constant(-1.0)
    
    def test_decay_constant_zero_half_life(self):
        """Test that zero half-life raises ValueError"""
        with pytest.raises(ValueError, match="Half-life must be positive"):
            decay_constant(0.0)
    
    def test_decay_constant_large_half_life(self):
        """Test decay constant with large half-life"""
        # Large half-life should give small decay constant
        hl = 1e10
        dc = decay_constant(hl)
        assert dc < 1e-9
        assert np.isclose(dc, np.log(2) / hl)


class TestHalfLife:
    """Tests for half_life function"""
    
    def test_half_life_basic(self):
        """Test basic half-life calculation"""
        # For decay constant of ln(2), half-life should be 1
        result = half_life(np.log(2))
        assert np.isclose(result, 1.0)
    
    def test_half_life_with_known_values(self):
        """Test half-life with known decay constant values"""
        dc = 0.0693
        hl = half_life(dc)
        # Verify the relationship: t₁/₂ = ln(2) / λ
        assert np.isclose(hl, np.log(2) / dc)
    
    def test_half_life_negative_decay_constant(self):
        """Test that negative decay constant raises ValueError"""
        with pytest.raises(ValueError, match="Decay constant must be positive"):
            half_life(-0.1)
    
    def test_half_life_zero_decay_constant(self):
        """Test that zero decay constant raises ValueError"""
        with pytest.raises(ValueError, match="Decay constant must be positive"):
            half_life(0.0)
    
    def test_half_life_decay_constant_inverse_relationship(self):
        """Test that decay_constant and half_life are inverse operations"""
        original_hl = 12.5
        dc = decay_constant(original_hl)
        recovered_hl = half_life(dc)
        assert np.isclose(original_hl, recovered_hl)


class TestActivity:
    """Tests for activity function"""
    
    def test_activity_basic(self):
        """Test basic activity calculation"""
        dc = 0.1  # decay constant
        n = 1000  # number of nuclei
        result = activity(dc, n)
        assert result == dc * n
        assert result == 100
    
    def test_activity_zero_decay_constant(self):
        """Test activity with zero decay constant (stable nucleus)"""
        result = activity(0.0, 1000)
        assert result == 0.0
    
    def test_activity_zero_nuclei(self):
        """Test activity with zero nuclei"""
        result = activity(0.1, 0)
        assert result == 0.0
    
    def test_activity_negative_decay_constant(self):
        """Test that negative decay constant raises ValueError"""
        with pytest.raises(ValueError, match="Decay constant must be non-negative"):
            activity(-0.1, 1000)
    
    def test_activity_negative_nuclei(self):
        """Test that negative number of nuclei raises ValueError"""
        with pytest.raises(ValueError, match="Number of nuclei must be non-negative"):
            activity(0.1, -1000)
    
    def test_activity_large_numbers(self):
        """Test activity with large numbers (realistic scenario)"""
        dc = 0.0693  # per second
        n = 1e23  # Avogadro-scale
        result = activity(dc, n)
        assert result == dc * n
        assert result > 1e21


class TestActivityAtTime:
    """Tests for activity_at_time function"""
    
    def test_activity_at_time_basic(self):
        """Test basic activity decay calculation"""
        initial_act = 1000.0
        dc = np.log(2) / 10.0  # half-life of 10 seconds
        time = 10.0  # one half-life
        result = activity_at_time(initial_act, dc, time)
        # After one half-life, activity should be half
        assert np.isclose(result, initial_act / 2)
    
    def test_activity_at_time_zero_time(self):
        """Test that activity at t=0 equals initial activity"""
        initial_act = 1000.0
        dc = 0.1
        result = activity_at_time(initial_act, dc, 0.0)
        assert np.isclose(result, initial_act)
    
    def test_activity_at_time_exponential_decay(self):
        """Test exponential decay relationship"""
        initial_act = 1000.0
        dc = 0.1
        time = 5.0
        result = activity_at_time(initial_act, dc, time)
        expected = initial_act * np.exp(-dc * time)
        assert np.isclose(result, expected)
    
    def test_activity_at_time_multiple_half_lives(self):
        """Test activity after multiple half-lives"""
        initial_act = 1000.0
        hl = 5.0
        dc = np.log(2) / hl
        
        # After 2 half-lives (10 seconds)
        time = 10.0
        result = activity_at_time(initial_act, dc, time)
        # Activity should be 1/4 of initial
        assert np.isclose(result, initial_act / 4)
        
        # After 3 half-lives (15 seconds)
        time = 15.0
        result = activity_at_time(initial_act, dc, time)
        # Activity should be 1/8 of initial
        assert np.isclose(result, initial_act / 8)
    
    def test_activity_at_time_zero_decay_constant(self):
        """Test with zero decay constant (stable nucleus)"""
        initial_act = 1000.0
        result = activity_at_time(initial_act, 0.0, 100.0)
        # Activity should remain constant for stable nucleus
        assert np.isclose(result, initial_act)
    
    def test_activity_at_time_negative_initial_activity(self):
        """Test that negative initial activity raises ValueError"""
        with pytest.raises(ValueError, match="Initial activity must be non-negative"):
            activity_at_time(-1000.0, 0.1, 10.0)
    
    def test_activity_at_time_negative_decay_constant(self):
        """Test that negative decay constant raises ValueError"""
        with pytest.raises(ValueError, match="Decay constant must be non-negative"):
            activity_at_time(1000.0, -0.1, 10.0)
    
    def test_activity_at_time_negative_time(self):
        """Test that negative time raises ValueError"""
        with pytest.raises(ValueError, match="Time must be non-negative"):
            activity_at_time(1000.0, 0.1, -10.0)
    
    def test_activity_at_time_large_time(self):
        """Test activity after very long time (should approach zero)"""
        initial_act = 1000.0
        dc = 0.1
        time = 100.0  # very long time
        result = activity_at_time(initial_act, dc, time)
        # Activity should be very small (less than 5% of initial)
        assert result < initial_act * 0.05
        assert result >= 0
    
    def test_activity_at_time_with_decay_constant_from_half_life(self):
        """Test using decay constant calculated from half-life"""
        initial_act = 5000.0
        hl = 7.3  # years (e.g., for some isotope)
        dc = decay_constant(hl)
        time = 7.3  # one half-life
        
        result = activity_at_time(initial_act, dc, time)
        # After one half-life, activity should be half
        assert np.isclose(result, initial_act / 2, rtol=1e-10)


class TestRadiationPointSource:
    """Tests for radiation_point_source function"""
    
    def test_point_source_basic(self):
        """Test basic point source calculation"""
        act = 1000  # Bq
        dist = 1.0  # meter
        result = radiation_point_source(act, dist)
        expected = act / (4 * np.pi * dist**2)
        assert np.isclose(result, expected)
    
    def test_point_source_inverse_square_law(self):
        """Test that flux follows inverse square law"""
        act = 1000
        dist1 = 1.0
        dist2 = 2.0
        flux1 = radiation_point_source(act, dist1)
        flux2 = radiation_point_source(act, dist2)
        # Flux at distance 2 should be 1/4 of flux at distance 1
        assert np.isclose(flux2, flux1 / 4)
    
    def test_point_source_zero_distance(self):
        """Test that zero distance raises ValueError"""
        with pytest.raises(ValueError, match="Distance must be positive"):
            radiation_point_source(1000, 0.0)
    
    def test_point_source_negative_distance(self):
        """Test that negative distance raises ValueError"""
        with pytest.raises(ValueError, match="Distance must be positive"):
            radiation_point_source(1000, -1.0)
    
    def test_point_source_negative_activity(self):
        """Test that negative activity raises ValueError"""
        with pytest.raises(ValueError, match="Activity must be non-negative"):
            radiation_point_source(-1000, 1.0)
    
    def test_point_source_zero_activity(self):
        """Test point source with zero activity"""
        result = radiation_point_source(0.0, 1.0)
        assert result == 0.0


class TestRadiationInfiniteLineSource:
    """Tests for radiation_infinite_line_source function"""
    
    def test_line_source_basic(self):
        """Test basic line source calculation"""
        act_per_length = 1000  # Bq/m
        dist = 1.0  # meter
        result = radiation_infinite_line_source(act_per_length, dist)
        expected = act_per_length / (2 * np.pi * dist)
        assert np.isclose(result, expected)
    
    def test_line_source_inverse_distance(self):
        """Test that flux follows inverse distance law"""
        act_per_length = 1000
        dist1 = 1.0
        dist2 = 2.0
        flux1 = radiation_infinite_line_source(act_per_length, dist1)
        flux2 = radiation_infinite_line_source(act_per_length, dist2)
        # Flux at distance 2 should be 1/2 of flux at distance 1
        assert np.isclose(flux2, flux1 / 2)
    
    def test_line_source_zero_distance(self):
        """Test that zero distance raises ValueError"""
        with pytest.raises(ValueError, match="Distance must be positive"):
            radiation_infinite_line_source(1000, 0.0)
    
    def test_line_source_negative_distance(self):
        """Test that negative distance raises ValueError"""
        with pytest.raises(ValueError, match="Distance must be positive"):
            radiation_infinite_line_source(1000, -1.0)
    
    def test_line_source_negative_activity(self):
        """Test that negative activity per length raises ValueError"""
        with pytest.raises(ValueError, match="Activity per length must be non-negative"):
            radiation_infinite_line_source(-1000, 1.0)
    
    def test_line_source_zero_activity(self):
        """Test line source with zero activity"""
        result = radiation_infinite_line_source(0.0, 1.0)
        assert result == 0.0


class TestRadiationSemiInfinitePlane:
    """Tests for radiation_semi_infinite_plane function"""
    
    def test_plane_source_basic(self):
        """Test basic plane source calculation"""
        act_per_area = 1000  # Bq/m²
        result = radiation_semi_infinite_plane(act_per_area)
        # For semi-infinite plane, flux is half the surface activity
        expected = act_per_area / 2
        assert np.isclose(result, expected)
    
    def test_plane_source_with_distance(self):
        """Test that flux is independent of distance for infinite plane"""
        act_per_area = 1000
        result1 = radiation_semi_infinite_plane(act_per_area, distance=1.0)
        result2 = radiation_semi_infinite_plane(act_per_area, distance=10.0)
        # Results should be the same regardless of distance
        assert np.isclose(result1, result2)
        assert np.isclose(result1, act_per_area / 2)
    
    def test_plane_source_negative_activity(self):
        """Test that negative activity per area raises ValueError"""
        with pytest.raises(ValueError, match="Activity per area must be non-negative"):
            radiation_semi_infinite_plane(-1000)
    
    def test_plane_source_negative_distance(self):
        """Test that negative distance raises ValueError"""
        with pytest.raises(ValueError, match="Distance must be non-negative"):
            radiation_semi_infinite_plane(1000, distance=-1.0)
    
    def test_plane_source_zero_activity(self):
        """Test plane source with zero activity"""
        result = radiation_semi_infinite_plane(0.0)
        assert result == 0.0
    
    def test_plane_source_zero_distance(self):
        """Test plane source at zero distance (on the surface)"""
        act_per_area = 1000
        result = radiation_semi_infinite_plane(act_per_area, distance=0.0)
        assert np.isclose(result, act_per_area / 2)


class TestIntegration:
    """Integration tests combining multiple functions"""
    
    def test_decay_half_life_activity_chain(self):
        """Test using decay constant to calculate activity"""
        # Given a half-life, calculate activity
        hl = 5.0  # seconds
        num_nuclei = 1e10
        
        dc = decay_constant(hl)
        act = activity(dc, num_nuclei)
        
        # Activity should be positive
        assert act > 0
        # Verify the calculation
        expected_act = (np.log(2) / hl) * num_nuclei
        assert np.isclose(act, expected_act)
    
    def test_point_source_at_different_distances(self):
        """Test point source flux at multiple distances"""
        # Start with half-life and nuclei count
        hl = 10.0
        num_nuclei = 1e15
        
        dc = decay_constant(hl)
        act = activity(dc, num_nuclei)
        
        # Calculate flux at different distances
        distances = [1.0, 2.0, 5.0, 10.0]
        fluxes = [radiation_point_source(act, d) for d in distances]
        
        # All fluxes should be positive
        assert all(f > 0 for f in fluxes)
        
        # Verify inverse square relationship
        for i in range(len(distances) - 1):
            ratio = (distances[i+1] / distances[i])**2
            assert np.isclose(fluxes[i] / fluxes[i+1], ratio, rtol=1e-10)
    
    def test_source_geometry_comparison(self):
        """Compare radiation flux from different source geometries"""
        # Same total activity but different geometries
        total_activity = 1e6  # Bq
        distance = 1.0  # meter
        
        # Point source
        flux_point = radiation_point_source(total_activity, distance)
        
        # Line source (activity per unit length)
        # For comparison, assume 1m length
        act_per_length = total_activity / 1.0
        flux_line = radiation_infinite_line_source(act_per_length, distance)
        
        # Plane source (activity per unit area)
        # For comparison, assume 1m² area
        act_per_area = total_activity / 1.0
        flux_plane = radiation_semi_infinite_plane(act_per_area)
        
        # All should be positive
        assert flux_point > 0
        assert flux_line > 0
        assert flux_plane > 0
        
        # Line source should give higher flux than point source at same distance
        # (because more of the source contributes)
        assert flux_line > flux_point
        
        # Plane source flux should be highest (entire plane contributes)
        assert flux_plane > flux_line
