"""
Tests for bertini_cascade.py
"""

import pytest
import math
import sys
import os

# Add parent directory to path to import modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from bertini_cascade import (
    TUNGSTEN_Z,
    TUNGSTEN_A,
    TUNGSTEN_N,
    BERTINI_MIN_ENERGY_MEV,
    BERTINI_MAX_ENERGY_MEV,
    nuclear_radius,
    fermi_momentum,
    nucleon_nucleon_cross_section,
    mean_free_path,
    BertiniCascade,
)


class TestNuclearRadius:
    """Tests for the nuclear_radius function"""

    def test_nuclear_radius_tungsten(self):
        """Nuclear radius of W-184 should be ~6.85 fm"""
        r = nuclear_radius(TUNGSTEN_A)
        assert isinstance(r, float)
        assert pytest.approx(r, rel=1e-3) == 1.2 * (184 ** (1.0 / 3.0))

    def test_nuclear_radius_scales_with_cube_root(self):
        """Radius should scale as A^(1/3)"""
        r1 = nuclear_radius(8)
        r2 = nuclear_radius(64)
        # Ratio of radii should be (64/8)^(1/3) = 2
        assert pytest.approx(r2 / r1, rel=1e-6) == 2.0

    def test_nuclear_radius_zero_raises(self):
        """Zero mass number should raise ValueError"""
        with pytest.raises(ValueError):
            nuclear_radius(0)

    def test_nuclear_radius_negative_raises(self):
        """Negative mass number should raise ValueError"""
        with pytest.raises(ValueError):
            nuclear_radius(-1)

    def test_nuclear_radius_positive(self):
        """Nuclear radius should always be positive"""
        for a in [1, 4, 12, 56, 184, 238]:
            assert nuclear_radius(a) > 0


class TestFermiMomentum:
    """Tests for the fermi_momentum function"""

    def test_fermi_momentum_saturation_density(self):
        """Fermi momentum at nuclear saturation density (~0.17 fm^-3) should be ~270 MeV/c"""
        pf = fermi_momentum(0.17)
        assert isinstance(pf, float)
        assert 250.0 < pf < 300.0

    def test_fermi_momentum_increases_with_density(self):
        """Fermi momentum should increase with density"""
        pf_low = fermi_momentum(0.10)
        pf_high = fermi_momentum(0.20)
        assert pf_high > pf_low

    def test_fermi_momentum_zero_density_raises(self):
        """Zero density should raise ValueError"""
        with pytest.raises(ValueError):
            fermi_momentum(0.0)

    def test_fermi_momentum_negative_density_raises(self):
        """Negative density should raise ValueError"""
        with pytest.raises(ValueError):
            fermi_momentum(-0.1)

    def test_fermi_momentum_positive(self):
        """Fermi momentum should always be positive"""
        assert fermi_momentum(0.17) > 0


class TestNucleonNucleonCrossSection:
    """Tests for the nucleon_nucleon_cross_section function"""

    def test_pp_cross_section_at_100_mev(self):
        """pp cross section at 100 MeV should be ~50 mb"""
        sigma = nucleon_nucleon_cross_section(100.0, is_pp=True)
        assert isinstance(sigma, float)
        assert sigma > 40.0

    def test_pn_cross_section_higher_than_pp_at_low_energy(self):
        """pn cross section should be higher than pp at low energies"""
        sigma_pp = nucleon_nucleon_cross_section(200.0, is_pp=True)
        sigma_pn = nucleon_nucleon_cross_section(200.0, is_pp=False)
        assert sigma_pn > sigma_pp

    def test_pp_cross_section_decreases_with_energy(self):
        """pp cross section should decrease with energy (trend)"""
        sigma_low = nucleon_nucleon_cross_section(200.0, is_pp=True)
        sigma_high = nucleon_nucleon_cross_section(2000.0, is_pp=True)
        assert sigma_low > sigma_high

    def test_cross_sections_positive(self):
        """All cross sections should be positive"""
        for T in [100.0, 500.0, 1000.0, 3000.0]:
            assert nucleon_nucleon_cross_section(T, is_pp=True) > 0
            assert nucleon_nucleon_cross_section(T, is_pp=False) > 0


class TestMeanFreePath:
    """Tests for the mean_free_path function"""

    def test_mean_free_path_typical_values(self):
        """Mean free path in tungsten should be physically reasonable (a few fm)"""
        cascade = BertiniCascade()
        mfp = mean_free_path(500.0, cascade.density, cascade.z_fraction)
        assert isinstance(mfp, float)
        assert 1.0 < mfp < 10.0  # typically 2–5 fm

    def test_mean_free_path_decreases_with_density(self):
        """Higher density should give shorter mean free path"""
        mfp_low = mean_free_path(500.0, 0.10, 0.4)
        mfp_high = mean_free_path(500.0, 0.20, 0.4)
        assert mfp_high < mfp_low

    def test_mean_free_path_zero_density_raises(self):
        """Zero density should raise ValueError"""
        with pytest.raises(ValueError):
            mean_free_path(500.0, 0.0, 0.4)

    def test_mean_free_path_invalid_z_fraction_raises(self):
        """z_fraction outside [0, 1] should raise ValueError"""
        with pytest.raises(ValueError):
            mean_free_path(500.0, 0.17, -0.1)
        with pytest.raises(ValueError):
            mean_free_path(500.0, 0.17, 1.1)


class TestBertiniCascadeInit:
    """Tests for BertiniCascade initialisation"""

    def test_default_target_is_tungsten(self):
        """Default target should be tungsten W-184"""
        cascade = BertiniCascade()
        assert cascade.z == TUNGSTEN_Z
        assert cascade.a == TUNGSTEN_A
        assert cascade.n == TUNGSTEN_N

    def test_nuclear_radius_computed(self):
        """Nuclear radius should be computed during init"""
        cascade = BertiniCascade()
        assert cascade.radius_fm == pytest.approx(nuclear_radius(TUNGSTEN_A))

    def test_density_computed(self):
        """Nuclear density should be positive"""
        cascade = BertiniCascade()
        assert cascade.density > 0

    def test_fermi_momentum_computed(self):
        """Fermi momentum should be computed during init"""
        cascade = BertiniCascade()
        assert cascade.fermi_momentum_mev_c > 0

    def test_invalid_mass_number_raises(self):
        """Non-positive mass number should raise ValueError"""
        with pytest.raises(ValueError):
            BertiniCascade(z=74, a=0)

    def test_invalid_atomic_number_raises(self):
        """Atomic number greater than A should raise ValueError"""
        with pytest.raises(ValueError):
            BertiniCascade(z=200, a=184)

    def test_negative_atomic_number_raises(self):
        """Negative atomic number should raise ValueError"""
        with pytest.raises(ValueError):
            BertiniCascade(z=-1, a=184)

    def test_custom_target(self):
        """BertiniCascade can be initialised with a custom target"""
        cascade = BertiniCascade(z=26, a=56)  # iron
        assert cascade.z == 26
        assert cascade.a == 56
        assert cascade.n == 30


class TestBertiniCascadeNuclearThickness:
    """Tests for BertiniCascade.nuclear_thickness"""

    def test_nuclear_thickness_equals_twice_radius(self):
        """Thickness should be 2R"""
        cascade = BertiniCascade()
        assert cascade.nuclear_thickness() == pytest.approx(2.0 * cascade.radius_fm)

    def test_nuclear_thickness_positive(self):
        """Thickness should be positive"""
        assert BertiniCascade().nuclear_thickness() > 0


class TestBertiniCascadeAverageCollisions:
    """Tests for BertiniCascade.average_number_of_collisions"""

    def test_average_collisions_positive(self):
        """Average number of collisions should be positive"""
        cascade = BertiniCascade()
        assert cascade.average_number_of_collisions(500.0) > 0

    def test_average_collisions_greater_than_one(self):
        """Heavy nucleus like tungsten should produce more than one collision on average"""
        cascade = BertiniCascade()
        assert cascade.average_number_of_collisions(500.0) > 1.0

    def test_average_collisions_varies_with_energy(self):
        """Number of collisions should change with proton energy"""
        cascade = BertiniCascade()
        nc_low = cascade.average_number_of_collisions(100.0)
        nc_high = cascade.average_number_of_collisions(3000.0)
        assert nc_low != nc_high


class TestBertiniCascadeRun:
    """Tests for BertiniCascade.run"""

    def test_run_returns_dict(self):
        """run() should return a dict"""
        cascade = BertiniCascade()
        result = cascade.run(500.0)
        assert isinstance(result, dict)

    def test_run_returns_expected_keys(self):
        """run() result should contain all expected keys"""
        cascade = BertiniCascade()
        result = cascade.run(500.0)
        expected_keys = {
            'kinetic_energy_mev',
            'nuclear_radius_fm',
            'nuclear_density_fm3',
            'fermi_momentum_mev_c',
            'mean_free_path_fm',
            'average_collisions',
        }
        assert expected_keys == set(result.keys())

    def test_run_energy_stored_in_result(self):
        """run() should store the input energy in the result"""
        cascade = BertiniCascade()
        T = 1000.0
        result = cascade.run(T)
        assert result['kinetic_energy_mev'] == T

    def test_run_at_minimum_energy(self):
        """run() should succeed at the minimum valid energy (100 MeV)"""
        cascade = BertiniCascade()
        result = cascade.run(BERTINI_MIN_ENERGY_MEV)
        assert result['kinetic_energy_mev'] == BERTINI_MIN_ENERGY_MEV

    def test_run_at_maximum_energy(self):
        """run() should succeed at the maximum valid energy (3000 MeV = 3 GeV)"""
        cascade = BertiniCascade()
        result = cascade.run(BERTINI_MAX_ENERGY_MEV)
        assert result['kinetic_energy_mev'] == BERTINI_MAX_ENERGY_MEV

    def test_run_below_minimum_energy_raises(self):
        """run() should raise ValueError for energy < 100 MeV"""
        cascade = BertiniCascade()
        with pytest.raises(ValueError, match="below the minimum"):
            cascade.run(99.9)

    def test_run_above_maximum_energy_raises(self):
        """run() should raise ValueError for energy > 3000 MeV"""
        cascade = BertiniCascade()
        with pytest.raises(ValueError, match="exceeds the maximum"):
            cascade.run(3000.1)

    def test_run_physical_values(self):
        """run() should return physically reasonable values"""
        cascade = BertiniCascade()
        result = cascade.run(500.0)
        assert result['nuclear_radius_fm'] > 0
        assert result['nuclear_density_fm3'] > 0
        assert result['fermi_momentum_mev_c'] > 0
        assert result['mean_free_path_fm'] > 0
        assert result['average_collisions'] > 0

    def test_run_multiple_energies_in_range(self):
        """run() should succeed for a range of energies"""
        cascade = BertiniCascade()
        for T in [100.0, 200.0, 500.0, 1000.0, 2000.0, 3000.0]:
            result = cascade.run(T)
            assert result['kinetic_energy_mev'] == T


class TestBertiniCascadeTungstenConstants:
    """Tests verifying tungsten-specific constants"""

    def test_tungsten_mass_number(self):
        """Tungsten A should be 184"""
        assert TUNGSTEN_A == 184

    def test_tungsten_atomic_number(self):
        """Tungsten Z should be 74"""
        assert TUNGSTEN_Z == 74

    def test_tungsten_neutron_number(self):
        """Tungsten N should be 110"""
        assert TUNGSTEN_N == 110

    def test_energy_range(self):
        """Energy range should be 100 MeV to 3000 MeV"""
        assert BERTINI_MIN_ENERGY_MEV == 100.0
        assert BERTINI_MAX_ENERGY_MEV == 3000.0
