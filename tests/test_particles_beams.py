"""
Tests for particles_beams.py
"""

import math
import pytest
import sys
import os

# Add parent directory to path to import modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from particles_beams import (
    beam_power,
    particles_per_second,
    particles_per_pulse,
    average_current,
    duty_factor,
    peak_current_from_average,
    lorentz_gamma,
    lorentz_beta,
    beam_velocity,
    magnetic_rigidity,
    beam_power_from_particle,
    particles_per_second_from_particle,
    lorentz_gamma_from_particle,
    lorentz_beta_from_particle,
)
from nuclear_constants import ELEMENTARY_CHARGE, MEV_TO_J, SPEED_OF_LIGHT
from particle import particle, proton, electron


# ---------------------------------------------------------------------------
# beam_power
# ---------------------------------------------------------------------------

class TestBeamPower:
    """Tests for beam_power function"""

    def test_cw_proton_beam_100_mev_1_ma(self):
        """1 mA of 100 MeV singly charged particles: P = I * E_MeV * 1e6"""
        # P = I * E_J / q = 1e-3 * (100 * MEV_TO_J) / ELEMENTARY_CHARGE
        expected = 1e-3 * 100 * MEV_TO_J / ELEMENTARY_CHARGE
        result = beam_power(1e-3, 100.0)
        assert math.isclose(result, expected, rel_tol=1e-9)

    def test_cw_beam_simple_value(self):
        """For singly-charged particles P (W) = I (A) * E (MeV) * 1e6"""
        # 1 mA, 100 MeV => 100 kW
        assert math.isclose(beam_power(1e-3, 100.0), 1e5, rel_tol=1e-9)

    def test_duty_factor_scales_power(self):
        """Halving the duty factor should halve the average power"""
        p_cw = beam_power(1e-3, 100.0, duty_factor=1.0)
        p_half = beam_power(1e-3, 100.0, duty_factor=0.5)
        assert math.isclose(p_half, p_cw * 0.5, rel_tol=1e-9)

    def test_doubly_charged_beam(self):
        """For charge_number=2 power is halved compared to charge_number=1"""
        p1 = beam_power(1e-3, 100.0, charge_number=1)
        p2 = beam_power(1e-3, 100.0, charge_number=2)
        assert math.isclose(p2, p1 / 2, rel_tol=1e-9)

    def test_zero_current_gives_zero_power(self):
        assert beam_power(0.0, 100.0) == 0.0

    def test_zero_energy_gives_zero_power(self):
        assert beam_power(1e-3, 0.0) == 0.0

    def test_negative_current_raises(self):
        with pytest.raises(ValueError, match="Current must be non-negative"):
            beam_power(-1e-3, 100.0)

    def test_negative_energy_raises(self):
        with pytest.raises(ValueError, match="Energy must be non-negative"):
            beam_power(1e-3, -100.0)

    def test_duty_factor_zero_raises(self):
        with pytest.raises(ValueError, match="Duty factor must be in"):
            beam_power(1e-3, 100.0, duty_factor=0.0)

    def test_duty_factor_greater_than_one_raises(self):
        with pytest.raises(ValueError, match="Duty factor must be in"):
            beam_power(1e-3, 100.0, duty_factor=1.1)

    def test_zero_charge_raises(self):
        with pytest.raises(ValueError, match="Charge number must not be zero"):
            beam_power(1e-3, 100.0, charge_number=0)


# ---------------------------------------------------------------------------
# particles_per_second
# ---------------------------------------------------------------------------

class TestParticlesPerSecond:
    """Tests for particles_per_second function"""

    def test_basic_calculation(self):
        """N = I / q"""
        current = 1e-3  # 1 mA
        expected = current / ELEMENTARY_CHARGE
        assert math.isclose(particles_per_second(current), expected, rel_tol=1e-9)

    def test_doubly_charged(self):
        """Doubly charged particles give half the flux for the same current"""
        n1 = particles_per_second(1e-3, charge_number=1)
        n2 = particles_per_second(1e-3, charge_number=2)
        assert math.isclose(n2, n1 / 2, rel_tol=1e-9)

    def test_zero_current(self):
        assert particles_per_second(0.0) == 0.0

    def test_negative_current_raises(self):
        with pytest.raises(ValueError, match="Current must be non-negative"):
            particles_per_second(-1e-3)

    def test_zero_charge_raises(self):
        with pytest.raises(ValueError, match="Charge number must not be zero"):
            particles_per_second(1e-3, charge_number=0)


# ---------------------------------------------------------------------------
# particles_per_pulse
# ---------------------------------------------------------------------------

class TestParticlesPerPulse:
    """Tests for particles_per_pulse function"""

    def test_basic_calculation(self):
        """N_pulse = I_peak * t_pulse / q"""
        I_peak = 0.1    # A
        t = 1e-3        # s
        expected = I_peak * t / ELEMENTARY_CHARGE
        assert math.isclose(particles_per_pulse(I_peak, t), expected, rel_tol=1e-9)

    def test_zero_pulse_length(self):
        assert particles_per_pulse(0.1, 0.0) == 0.0

    def test_zero_peak_current(self):
        assert particles_per_pulse(0.0, 1e-3) == 0.0

    def test_negative_peak_current_raises(self):
        with pytest.raises(ValueError, match="Peak current must be non-negative"):
            particles_per_pulse(-0.1, 1e-3)

    def test_negative_pulse_length_raises(self):
        with pytest.raises(ValueError, match="Pulse length must be non-negative"):
            particles_per_pulse(0.1, -1e-3)

    def test_zero_charge_raises(self):
        with pytest.raises(ValueError, match="Charge number must not be zero"):
            particles_per_pulse(0.1, 1e-3, charge_number=0)

    def test_doubly_charged_gives_half_particles(self):
        n1 = particles_per_pulse(0.1, 1e-3, charge_number=1)
        n2 = particles_per_pulse(0.1, 1e-3, charge_number=2)
        assert math.isclose(n2, n1 / 2, rel_tol=1e-9)


# ---------------------------------------------------------------------------
# average_current
# ---------------------------------------------------------------------------

class TestAverageCurrent:
    """Tests for average_current function"""

    def test_basic_calculation(self):
        """I_avg = I_peak * df"""
        assert math.isclose(average_current(0.1, 0.05), 0.005, rel_tol=1e-9)

    def test_full_duty_factor(self):
        """CW beam: average == peak"""
        assert math.isclose(average_current(0.1, 1.0), 0.1, rel_tol=1e-9)

    def test_zero_peak_current(self):
        assert average_current(0.0, 0.5) == 0.0

    def test_negative_peak_current_raises(self):
        with pytest.raises(ValueError, match="Peak current must be non-negative"):
            average_current(-0.1, 0.5)

    def test_duty_factor_zero_raises(self):
        with pytest.raises(ValueError, match="Duty factor must be in"):
            average_current(0.1, 0.0)

    def test_duty_factor_greater_than_one_raises(self):
        with pytest.raises(ValueError, match="Duty factor must be in"):
            average_current(0.1, 1.5)


# ---------------------------------------------------------------------------
# duty_factor
# ---------------------------------------------------------------------------

class TestDutyFactor:
    """Tests for duty_factor function"""

    def test_basic_calculation(self):
        """df = t_pulse * f_rep"""
        assert math.isclose(duty_factor(1e-3, 50.0), 0.05, rel_tol=1e-9)

    def test_cw_equivalent(self):
        """1 s pulse at 1 Hz = duty factor 1"""
        assert math.isclose(duty_factor(1.0, 1.0), 1.0, rel_tol=1e-9)

    def test_zero_pulse_length(self):
        assert duty_factor(0.0, 100.0) == 0.0

    def test_negative_pulse_length_raises(self):
        with pytest.raises(ValueError, match="Pulse length must be non-negative"):
            duty_factor(-1e-3, 50.0)

    def test_negative_rep_rate_raises(self):
        with pytest.raises(ValueError, match="Repetition rate must be non-negative"):
            duty_factor(1e-3, -50.0)

    def test_duty_factor_exceeds_one_raises(self):
        with pytest.raises(ValueError, match="Duty factor exceeds 1"):
            duty_factor(0.1, 50.0)  # 0.1 s * 50 Hz = 5 > 1


# ---------------------------------------------------------------------------
# peak_current_from_average
# ---------------------------------------------------------------------------

class TestPeakCurrentFromAverage:
    """Tests for peak_current_from_average function"""

    def test_basic_calculation(self):
        """I_peak = I_avg / df"""
        assert math.isclose(peak_current_from_average(5e-3, 0.05), 0.1, rel_tol=1e-9)

    def test_cw_beam(self):
        """CW beam: peak == average"""
        assert math.isclose(peak_current_from_average(0.1, 1.0), 0.1, rel_tol=1e-9)

    def test_zero_average_current(self):
        assert peak_current_from_average(0.0, 0.5) == 0.0

    def test_negative_average_raises(self):
        with pytest.raises(ValueError, match="Average current must be non-negative"):
            peak_current_from_average(-1e-3, 0.5)

    def test_duty_factor_zero_raises(self):
        with pytest.raises(ValueError, match="Duty factor must be in"):
            peak_current_from_average(1e-3, 0.0)

    def test_duty_factor_greater_than_one_raises(self):
        with pytest.raises(ValueError, match="Duty factor must be in"):
            peak_current_from_average(1e-3, 2.0)

    def test_round_trip_with_average_current(self):
        """peak_current_from_average(average_current(I, df), df) == I"""
        I_peak = 0.1
        df = 0.05
        I_avg = average_current(I_peak, df)
        assert math.isclose(peak_current_from_average(I_avg, df), I_peak, rel_tol=1e-9)


# ---------------------------------------------------------------------------
# lorentz_gamma
# ---------------------------------------------------------------------------

class TestLorentzGamma:
    """Tests for lorentz_gamma function"""

    def test_zero_kinetic_energy(self):
        """A particle at rest has gamma = 1"""
        assert math.isclose(lorentz_gamma(0.0, 938.272), 1.0, rel_tol=1e-9)

    def test_non_relativistic_proton(self):
        """100 MeV proton (KE << m0c²): gamma ~= 1 + KE/m0c²"""
        ke = 100.0
        m = 938.272
        expected = 1.0 + ke / m
        assert math.isclose(lorentz_gamma(ke, m), expected, rel_tol=1e-9)

    def test_ultra_relativistic(self):
        """Very high energy: gamma >> 1"""
        gamma = lorentz_gamma(1e6, 938.272)
        assert gamma > 1000

    def test_negative_energy_raises(self):
        with pytest.raises(ValueError, match="Kinetic energy must be non-negative"):
            lorentz_gamma(-1.0, 938.272)

    def test_zero_rest_mass_raises(self):
        with pytest.raises(ValueError, match="Rest mass energy must be positive"):
            lorentz_gamma(100.0, 0.0)

    def test_negative_rest_mass_raises(self):
        with pytest.raises(ValueError, match="Rest mass energy must be positive"):
            lorentz_gamma(100.0, -938.272)


# ---------------------------------------------------------------------------
# lorentz_beta
# ---------------------------------------------------------------------------

class TestLorentzBeta:
    """Tests for lorentz_beta function"""

    def test_zero_kinetic_energy(self):
        """A particle at rest has beta = 0"""
        assert math.isclose(lorentz_beta(0.0, 938.272), 0.0, abs_tol=1e-12)

    def test_beta_less_than_one(self):
        """Beta must always be < 1 (sub-luminal)"""
        assert lorentz_beta(100.0, 938.272) < 1.0

    def test_high_energy_approaches_one(self):
        """Ultra-relativistic particles approach beta = 1"""
        beta = lorentz_beta(1e9, 938.272)
        assert beta > 0.9999

    def test_beta_formula(self):
        """Verify beta = sqrt(1 - 1/gamma^2)"""
        ke = 100.0
        m = 938.272
        gamma = lorentz_gamma(ke, m)
        expected = math.sqrt(1.0 - 1.0 / gamma**2)
        assert math.isclose(lorentz_beta(ke, m), expected, rel_tol=1e-9)

    def test_negative_energy_raises(self):
        with pytest.raises(ValueError, match="Kinetic energy must be non-negative"):
            lorentz_beta(-1.0, 938.272)

    def test_zero_rest_mass_raises(self):
        with pytest.raises(ValueError, match="Rest mass energy must be positive"):
            lorentz_beta(100.0, 0.0)


# ---------------------------------------------------------------------------
# beam_velocity
# ---------------------------------------------------------------------------

class TestBeamVelocity:
    """Tests for beam_velocity function"""

    def test_zero_kinetic_energy(self):
        """A particle at rest has v = 0"""
        assert math.isclose(beam_velocity(0.0, 938.272), 0.0, abs_tol=1e-6)

    def test_velocity_less_than_c(self):
        """Velocity must be sub-luminal"""
        assert beam_velocity(100.0, 938.272) < SPEED_OF_LIGHT

    def test_velocity_formula(self):
        """v = beta * c"""
        ke = 100.0
        m = 938.272
        beta = lorentz_beta(ke, m)
        expected = beta * SPEED_OF_LIGHT
        assert math.isclose(beam_velocity(ke, m), expected, rel_tol=1e-9)

    def test_ultra_relativistic_approaches_c(self):
        """High-energy particles approach c"""
        v = beam_velocity(1e9, 938.272)
        assert v > 0.9999 * SPEED_OF_LIGHT

    def test_negative_energy_raises(self):
        with pytest.raises(ValueError, match="Kinetic energy must be non-negative"):
            beam_velocity(-1.0, 938.272)

    def test_zero_rest_mass_raises(self):
        with pytest.raises(ValueError, match="Rest mass energy must be positive"):
            beam_velocity(100.0, 0.0)


# ---------------------------------------------------------------------------
# magnetic_rigidity
# ---------------------------------------------------------------------------

class TestMagneticRigidity:
    """Tests for magnetic_rigidity function"""

    def test_proton_100_mev(self):
        """100 MeV proton: Bρ should be approximately 1.48 T·m"""
        # pc = sqrt((100+938.272)^2 - 938.272^2) MeV
        ke = 100.0
        m = 938.272
        total_e = ke + m
        pc_mev = math.sqrt(total_e**2 - m**2)
        p_si = pc_mev * MEV_TO_J / SPEED_OF_LIGHT
        expected = p_si / ELEMENTARY_CHARGE
        result = magnetic_rigidity(ke, m)
        assert math.isclose(result, expected, rel_tol=1e-9)

    def test_doubly_charged_half_rigidity(self):
        """Doubling the charge halves the rigidity"""
        br1 = magnetic_rigidity(100.0, 938.272, charge_number=1)
        br2 = magnetic_rigidity(100.0, 938.272, charge_number=2)
        assert math.isclose(br2, br1 / 2, rel_tol=1e-9)

    def test_rigidity_increases_with_energy(self):
        """Higher energy means larger rigidity"""
        br_low = magnetic_rigidity(100.0, 938.272)
        br_high = magnetic_rigidity(1000.0, 938.272)
        assert br_high > br_low

    def test_negative_energy_raises(self):
        with pytest.raises(ValueError, match="Kinetic energy must be non-negative"):
            magnetic_rigidity(-1.0, 938.272)

    def test_zero_rest_mass_raises(self):
        with pytest.raises(ValueError, match="Rest mass energy must be positive"):
            magnetic_rigidity(100.0, 0.0)

    def test_zero_charge_raises(self):
        with pytest.raises(ValueError, match="Charge number must not be zero"):
            magnetic_rigidity(100.0, 938.272, charge_number=0)


# ---------------------------------------------------------------------------
# beam_power_from_particle
# ---------------------------------------------------------------------------

class TestBeamPowerFromParticle:
    """Tests for beam_power_from_particle function"""

    def test_proton_matches_direct_function(self):
        """Result should match beam_power() with equivalent parameters"""
        p = particle.from_name("proton")
        p.energy = 100.0
        charge_number = p.charge / ELEMENTARY_CHARGE
        expected = beam_power(1e-3, p.energy, charge_number=charge_number)
        result = beam_power_from_particle(p, 1e-3)
        assert math.isclose(result, expected, rel_tol=1e-9)

    def test_with_duty_factor(self):
        """Duty factor is passed through correctly"""
        p = particle.from_name("proton")
        p.energy = 100.0
        p_cw = beam_power_from_particle(p, 1e-3, duty_factor=1.0)
        p_half = beam_power_from_particle(p, 1e-3, duty_factor=0.5)
        assert math.isclose(p_half, p_cw * 0.5, rel_tol=1e-9)

    def test_non_particle_raises(self):
        with pytest.raises(TypeError, match="p must be an instance of particle"):
            beam_power_from_particle("not a particle", 1e-3)

    def test_zero_charge_particle_raises(self):
        p = particle()  # default charge = 0
        with pytest.raises(ValueError, match="Particle charge must not be zero"):
            beam_power_from_particle(p, 1e-3)


# ---------------------------------------------------------------------------
# particles_per_second_from_particle
# ---------------------------------------------------------------------------

class TestParticlesPerSecondFromParticle:
    """Tests for particles_per_second_from_particle function"""

    def test_proton_matches_direct_function(self):
        """Result should match particles_per_second() with charge_number=1"""
        p = particle.from_name("proton")
        expected = particles_per_second(1e-3, charge_number=1)
        result = particles_per_second_from_particle(p, 1e-3)
        assert math.isclose(result, expected, rel_tol=1e-9)

    def test_non_particle_raises(self):
        with pytest.raises(TypeError, match="p must be an instance of particle"):
            particles_per_second_from_particle(42, 1e-3)

    def test_zero_charge_particle_raises(self):
        p = particle()  # default charge = 0
        with pytest.raises(ValueError, match="Particle charge must not be zero"):
            particles_per_second_from_particle(p, 1e-3)


# ---------------------------------------------------------------------------
# lorentz_gamma_from_particle
# ---------------------------------------------------------------------------

class TestLorentzGammaFromParticle:
    """Tests for lorentz_gamma_from_particle function"""

    def test_proton_100_mev(self):
        """gamma for 100 MeV proton should match direct calculation"""
        p = particle.from_name("proton")
        p.energy = 100.0
        expected = lorentz_gamma(100.0, p.mass_mev)
        assert math.isclose(lorentz_gamma_from_particle(p), expected, rel_tol=1e-9)

    def test_non_particle_raises(self):
        with pytest.raises(TypeError, match="p must be an instance of particle"):
            lorentz_gamma_from_particle("not a particle")

    def test_zero_mass_particle_raises(self):
        p = particle()  # mass_mev defaults to 0.0
        with pytest.raises(ValueError, match="must be positive"):
            lorentz_gamma_from_particle(p)


# ---------------------------------------------------------------------------
# lorentz_beta_from_particle
# ---------------------------------------------------------------------------

class TestLorentzBetaFromParticle:
    """Tests for lorentz_beta_from_particle function"""

    def test_proton_100_mev(self):
        """beta for 100 MeV proton should match direct calculation"""
        p = particle.from_name("proton")
        p.energy = 100.0
        expected = lorentz_beta(100.0, p.mass_mev)
        assert math.isclose(lorentz_beta_from_particle(p), expected, rel_tol=1e-9)

    def test_beta_less_than_one(self):
        """beta must always be sub-luminal"""
        p = particle.from_name("proton")
        p.energy = 100.0
        assert lorentz_beta_from_particle(p) < 1.0

    def test_non_particle_raises(self):
        with pytest.raises(TypeError, match="p must be an instance of particle"):
            lorentz_beta_from_particle(None)

    def test_zero_mass_particle_raises(self):
        p = particle()  # mass_mev defaults to 0.0
        with pytest.raises(ValueError, match="must be positive"):
            lorentz_beta_from_particle(p)
