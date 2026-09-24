# BH absorption of ULDM via a moving imaginary-potential sink.

import numpy as np
import numexpr as ne


def BondiHoyleCalibration(M, v, cs=0.0):
    """
    Sink.Amplitude / Sink.Radius that make ApplySink's early-time absorption
    rate match the classical Bondi-Hoyle-Lyttleton rate
        Mdot_BHL = 4*pi*(G*M)^2*rho / (v^2+cs^2)^{3/2}
    at a given local density rho, in code units (G=1).

    Derivation: for small h*V_sink, ApplySink locally decays density as
    rho_dot(x) = -2*V_sink(x)*rho(x), so integrating the Gaussian kernel
    (assuming rho ~ uniform over it) gives
        Mdot_sink = 2*(2*pi)^{3/2} * Amplitude * R^3 * rho_local.
    Setting R to the Bondi radius 2M/(v^2+cs^2) and equating Mdot_sink to
    Mdot_BHL, the rho_local dependence cancels on both sides (Amplitude must
    be density-independent), leaving Amplitude = (v^2+cs^2)^{3/2}/(8*sqrt(2*pi)*M).

    Caveat: this is a dimensional match between a local-decay sink and a
    flux-based accretion rate, not a dynamical derivation - whether the
    running sim actually reproduces Mdot_BHL depends on whether gravity
    replenishes the depleted region fast enough, and on Radius staying
    resolvable on the grid. Verify empirically, don't trust blindly.

    Returns (Amplitude, Radius).
    """
    vsq = v**2 + cs**2
    Radius = 2 * M / vsq
    Amplitude = vsq**1.5 / (8 * np.sqrt(2 * np.pi) * M)
    return Amplitude, Radius


def ApplySink(psi, rho, TMState, masslist, SinkIdx, SinkAmplitude, SinkRadius,
              xarray, yarray, zarray, h, Vcell, Feedback=True,
              Dynamic=False, VFloor=0.05, RadiusCap=None,
              ConserveMomentum=False, kxarray=None, kyarray=None, kzarray=None):
    """
    Damp psi with a Gaussian imaginary potential centred on the current
    position of particle SinkIdx (from TMState), and optionally feed the
    absorbed ULDM mass into masslist[SinkIdx].

    If Dynamic=True, SinkAmplitude/SinkRadius are ignored and instead
    recomputed every call from the particle's *current* mass and speed via
    BondiHoyleCalibration() - so the sink tracks the Bondi-Hoyle rate as the
    particle accelerates/decelerates and grows, rather than using one fixed
    (Amplitude, Radius) for the whole run. Speed is floored at VFloor to
    keep the Bondi radius finite when the particle is near-stationary (e.g.
    at a bound orbit's turning point). RadiusCap (if given) additionally
    clamps the Bondi radius - as M grows and v drops, R ~ M/v^2 can run away
    well past the box, at which point the "local" sink isn't local anymore.
    Amplitude is left uncapped, so capping R under-delivers relative to the
    naive (divergent) Mdot_BHL rather than over-delivering - the standard
    regularization used by sink-particle accretion schemes at low relative
    velocity.

    If ConserveMomentum=True (requires kxarray/kyarray/kzarray, the sparse
    spectral wavenumber grids already built in Evolve.py), the absorbed
    ULDM's momentum is deposited onto the sink particle's velocity, not just
    its mass. Derivation: psi *= exp(-h*V_sink) is a real multiplicative
    factor, so it rescales amplitude but leaves the phase - and therefore
    the local velocity field v(x) = j(x)/rho(x), with current density
    j = Im(psi* grad psi) - exactly unchanged. The momentum removed from the
    field is then Delta_p = integral of j(x)*(1-exp(-2*h*V_sink(x))) dV,
    the same absorbed-fraction weight used for the mass integral. That
    momentum is added to the particle's (mass-weighted) momentum before
    dividing by the new mass, i.e. a rocket-equation update, not a plain
    velocity add, since mass and velocity change in the same step. Without
    this, the sink is a pure mass leak - the particle gains mass but no
    recoil, which is unphysical (real accretion carries a drag force on the
    accretor set by the relative velocity of the swallowed gas).

    Returns the updated (psi, rho, masslist). TMState is mutated in place
    when ConserveMomentum=True (its velocity entries for SinkIdx are updated
    directly - it is not part of the return tuple).
    """
    bx = TMState[int(SinkIdx * 6) + 0]
    by = TMState[int(SinkIdx * 6) + 1]
    bz = TMState[int(SinkIdx * 6) + 2]

    if Dynamic:
        vx = TMState[int(SinkIdx * 6) + 3]
        vy = TMState[int(SinkIdx * 6) + 4]
        vz = TMState[int(SinkIdx * 6) + 5]
        speed = max(np.sqrt(vx**2 + vy**2 + vz**2), VFloor)
        SinkAmplitude, SinkRadius = BondiHoyleCalibration(masslist[SinkIdx], speed)
        if RadiusCap is not None:
            SinkRadius = min(SinkRadius, RadiusCap)

    SinkPot = ne.evaluate(
        "SinkAmplitude*exp(-0.5*((xarray-bx)**2+(yarray-by)**2+(zarray-bz)**2)/SinkRadius**2)"
    )

    if ConserveMomentum:
        # Current density j = Im(psi* grad psi), computed from psi BEFORE
        # the sink is applied (the field's own velocity field, in the box
        # frame - the same frame TMState velocities are expressed in).
        psi_k = np.fft.fftn(psi)
        dpsi_dx = np.fft.ifftn(1j * kxarray * psi_k)
        dpsi_dy = np.fft.ifftn(1j * kyarray * psi_k)
        dpsi_dz = np.fft.ifftn(1j * kzarray * psi_k)
        psi_conj = np.conj(psi)
        jx = np.imag(psi_conj * dpsi_dx)
        jy = np.imag(psi_conj * dpsi_dy)
        jz = np.imag(psi_conj * dpsi_dz)

    rho_preabsorb = rho
    psi = ne.evaluate("psi*exp(-h*SinkPot)")
    rho = ne.evaluate("abs(abs(psi)**2)").real

    if Feedback:
        M_old = masslist[SinkIdx]
        dM = Vcell * np.sum(rho_preabsorb - rho)
        masslist[SinkIdx] = M_old + dM

        if ConserveMomentum:
            AbsorbFrac = ne.evaluate("1 - exp(-2*h*SinkPot)")
            dpx = Vcell * np.sum(jx * AbsorbFrac)
            dpy = Vcell * np.sum(jy * AbsorbFrac)
            dpz = Vcell * np.sum(jz * AbsorbFrac)

            vx_old = TMState[int(SinkIdx * 6) + 3]
            vy_old = TMState[int(SinkIdx * 6) + 4]
            vz_old = TMState[int(SinkIdx * 6) + 5]

            M_new = masslist[SinkIdx]
            TMState[int(SinkIdx * 6) + 3] = (M_old * vx_old + dpx) / M_new
            TMState[int(SinkIdx * 6) + 4] = (M_old * vy_old + dpy) / M_new
            TMState[int(SinkIdx * 6) + 5] = (M_old * vz_old + dpz) / M_new

    return psi, rho, masslist
