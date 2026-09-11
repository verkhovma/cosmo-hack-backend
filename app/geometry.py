import numpy as np

R = 6371.0
MU = 398600.435507
T_EARTH = 86164.09054

def satellite_positions(scenario, t_array):
    """Возвращает (n_times, n_sats, 3) земные координаты."""
    h = scenario.environment.altitude_km
    i = np.radians(scenario.environment.inclination_deg)
    r = R + h
    n = np.sqrt(MU / r**3)

    sats = [s for s in scenario.design.satellites
            if s.launch_batch <= scenario.design.launch_stage]
    planes = {p.id: p for p in scenario.design.planes}

    # u(t) для всех спутников и времён
    slot = np.array([s.slot_deg for s in sats])
    phase = np.array([planes[s.plane_id].phase_deg for s in sats])
    raan = np.radians([planes[s.plane_id].raan_deg for s in sats])

    u = np.radians(slot + phase)[None, :] + n * t_array[:, None]  # (T, N)

    # Инерциальные координаты
    x = r * (np.cos(raan)*np.cos(u) - np.sin(raan)*np.sin(u)*np.cos(i))
    y = r * (np.sin(raan)*np.cos(u) + np.cos(raan)*np.sin(u)*np.cos(i))
    z = r * np.sin(u) * np.sin(i)

    # Поворот к Земле
    theta = np.radians(scenario.environment.earth_angle0_deg) \
            + 2*np.pi * t_array / T_EARTH
    ct, st = np.cos(theta)[:, None], np.sin(theta)[:, None]
    xe = ct*x + st*y
    ye = -st*x + ct*y
    ze = z
    return np.stack([xe, ye, ze], axis=-1)  # (T, N, 3)



def ground_contacts(pos, ground_sites, min_elev_deg):
    """pos: (T, N, 3). Возвращает булеву матрицу (T, N, G)."""
    g = np.array([R*np.cos(np.radians(s.lat_deg))*np.cos(np.radians(s.lon_deg))
                  for s in ground_sites])  # и аналогично y, z
    # ... векторно: elev = arcsin(((s-g)·g)/(‖s-g‖·R))
    return elev >= np.radians(min_elev_deg)

def isl_contacts(pos, isl_range_km):
    """pos: (T, N, 3) → (T, N, N) булева матрица связей."""
    diff = pos[:, :, None, :] - pos[:, None, :, :]      # (T, N, N, 3)
    dist = np.linalg.norm(diff, axis=-1)
    a = pos[:, :, None, :]
    d = diff
    dd = np.einsum('...i,...i->...', d, d) + 1e-9
    q = np.clip(-np.einsum('...i,...i->...', a, d) / dd, 0, 1)
    closest = a + q[..., None] * d
    r_closest = np.linalg.norm(closest, axis=-1)
    return (dist < isl_range_km) & (r_closest > R)
