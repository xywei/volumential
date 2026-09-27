"""Reference-bank identity and stable residual tests (host-only)."""
import numpy as np
import pytest

from volumential import rke_table_assembly as rta


@pytest.mark.parametrize("dim", [2, 3])
def test_one_bank_serves_distinct_levels_and_roots(tmp_path, dim):
    options = dict(window_theta=4.0, chan_regular_order=10, chan_radial_order=25)
    cache = tmp_path / "bank"
    reference = rta.get_windowed_channel_table(cache, dim, 2, 1, root_extent=1, **options)
    ids, values = map(np.asarray, reference.get_reduced_table_data())
    checksum = reference._windowed_reference_checksum
    for root, level in [(1, 3), (3, 1), (1e-3, 0), (1e3, 0)]:
        h = root * 2.0**-level
        table = rta.get_windowed_channel_table(
            cache, dim, 2, 1, root_extent=root, source_box_level=level, **options)
        assert table._windowed_cache_disposition == "hit"
        assert table._windowed_reference_checksum == checksum
        assert table.source_box_extent == h
        np.testing.assert_allclose(table.get_reduced_table_data()[1], h*h*values, rtol=3e-16)
        # Independently integrate at the physical extent, so this is not
        # merely a check that the accessor repeats its own scaling formula.
        direct_ids, direct = rta._duffy_channel_entry_values(
            table, rta._normalized_windowed_channel_profile(dim, 1, (h/4)**2), 10, 25)
        order = np.argsort(direct_ids)
        np.testing.assert_array_equal(direct_ids[order], ids)
        np.testing.assert_allclose(direct[order], h*h*values, rtol=2e-12, atol=1e-25*h*h)
    assert len(list((tmp_path / "bank.windowed").glob("*.npz"))) == 1


@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("decay", [3.0+0j, -3.0j, 1.0-2.0j])
def test_residual_near_coincidence_matches_high_precision(dim, decay):
    import mpmath as mp
    from scipy.special import kv
    t = 1/16**2
    p = 6
    if dim == 2:
        kernel = lambda r: kv(0, decay*np.asarray(r))/(2*np.pi)
    else:
        kernel = lambda r: np.exp(-decay*np.asarray(r))/(4*np.pi*np.asarray(r))
    stable = rta.windowed_remainder_profile(
        dim, decay**2, kernel, t, p, stable_canonical_kernel=True)
    radii = np.r_[0., np.geomspace(1e-16, .3, 25)]
    with mp.workdps(90):
        a = mp.mpc(decay)
        tt = mp.mpf(t)
        refs = []
        for radius in radii:
            r = mp.mpf(radius if radius else 1e-40)
            x = r*r/(4*tt)
            if dim == 2:
                val = mp.besselk(0,a*r)
                for m in range(p):
                    val -= (-a*a*tt)**m/mp.factorial(m)*mp.expint(m+1,x)/2
                val /= 2*mp.pi
            else:
                val = mp.exp(-a*r)/r
                for m in range(p):
                    val -= (-a*a*tt)**m/mp.factorial(m)*mp.expint(m+mp.mpf('.5'),x)/(2*mp.sqrt(mp.pi*tt))
                val /= 4*mp.pi
            refs.append(complex(val))
    np.testing.assert_allclose(stable(radii), refs, rtol=3e-13, atol=2e-14)
