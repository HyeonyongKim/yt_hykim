"""
Test load_universal(bypass_native=True) against real simulation datasets.
Forces the universal inspect -> adapt -> build pipeline, completely
bypassing yt's native frontends.
"""

import warnings
import traceback

warnings.filterwarnings("ignore")

from yt_universal import load_universal, inspect_path

DATASETS = [
    ("/data2/HY.Kim/highz/64Mpc/enzo/hydro_new/RD0048/FD0048", "Enzo"),
    ("/data1/HY.Kim/Data_000337", "GAMER"),
    ("/data2/HY.Kim/highz/64Mpc/gizmo/output_high/snapshot_059.hdf5", "Gizmo"),
    ("/data2/HY.Kim/highz/64Mpc/ramses/halo5/output_00053", "RAMSES"),
    ("/data2/HY.Kim/highz/64Mpc/changa/ncal-h5.000226", "ChaNGa"),
]

GAS_FIELDS = [
    ("gas", "density"),
    ("gas", "temperature"),
    ("gas", "velocity_x"),
]

PARTICLE_FIELDS = [
    ("all", "particle_position_x"),
    ("all", "particle_mass"),
]


def test_dataset(path, label):
    print(f"\n{'='*70}")
    print(f"  {label}: {path}")
    print(f"{'='*70}")

    # Inspect
    sig = inspect_path(path)
    print(f"\n  [Inspection]")
    print(f"    Container:  {sig.container_type.name}")
    print(f"    Layout:     {sig.layout_type.name}")
    print(f"    Family:     {sig.candidate_family}")
    print(f"    Confidence: {sig.confidence:.0%}")
    if sig.yt_hint:
        print(f"    yt hint:    {sig.yt_hint} (IGNORED — bypass_native=True)")

    # Load with bypass_native=True (skip yt.load entirely)
    print(f"\n  [Loading via UNIVERSAL pipeline only]")
    try:
        ds = load_universal(path, bypass_native=True)
        print(f"    Frontend:   {type(ds).__name__}")
        print(f"    Dimensions: {ds.domain_dimensions}")
        print(f"    Fields:     {len(ds.field_list)} on-disk, {len(ds.derived_field_list)} derived")
    except Exception as e:
        print(f"    LOAD FAILED: {type(e).__name__}: {e}")
        return "FAIL"

    # Check universal fields
    ad = ds.all_data()

    print(f"\n  [Universal Gas Fields]")
    for field in GAS_FIELDS:
        try:
            data = ad[field]
            print(f"    {str(field):40s} -> size={data.size:>12d}, "
                  f"min={float(data.min()):12.4e}, max={float(data.max()):12.4e}")
        except Exception as e:
            print(f"    {str(field):40s} -> N/A ({type(e).__name__})")

    print(f"\n  [Universal Particle Fields]")
    for field in PARTICLE_FIELDS:
        try:
            data = ad[field]
            print(f"    {str(field):40s} -> size={data.size:>12d}, "
                  f"min={float(data.min()):12.4e}, max={float(data.max()):12.4e}")
        except Exception as e:
            print(f"    {str(field):40s} -> N/A ({type(e).__name__})")

    print(f"\n  [Native fields (first 10)]")
    for field in ds.field_list[:10]:
        print(f"    {field}")

    return "PASS"


if __name__ == "__main__":
    results = {}
    for path, label in DATASETS:
        try:
            results[label] = test_dataset(path, label)
        except Exception as e:
            results[label] = f"ERROR"
            traceback.print_exc()

    print(f"\n\n{'='*70}")
    print(f"  SUMMARY (bypass_native=True — no yt frontends used)")
    print(f"{'='*70}")
    for label, status in results.items():
        print(f"    {label:20s} {status}")
    print()
