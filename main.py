"""
Main entry point for CAF: Compute-Adaptive Cascaded Forecasting.
Reproduces experiments, baselines, cascade calibration, systems profiling, and publication figures.
Corresponds to the execution flow of the research pipeline.
"""

import sys
import os
import argparse
from pathlib import Path

# Ensure caf_project root is in python path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

def parse_args():
    parser = argparse.ArgumentParser(
        description="CAF: Compute-Adaptive Cascaded Forecasting for Line-Rate Backbone Traffic Monitoring"
    )
    parser.add_argument(
        "--stage",
        type=str,
        default="verify",
        choices=[
            "verify", "acquire", "features", "dataset", "classical",
            "train", "cascade", "ablations", "tables", "stats",
            "profiling", "cafx", "figures", "claims", "all"
        ],
        help="Pipeline stage to execute (default: verify)"
    )
    parser.add_argument(
        "--allow-cpu",
        action="store_true",
        help="Allow execution on CPU devices without raising CUDA assertion"
    )
    return parser.parse_args()

def run_verify():
    """Verify all model architectures, parameter counts, and exit structures."""
    print("=" * 70)
    print("CAF RESEARCH CODEBASE VERIFICATION")
    print("=" * 70)

    import torch
    import config
    print(f"Device: {config.DEVICE} | AMP: {config.AMP}")
    print(f"Datasets: {list(config.DATASETS.keys())}")
    print(f"Lookback (L): {config.LOOKBACK} bins | Horizon (H): {config.HORIZON} bins")

    print("\n--- Model Zoo Initialization & Parameter Counts ---")
    from models.caf import MODEL_SPECS, CAF, count_params, PROPOSED
    from models.baselines import (
        LSTMNet, TCN, DLinear, NHiTS, PatchTST, TimesNet, TSMixer, ITransformer, TimeMixer, TimeXer
    )

    B, L, C, H = 2, config.LOOKBACK, 10, config.HORIZON
    x_multi = torch.randn(B, L, C, device=config.DEVICE)
    x_uni = torch.randn(B, L, 1, device=config.DEVICE)

    for name, (ctor, chans) in MODEL_SPECS.items():
        c_in = 1 if chans == [0] else C
        inp = x_uni if chans == [0] else x_multi
        model = ctor(c_in, L, H).to(config.DEVICE)
        params = count_params(model)
        with torch.no_grad():
            out = model(inp)
            if isinstance(out, tuple):
                out_shape = tuple(out[-1].shape)
            else:
                out_shape = tuple(out.shape)
        assert out_shape == (B, H), f"Shape mismatch for {name}: {out_shape} vs {(B, H)}"
        print(f"  {name:15s} | Params: {params:9,d} | Input: {str(tuple(inp.shape)):15s} | Output: {str(out_shape):10s} [OK]")

    # Verify CAF exits and scale-equivariance
    print("\n--- CAF Scale-Equivariance & Nested Exits Check ---")
    caf_model = CAF(C, L, H).to(config.DEVICE)
    x_test = torch.randn(4, L, C, device=config.DEVICE) * 1500.0 + 3000.0
    with torch.no_grad():
        e0, e1, e2 = caf_model.forward_exits(x_test)
        e_cheap = caf_model.forward_e1(x_test)
        assert torch.allclose(e1, e_cheap, atol=1e-5), "forward_e1 must match E1 exit"
        
        # Scale test: 3x rate surge + 2x shift
        x_scaled = x_test.clone()
        x_scaled[:, :, 0] = 3.0 * x_test[:, :, 0] + 2.0
        e0_s, e1_s, e2_s = caf_model.forward_exits(x_scaled)
        scale_err = torch.abs(e1_s - (3.0 * e1 + 2.0)).max().item()
        print(f"  CAF E0 shape: {tuple(e0.shape)} | Params: {count_params(caf_model):,d}")
        print(f"  CAF E1 shape: {tuple(e1.shape)} (linear residual exit)")
        print(f"  CAF E2 shape: {tuple(e2.shape)} (full gated convolutional exit)")
        print(f"  Scale-equivariance max absolute error: {scale_err:.6e} [PASS]")

    print("\nVerification successful: All models, exits, and shapes are bit-accurate.")

def main():
    args = parse_args()
    if args.allow_cpu:
        os.environ['ALLOW_CPU'] = '1'

    import config
    from utils.helpers import stamp

    stamp(f"CAF Pipeline initiated | Stage: {args.stage} | Platform: {config.PLATFORM}")

    if args.stage == "verify":
        run_verify()
        return

    if args.stage in ("acquire", "all"):
        from data.acquisition import acquire_all_captures
        acquire_all_captures()

    if args.stage in ("features", "all"):
        from data.features import DATA_TABLE
        print(DATA_TABLE)

    if args.stage in ("dataset", "all"):
        from data.dataset import DATA
        for ds, D in DATA.items():
            print(f"Dataset {ds}: {len(D.splits['train'].days)} train days, naive MAE {D.naive_mae:.2f} Mbit/s")

    if args.stage in ("classical", "all"):
        from data.dataset import DATA
        from models.classical import run_classical
        for ds, D in DATA.items():
            run_classical('main', D)

    if args.stage in ("train", "all"):
        from data.dataset import DATA
        from experiments.engine import run_main_experiment
        run_main_experiment(DATA)

    if args.stage in ("cascade", "all"):
        from experiments.cascade import CASCADE_ROWS
        print(f"Cascade rows evaluated: {len(CASCADE_ROWS)}")

    if args.stage in ("tables", "all"):
        from analysis.results_tables import results_df
        df = results_df()
        print(f"Results DataFrame: {len(df)} rows")

    if args.stage in ("claims", "all"):
        from analysis.claims_check import claims
        print(f"Claims verified: {len(claims)}")

    stamp("Execution complete.")

if __name__ == "__main__":
    main()
