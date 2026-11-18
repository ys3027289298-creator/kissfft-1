"""End-to-end driver: build materials, run the phase vocoder, evaluate.

Usage:
  python3 -I run_all.py <repo_root> <work_dir>

work_dir gets:
  materials/   input WAVs
  outputs/     stretched WAVs
  results.json full metrics
  report.txt   human-readable summary
"""

import json
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from wavio import read_wav
import gen_materials
import evaluate

MATERIALS = ("pure_tone", "speech", "chord", "noise")
ALPHAS = (0.5, 1.0, 2.0)
NFFT = 2048
# N/8 keeps the synthesis hop at N/4 (75% overlap) for the worst case
# alpha=2 (hop_syn = alpha*hop_ana); overlap is higher for smaller alpha.
HOP_ANA = NFFT // 8


def main():
    repo_root = os.path.abspath(sys.argv[1])
    work = os.path.abspath(sys.argv[2])
    mat_dir = os.path.join(work, "materials")
    out_dir = os.path.join(work, "outputs")
    os.makedirs(mat_dir, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)

    # 1. materials ---------------------------------------------------------
    old_argv = sys.argv
    sys.argv = ["gen_materials.py", mat_dir]
    gen_materials.main()
    sys.argv = old_argv

    cli = os.path.join(repo_root, "phase_vocoder", "pv_stretch")
    if not os.path.exists(cli):
        raise SystemExit("pv_stretch not built; run make in phase_vocoder/")

    results = {}
    for name in MATERIALS:
        in_path = os.path.join(mat_dir, name + ".wav")
        x, fs = read_wav(in_path)
        results[name] = {"sample_rate": fs, "input_samples": len(x)}
        for alpha in ALPHAS:
            tag = ("%g" % alpha).replace(".", "p")
            out_path = os.path.join(out_dir, "%s_a%s.wav" % (name, tag))
            cmd = [cli, "-a", str(alpha), "-n", str(NFFT),
                   "-ha", str(HOP_ANA), in_path, out_path]
            proc = subprocess.run(cmd, capture_output=True, text=True)
            if proc.returncode != 0:
                raise SystemExit("pv_stretch failed:\n" + proc.stderr)
            y, fs2 = read_wav(out_path)
            assert fs2 == fs
            hop_syn = max(1, int(round(alpha * HOP_ANA)))
            results[name]["%g" % alpha] = evaluate.evaluate_pair(
                name, x, y, fs, alpha, hop_syn)
            if alpha == 1.0:
                results[name]["%g" % alpha]["identity_snr_db"] = \
                    evaluate.reconstruction_snr(x, y, fs=fs)

    with open(os.path.join(work, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    report = format_report(results)
    with open(os.path.join(work, "report.txt"), "w") as f:
        f.write(report)
    print(report)


def fmt(v, nd=3):
    return "nan" if v is None or (isinstance(v, float) and np.isnan(v)) \
        else ("%.*f" % (nd, v))


def format_report(r):
    L = []
    L.append("=" * 78)
    L.append("PHASE VOCODER EVALUATION  (NFFT=%d, hop_ana=%d, fs=48000)"
             % (NFFT, HOP_ANA))
    L.append("=" * 78)

    for name in MATERIALS:
        L.append("")
        L.append("-- %s --" % name)
        for alpha in ALPHAS:
            e = r[name]["%g" % alpha]
            L.append("  alpha=%g  duration ratio=%.4f (err %.3f%%)  "
                     "hop-mod %.1f dB (in %.1f dB)  clicks %.2f/s"
                     % (alpha, e["duration_ratio"],
                        e["duration_error_pct"],
                        e["hop_modulation_db"],
                        e["hop_modulation_input_db"],
                        e["click_rate_per_s"]))
            if alpha == 1.0:
                L.append("    identity reconstruction SNR = %.1f dB"
                         % e["identity_snr_db"])
            if "pitch" in e:
                p = e["pitch"]
                if name == "pure_tone":
                    L.append("    pitch bias=%.3f cents  drift=%.3f cents "
                             "[<1 cent: %s]  p95|err|=%.3f cents"
                             % (p["bias_cents"], p["drift_cents"],
                                "PASS" if p["drift_pass_lt_1cent"] else "FAIL",
                                p["output"]["p95_abs_cents"]))
                elif name == "speech":
                    L.append("    F0 %.1f -> %.1f Hz  median bias=%.2f cents"
                             "  frame p95|err|=%.2f cents"
                             % (p["f0_in_median_hz"], p["f0_out_median_hz"],
                                p["median_bias_cents"],
                                p["p95_abs_err_cents"]))
                elif name == "chord":
                    for row in p["notes"]:
                        L.append("    note %7.2f Hz: %.2f -> %.2f Hz, "
                                 "err %+.2f cents"
                                 % (row["note_hz"], row["f_in_hz"],
                                    row["f_out_hz"], row["err_cents"]))
                    L.append("    chord max|err| = %.2f cents"
                             % p["max_abs_err_cents"])
            if "spectral" in e:
                s = e["spectral"]
                L.append("    centroid %.0f -> %.0f Hz (ratio %.4f)  "
                         "flatness ratio %.4f"
                         % (s["centroid_in_hz"], s["centroid_out_hz"],
                            s["centroid_ratio"], s["flatness_ratio"]))
            for tr in e["transients"]:
                if tr["detected"]:
                    L.append("    transient @ %.2fs: timing err %.1f ms, "
                             "attack blur x%.2f"
                             % (tr["t_in"], tr["timing_error_ms"],
                                tr["blur_factor"]))
                else:
                    L.append("    transient @ %.2fs: NOT DETECTED"
                             % tr["t_in"])

    L.append("")
    L.append("=" * 78)
    L.append("ACCEPTANCE")
    L.append("=" * 78)
    checks = []

    for name in MATERIALS:
        for alpha in (0.5, 2.0):
            e = r[name]["%g" % alpha]
            checks.append(("duration %s alpha=%g within 2%%" % (name, alpha),
                           abs(e["duration_error_pct"]) < 2.0))
            # No periodic clicks. An absolute -20 dB floor means the hop
            # modulation is inaudible; below it the exact figure is
            # numerical noise and relative comparison is meaningless. Above
            # the floor, the output must not worsen the input's own
            # modulation by more than 3 dB.
            hm, hmi = e["hop_modulation_db"], e["hop_modulation_input_db"]
            hop_ok = hm < -20.0 or hm < hmi + 3.0
            checks.append(("no periodic clicks %s alpha=%g (hop-mod %.1f dB)"
                           % (name, alpha, hm), hop_ok))
    pt2 = r["pure_tone"]["2"]["pitch"]
    pt05 = r["pure_tone"]["0.5"]["pitch"]
    checks.append(("pure-tone pitch drift < 1 cent (alpha=2)",
                   pt2["drift_pass_lt_1cent"]))
    checks.append(("pure-tone pitch drift < 1 cent (alpha=0.5)",
                   pt05["drift_pass_lt_1cent"]))
    checks.append(("pure-tone pitch bias < 1 cent (alpha=2)",
                   abs(pt2["bias_cents"]) < 1.0))
    checks.append(("speech median F0 bias < 5 cents (both factors)",
                   abs(r["speech"]["2"]["pitch"]["median_bias_cents"]) < 5.0
                   and abs(r["speech"]["0.5"]["pitch"]["median_bias_cents"])
                   < 5.0))
    checks.append(("chord max pitch error < 5 cents (both factors)",
                   r["chord"]["2"]["pitch"]["max_abs_err_cents"] < 5.0
                   and r["chord"]["0.5"]["pitch"]["max_abs_err_cents"]
                   < 5.0))
    checks.append(("alpha=1 interior identity SNR > 40 dB (all materials)",
                   all(r[n]["1"]["identity_snr_db"] > 40.0
                       for n in MATERIALS)))

    for label, ok in checks:
        L.append("  [%s] %s" % ("PASS" if ok else "FAIL", label))
    L.append("")
    n_pass = sum(1 for _, ok in checks if ok)
    L.append("%d/%d checks passed" % (n_pass, len(checks)))
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    main()
