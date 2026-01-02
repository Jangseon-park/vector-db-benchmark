#!/usr/bin/env python3
"""
AMD L3 cache partitioning via Linux resctrl (Python implementation).

Designed for vector-db-benchmark/contention:
  - Prober taskset CPUs   -> groupA (bigger L3 share)
  - Amplifier taskset CPUs-> groupB (smaller L3 share)

Typical use (as root):
  python3 contention/amd_cat.py --auto-cpus

Dry-run (no writes):
  python3 contention/amd_cat.py --dry-run --auto-cpus
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Config:
    resctrl: Path
    group_a: str
    group_b: str
    cpus_a: str
    cpus_b: str
    small_ways: int
    dry_run: bool
    auto_cpus: bool
    amplifier_py: Path
    prober_py: Path


def _read_text(p: Path) -> str:
    return p.read_text(encoding="utf-8", errors="ignore").strip()


def _write_text(p: Path, s: str) -> None:
    p.write_text(s + "\n", encoding="utf-8")


def _mount_resctrl(resctrl: Path) -> None:
    resctrl.mkdir(parents=True, exist_ok=True)
    subprocess.run(["mount", "-t", "resctrl", "resctrl", str(resctrl)], check=False)


def _parse_l3_domains_from_schemata_line(l3_line: str) -> list[str]:
    if not l3_line.startswith("L3:"):
        raise ValueError(f"unexpected schemata L3 line: {l3_line!r}")
    dom_part = l3_line[len("L3:") :]
    domains: list[str] = []
    for entry in dom_part.split(";"):
        entry = entry.strip()
        if not entry:
            continue
        dom = entry.split("=", 1)[0].strip()
        if not dom:
            continue
        if not re.fullmatch(r"\d+", dom):
            raise ValueError(f"unexpected domain id: {dom!r} from {entry!r}")
        domains.append(dom)
    if not domains:
        raise ValueError(f"no domains parsed from {l3_line!r}")
    return domains


def _format_mask(hex_width: int, value: int) -> str:
    return f"{value:0{hex_width}x}"


def _extract_taskset_cpu_range_from_code(code: str) -> str | None:
    m = re.search(r"taskset\s+-c\s+([0-9,\-]+)", code)
    return m.group(1) if m else None


def _auto_detect_cpus_from_codebase(amplifier_py: Path, prober_py: Path) -> tuple[str, str]:
    amp_code = amplifier_py.read_text(encoding="utf-8", errors="ignore")
    prob_code = prober_py.read_text(encoding="utf-8", errors="ignore")
    amp = _extract_taskset_cpu_range_from_code(amp_code)
    prob = _extract_taskset_cpu_range_from_code(prob_code)
    if not amp:
        raise SystemExit(f"[!] --auto-cpus enabled, but couldn't find 'taskset -c ...' in {amplifier_py}")
    if not prob:
        raise SystemExit(f"[!] --auto-cpus enabled, but couldn't find 'taskset -c ...' in {prober_py}")
    return prob, amp


def _read_group_state(resctrl: Path) -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, str]] = {}
    if not resctrl.is_dir():
        return out
    for p in sorted(resctrl.iterdir()):
        if not p.is_dir():
            continue
        if p.name in {"info", "mon_data", "mon_groups", "tasks"}:
            continue
        state: dict[str, str] = {}
        cpus_p = p / "cpus"
        schemata_p = p / "schemata"
        if cpus_p.exists():
            state["cpus"] = _read_text(cpus_p)
        if schemata_p.exists():
            state["schemata"] = _read_text(schemata_p)
        if state:
            out[p.name] = state
    return out


def _is_probably_cpumask(s: str) -> bool:
    s = s.strip().lower().removeprefix("0x")
    return bool(re.fullmatch(r"[0-9a-f]{1,8}(?:,[0-9a-f]{1,8})+", s))


def _parse_cpulist(s: str) -> set[int]:
    s = s.strip()
    if not s:
        return set()
    cpus: set[int] = set()
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo_s, hi_s = part.split("-", 1)
            lo = int(lo_s)
            hi = int(hi_s)
            if hi < lo:
                raise ValueError(f"bad range {part!r}")
            for c in range(lo, hi + 1):
                cpus.add(c)
        else:
            cpus.add(int(part))
    return cpus


def _cpulist_to_cpumask_str(cpulist: str, template_mask: str) -> str:
    cpus = _parse_cpulist(cpulist)
    if not cpus:
        return "0"

    chunks = [0] * len(template_mask.split(","))
    for cpu in cpus:
        if cpu < 0:
            raise ValueError(f"negative cpu id: {cpu}")
        idx = cpu // 32
        bit = cpu % 32
        if idx >= len(chunks):
            raise ValueError(
                f"cpu id {cpu} exceeds template mask capacity "
                f"({len(chunks) * 32} bits) from {template_mask!r}"
            )
        chunks[idx] |= 1 << bit

    return ",".join(f"{v:08x}" for v in reversed(chunks))


def _write_resctrl_cpus(cpus_path: Path, cpus_spec: str, template_mask: str) -> None:
    try:
        _write_text(cpus_path, cpus_spec)
        return
    except OSError as e:
        if e.errno != 22:
            raise

    if _is_probably_cpumask(cpus_spec):
        raise

    mask = _cpulist_to_cpumask_str(cpus_spec, template_mask)
    print(f"[*] cpus file rejected cpulist; retrying with cpumask: {mask}")
    _write_text(cpus_path, mask)


def _main(cfg: Config) -> None:
    print("[*] Mounting resctrl...")
    _mount_resctrl(cfg.resctrl)

    info_l3 = cfg.resctrl / "info" / "L3"
    if not info_l3.is_dir():
        raise SystemExit(
            f"[!] resctrl L3 control not available at {info_l3}\n"
            "    Check kernel config (CONFIG_X86_CPU_RESCTRL), BIOS settings, and CPU support."
        )

    print("[*] Reading L3 cbm_mask + domains...")
    cbm_mask_raw = _read_text(info_l3 / "cbm_mask")
    cbm_mask_raw = cbm_mask_raw.lower().removeprefix("0x")
    cbm_mask_val = int(cbm_mask_raw, 16)

    schemata_root = _read_text(cfg.resctrl / "schemata")
    l3_line = next((ln.strip() for ln in schemata_root.splitlines() if ln.strip().startswith("L3:")), "")
    if not l3_line:
        raise SystemExit(f"[!] No L3 line found in {cfg.resctrl}/schemata")

    if cfg.auto_cpus:
        print("[*] Auto-detecting CPU sets from codebase...")
        cpus_a, cpus_b = _auto_detect_cpus_from_codebase(cfg.amplifier_py, cfg.prober_py)
        cfg = Config(
            resctrl=cfg.resctrl,
            group_a=cfg.group_a,
            group_b=cfg.group_b,
            cpus_a=cpus_a,
            cpus_b=cpus_b,
            small_ways=cfg.small_ways,
            dry_run=cfg.dry_run,
            auto_cpus=cfg.auto_cpus,
            amplifier_py=cfg.amplifier_py,
            prober_py=cfg.prober_py,
        )
        print(f"    - cpus_a (groupA/prober)     = {cfg.cpus_a}")
        print(f"    - cpus_b (groupB/amplifier) = {cfg.cpus_b}")

    if cfg.small_ways < 1:
        raise SystemExit("[!] small_ways must be >= 1")

    small_mask_val = (1 << cfg.small_ways) - 1
    if (small_mask_val & ~cbm_mask_val) != 0:
        raise SystemExit(f"[!] small_ways={cfg.small_ways} produces mask outside cbm_mask (0x{cbm_mask_raw})")

    big_mask_val = cbm_mask_val & ~small_mask_val
    hex_width = len(cbm_mask_raw)
    small_mask_hex = _format_mask(hex_width, small_mask_val)
    big_mask_hex = _format_mask(hex_width, big_mask_val)

    print(f"[*] Using masks (cbm_mask=0x{cbm_mask_raw}):")
    print(f"    - {cfg.group_a}(big)   = 0x{big_mask_hex}")
    print(f"    - {cfg.group_b}(small) = 0x{small_mask_hex}")

    print("[*] Building schemata lines from detected domains...")
    domains = _parse_l3_domains_from_schemata_line(l3_line)
    new_l3_a = "L3:" + ";".join(f"{d}={big_mask_hex}" for d in domains)
    new_l3_b = "L3:" + ";".join(f"{d}={small_mask_hex}" for d in domains)

    print("[*] Current resctrl state (before):")
    root_cpus = ""
    try:
        root_cpus = _read_text(cfg.resctrl / "cpus")
        print(f"    - root cpus: {root_cpus}")
    except Exception:
        print("    - root cpus: <unavailable>")

    groups = _read_group_state(cfg.resctrl)
    if not groups:
        print("    - groups: <none>")
    else:
        print("    - groups:")
        for g, st in groups.items():
            cpus_s = st.get("cpus", "<missing>")
            l3_s = ""
            if "schemata" in st:
                l3_s = next((ln for ln in st["schemata"].splitlines() if ln.strip().startswith("L3:")), "").strip()
            print(f"      - {g}: cpus={cpus_s}  L3={l3_s or '<missing>'}")

    print("[*] Planned config:")
    print(f"    - write {cfg.group_a}/schemata: {new_l3_a}")
    print(f"    - write {cfg.group_a}/cpus: {cfg.cpus_a}")
    print(f"    - write {cfg.group_b}/schemata: {new_l3_b}")
    print(f"    - write {cfg.group_b}/cpus: {cfg.cpus_b}")

    if cfg.dry_run:
        print("[*] Dry-run enabled: skipping group creation and resctrl writes.")
        return

    if not root_cpus:
        root_cpus = _read_text(cfg.resctrl / "cpus")

    print("[*] Creating groups...")
    (cfg.resctrl / cfg.group_a).mkdir(parents=True, exist_ok=True)
    (cfg.resctrl / cfg.group_b).mkdir(parents=True, exist_ok=True)

    print(f"[*] Configuring {cfg.group_a}...")
    _write_text(cfg.resctrl / cfg.group_a / "schemata", new_l3_a)
    _write_resctrl_cpus(cfg.resctrl / cfg.group_a / "cpus", cfg.cpus_a, root_cpus)

    print(f"[*] Configuring {cfg.group_b}...")
    _write_text(cfg.resctrl / cfg.group_b / "schemata", new_l3_b)
    _write_resctrl_cpus(cfg.resctrl / cfg.group_b / "cpus", cfg.cpus_b, root_cpus)


def _config_from_env_and_args() -> Config:
    parser = argparse.ArgumentParser(description="AMD resctrl L3 cache partitioning helper (vector-db-benchmark)")
    parser.add_argument("--resctrl", default=os.environ.get("RESCTRL", "/sys/fs/resctrl"))
    parser.add_argument("--group-a", default=os.environ.get("GROUP_A", "groupA"))
    parser.add_argument("--group-b", default=os.environ.get("GROUP_B", "groupB"))
    parser.add_argument("--cpus-a", default=os.environ.get("CPUS_A", "10-19"))
    parser.add_argument("--cpus-b", default=os.environ.get("CPUS_B", "0-9"))
    parser.add_argument("--small-ways", type=int, default=int(os.environ.get("SMALL_WAYS", "4")))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--auto-cpus", action="store_true")
    default_dir = Path(__file__).resolve().parent
    parser.add_argument("--amplifier-py", default=str(default_dir / "Amplifier.py"))
    parser.add_argument("--prober-py", default=str(default_dir / "Prober.py"))
    args = parser.parse_args()

    return Config(
        resctrl=Path(args.resctrl),
        group_a=args.group_a,
        group_b=args.group_b,
        cpus_a=args.cpus_a,
        cpus_b=args.cpus_b,
        small_ways=args.small_ways,
        dry_run=bool(args.dry_run),
        auto_cpus=bool(args.auto_cpus),
        amplifier_py=Path(args.amplifier_py),
        prober_py=Path(args.prober_py),
    )


if __name__ == "__main__":
    _main(_config_from_env_and_args())


