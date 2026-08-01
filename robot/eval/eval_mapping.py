"""
eval_mapping.py — Vyhodnocení efektivity mapování prostoru (kapitola 2.5.3).

Mapování prostoru zajišťuje PolarScan: během řízené 360° rotace vzorkuje
hloubkovou mapu v N_STEPS krocích a sestavuje lokální polární mapu volnosti
(clearance_map). Samotná mapa se do záznamů relací neukládá, takže se její
efektivita vyhodnocuje nepřímo, z časovaných záznamů ve search_logs/:

  * doba jednoho úplného skenu a její reprodukovatelnost napříč zkouškami
    (ověřuje, že odhad natočení z časované rotace nedriftuje),
  * porovnání s teoretickou dobou vypočtenou z kalibračních konstant
    (ukazuje režii, kterou model konstant nezahrnuje),
  * přínos předčasného ukončení skenu při nalezení cíle (stop_on_target),
  * přesnost úhlové lokalizace cíle - azimut první detekce po skenu ukazuje,
    jak daleko od osy kamery cíl po skenu zůstal.

Skenovací fáze se z logu poznají podle přechodů do stavu SCANNING a zpět;
podle následujícího stavu se rozliší, jak sken skončil:

    -> SEARCHING     sken doběhl celý, cíl nenalezen
    -> APPROACHING   sken přerušen nálezem cíle (stop_on_target)
    -> jinak         sken přerušen zvenčí (stop() od uživatele)

Použití
-------
    python eval_mapping.py [--logs search_logs] [--out eval_mapping_results.json]

Nevyžaduje hardware — pracuje pouze nad uloženými záznamy relací.
"""

import argparse
import json
import statistics
from pathlib import Path

from polar_scan import DEG_PER_SEC, N_STEPS, SETTLE_S

# Stavy, do kterých může skenovací fáze přejít (viz state_machine.py)
_SCANNING = "SCANNING"
_FOUND_TARGET = "APPROACHING"
_COMPLETED = "SEARCHING"


def theoretical_scan_duration() -> float:
    """Doba úplného skenu podle kalibračních konstant, bez jakékoli režie."""
    step_angle_deg = 360.0 / N_STEPS
    return N_STEPS * (step_angle_deg / DEG_PER_SEC + SETTLE_S)


def _scans_in_session(events: list[dict], total_duration: float) -> list[tuple[float, str]]:
    """Vrátí (doba, koncový_stav) pro každou skenovací fázi v jedné relaci."""
    scans: list[tuple[float, str]] = []
    start: float | None = None
    for e in events:
        if e.get("type") != "state":
            continue
        if e["state"] == _SCANNING:
            start = e["t"]
        elif start is not None:
            scans.append((e["t"] - start, e["state"]))
            start = None
    if start is not None:
        # Relace skončila, zatímco robot ještě skenoval.
        scans.append((total_duration - start, "INTERRUPTED"))
    return scans


def _bearings_after_scan(events: list[dict]) -> list[float]:
    """
    Azimut první detekce po každém přechodu SCANNING -> APPROACHING.

    Vyjadřuje, jak daleko od osy kamery cíl zůstal poté, co ho sken zachytil
    a zastavil se na něm.
    """
    bearings: list[float] = []
    for i, e in enumerate(events):
        if e.get("type") != "state" or e["state"] != _FOUND_TARGET:
            continue
        previous_states = [x for x in events[:i] if x.get("type") == "state"]
        if not previous_states or previous_states[-1]["state"] != _SCANNING:
            continue  # do APPROACHING se přešlo odjinud než ze skenu
        following = [x for x in events[i:]
                     if x.get("type") == "detection" and x.get("bearing") is not None]
        if following:
            bearings.append(following[0]["bearing"])
    return bearings


def evaluate(logs_dir: Path) -> dict:
    session_files = sorted(logs_dir.glob("*.json"))
    if not session_files:
        raise SystemExit(f"Ve složce {logs_dir} nejsou žádné záznamy relací (*.json).")

    completed: list[float] = []    # doby úplných skenů
    found: list[float] = []        # doby skenů přerušených nálezem cíle
    interrupted: list[float] = []  # doby skenů přerušených zvenčí
    bearings: list[float] = []
    sessions_with_scan = 0

    for path in session_files:
        try:
            data = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        events = data.get("events", [])
        scans = _scans_in_session(events, data.get("duration_s") or 0.0)
        if not scans:
            continue
        sessions_with_scan += 1
        for duration, end_state in scans:
            if end_state == _COMPLETED:
                completed.append(duration)
            elif end_state == _FOUND_TARGET:
                found.append(duration)
            else:
                interrupted.append(duration)
        bearings.extend(_bearings_after_scan(events))

    n_total = len(completed) + len(found) + len(interrupted)
    theoretical = theoretical_scan_duration()
    abs_bearings = [abs(b) for b in bearings]

    result = {
        "n_sessions_scanned": sessions_with_scan,
        "n_scans_total": n_total,
        "n_scans_completed": len(completed),
        "n_scans_target_found": len(found),
        "n_scans_interrupted": len(interrupted),
        "theoretical_scan_s": round(theoretical, 1),
        "n_steps": N_STEPS,
        "deg_per_sec": DEG_PER_SEC,
        "settle_s": SETTLE_S,
    }

    if completed:
        mean_full = statistics.mean(completed)
        result.update({
            "full_scan_mean_s":  round(mean_full, 1),
            "full_scan_stdev_s": round(statistics.stdev(completed), 2) if len(completed) > 1 else None,
            "full_scan_min_s":   round(min(completed), 1),
            "full_scan_max_s":   round(max(completed), 1),
            "overhead_total_s":  round(mean_full - theoretical, 1),
            "overhead_per_step_ms": round((mean_full - theoretical) / N_STEPS * 1000),
        })
    if found:
        result.update({
            "found_scan_mean_s": round(statistics.mean(found), 1),
            "found_scan_min_s":  round(min(found), 1),
            "found_scan_max_s":  round(max(found), 1),
        })
    if completed and found:
        result["saved_per_scan_s"] = round(statistics.mean(completed) - statistics.mean(found), 1)
        result["saved_total_s"] = round(
            (statistics.mean(completed) - statistics.mean(found)) * len(found), 0
        )
    if abs_bearings:
        result.update({
            "n_bearings":          len(abs_bearings),
            "bearing_mean_deg":    round(statistics.mean(abs_bearings), 1),
            "bearing_median_deg":  round(statistics.median(abs_bearings), 1),
            "bearing_max_deg":     round(max(abs_bearings), 1),
        })
    return result


def _print_table(r: dict) -> None:
    W = 74
    row = "  {:<48}{}"
    print(f"\n{'─'*W}")
    print("  EFEKTIVITA MAPOVÁNÍ PROSTORU  (kapitola 2.5.3)")
    print(f"{'─'*W}")
    print(row.format("Vyhodnocených skenů celkem",
                     f"{r['n_scans_total']} (ve {r['n_sessions_scanned']} zkouškách)"))
    print(row.format("Úplný sken (bez nálezu cíle)", f"{r['n_scans_completed']}×"))
    print(row.format("Sken přerušený nálezem cíle", f"{r['n_scans_target_found']}×"))
    print(row.format("Sken přerušený zásahem uživatele", f"{r['n_scans_interrupted']}×"))

    if "full_scan_mean_s" in r:
        stdev = (f"σ = {r['full_scan_stdev_s']:.2f} s; " if r["full_scan_stdev_s"] else "")
        print(row.format("Doba úplného skenu",
                         f"{r['full_scan_mean_s']:.1f} s "
                         f"({stdev}rozsah {r['full_scan_min_s']:.1f}–{r['full_scan_max_s']:.1f} s)"))
    print(row.format("Teoretická doba z kalibračních konstant",
                     f"{r['theoretical_scan_s']:.1f} s"))
    if "overhead_total_s" in r:
        print(row.format("Režie oproti teorii",
                         f"+ {r['overhead_total_s']:.1f} s "
                         f"(přibližně {r['overhead_per_step_ms']} ms na krok)"))
    if "found_scan_mean_s" in r:
        print(row.format("Doba skenu přerušeného nálezem",
                         f"{r['found_scan_mean_s']:.1f} s "
                         f"(rozsah {r['found_scan_min_s']:.1f}–{r['found_scan_max_s']:.1f} s)"))
    if "bearing_mean_deg" in r:
        print(row.format("Ø odchylka cíle od osy kamery po skenu",
                         f"{r['bearing_mean_deg']:.1f}° "
                         f"(medián {r['bearing_median_deg']:.1f}°; "
                         f"max {r['bearing_max_deg']:.1f}°)"))
    print(f"{'─'*W}")

    if "saved_per_scan_s" in r:
        print(f"\n  Předčasné ukončení skenu ušetřilo v průměru "
              f"{r['saved_per_scan_s']:.1f} s na sken "
              f"(celkem přibližně {r['saved_total_s']:.0f} s).")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Vyhodnocení efektivity mapování prostoru ze záznamů relací"
    )
    parser.add_argument("--logs", type=Path, default=Path("search_logs"),
                        metavar="DIR", help="složka se záznamy relací (výchozí: search_logs)")
    parser.add_argument("--out", type=Path, default=Path("eval_mapping_results.json"),
                        metavar="FILE", help="výstupní JSON (výchozí: eval_mapping_results.json)")
    args = parser.parse_args()

    result = evaluate(args.logs.expanduser().resolve())
    _print_table(result)
    args.out.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"  Výsledky uloženy → {args.out}\n")


if __name__ == "__main__":
    main()
