# OmniBot — autonomní vyhledávání objektů

Softwarový systém pro mobilního robota OmniBot, který kombinuje detekci nápojových
obalů neuronovou sítí na akcelerátoru Hailo s navigačním a vyhledávacím zásobníkem
umožňujícím robotovi samostatně prohledat neznámý prostor a cílový objekt v něm nalézt.

Repozitář tvoří praktickou část bakalářské práce *Softwarový systém mobilního robota
s využitím neuronových sítí pro detekci objektů* (Jan Neumann, Univerzita Pardubice, 2026).

---

## Hardware

| Komponenta | Popis |
|---|---|
| Raspberry Pi 5 | hlavní výpočetní jednotka |
| Hailo-8 (AI HAT+) | akcelerátor pro inferenci obou neuronových sítí |
| Kamerový modul (CSI) | zdroj obrazu; **namontován otočený o 180°** |
| PCA9685 (I2C) | řadič PWM pro 4 motory omnidirekcionálního podvozku |
| HC-SR04 (GPIO) | ultrazvukový dálkoměr pro bezpečnostní zóny |

## Softwarové požadavky

- Raspberry Pi OS (Bookworm) s funkčním `rpicam-vid`
- Python 3.13 ve virtuálním prostředí `venv_hailo_rpi_examples/`
- HailoRT SDK (instaluje se skriptem níže)
- [`just`](https://github.com/casey/just) — spouštěč příkazů (`sudo apt install just`)

## Instalace

```bash
git clone https://github.com/TemplairTheWise/Omnibot.git
cd Omnibot

# Stažení a instalace Hailo SDK + systémových modelů (čte hailo/config.yaml)
just install
```

Detekční model `robot/models/yolo26_split.hef` je součástí repozitáře.
Hloubkový model `scdepthv3.hef` stáhne `just install` do `hailo/resources/models/hailo8l/`.

## Rychlý start

**Před každou prací je nutné aktivovat prostředí** — nastaví `PYTHONPATH`
a aktivuje virtuální prostředí pro aktuální relaci terminálu:

```bash
source setup_env.sh
```

Poté už stačí:

```bash
just              # vypíše všechny dostupné příkazy
just server       # webové rozhraní → http://<ip-adresa-pi>:5000
```

Ve webovém rozhraní je záložka **Autonomous** (spuštění hledání, výběr cílové třídy,
volba úvodního skenu, telemetrie) a **Manual** (D-pad, rotace, gripper, klávesové zkratky
WASD / Q,E / G / mezerník).

## Přehled příkazů

### Provoz

| Příkaz | Popis |
|---|---|
| `just server` | webový ovládací server na portu 5000 |
| `just drive` | klávesnicový teleop bez webového rozhraní |

### Diagnostika hardwaru

| Příkaz | Popis |
|---|---|
| `just distance` | živé odečty ze sonaru |
| `just force-test` | test servo kanálů, kalibrace konstant `_TRIM` |
| `just detect-image` | detekce na jednom obrázku → `robot/assets/output_detection.jpg` |
| `just detect-video` | živá detekce z kamery (okno; ukončení klávesou `q`) |

### Sběr dat a vyhodnocení

| Příkaz | Popis |
|---|---|
| `just capture [ADRESÁŘ]` | interaktivní sběr testovací sady (SPACE = uložit, Q = konec) |
| `just review [ADRESÁŘ]` | ruční kontrola a oprava anotací — **nutné před vyhodnocením** |
| `just eval-detection ADRESÁŘ` | mAP@0,5 vyhodnocení detektoru |
| `just eval-detection-agnostic ADRESÁŘ` | vyhodnocení bez ohledu na třídu (jen lokalizace) |
| `just eval-navigation [POČET]` | interaktivní navigační zkoušky (vyžaduje přítomnost u robota) |
| `just eval-mapping` | efektivita mapování prostoru ze záznamů relací (bez hardwaru) |

### Testování

| Příkaz | Popis |
|---|---|
| `just test` | spustí sadu unit testů |
| `just coverage` | unit testy s reportem pokrytí jádra systému |

### Ostatní

| Příkaz | Popis |
|---|---|
| `just install` / `just install-resources` / `just install-all` | instalace Hailo SDK a modelů |
| `just clean` | smaže `__pycache__` a `.pytest_cache` |

## Struktura repozitáře

```
robot/
├── robot_server.py        Flask API + webové UI (port 5000)
├── inference_pipeline.py  čtení kamery + inference obou modelů na Hailo
├── vfh.py                 Vector Field Histogram
├── navigator.py           VFH + sonar → pohybové příkazy
├── polar_scan.py          360° polární sken, mapa volnosti prostoru
├── sonar_guard.py         bezpečnostní vlákno nad HC-SR04
├── state_machine.py       stavový automat autonomního hledání
├── omnibot.py             řízení motorů přes PCA9685/I2C
├── drive.py, distance.py, force_test.py   ruční diagnostiky
├── models/                zkompilované modely (.hef)
├── eval/                  nástroje pro sběr dat a vyhodnocení
└── tests/                 unit testy (pytest)

training/                  skripty pro trénování a kompilaci do HEF
hailo/                     upstream hailo-rpi5-examples (SDK, instalace)
setup_env.sh               aktivace prostředí — nutné spustit přes `source`
justfile                   definice všech příkazů výše
```

## Reprodukce výsledků z práce

**Kapitola 2.5.1 — přesnost detekce.** Vyžaduje Hailo, ale ne pohyb robota.
Testovací sadu z elektronické přílohy (`data/dataset/`) nejprve zkopírujte
do kořene repozitáře jako `dataset/`, poté:

```bash
just eval-detection dataset --exclude cup-disposable
just eval-detection-agnostic dataset --exclude cup-disposable
```

**Kapitola 2.5.2 — navigace.** Vyžaduje fyzicky přítomného robota a připravenou
místnost podle scénářů popsaných v příloze. Výsledky se průběžně zapisují
do `eval_navigation_results.csv`, podrobné záznamy relací do `search_logs/`
a anotovaná videa do `attempts/`:

```bash
just eval-navigation 5 --scan
```

**Kapitola 2.5.3 — mapování prostoru.** Odvozeno zpětnou analýzou záznamů
ve `search_logs/`, nevyžaduje žádný hardware. Zdrojová data jsou součástí
elektronické přílohy — po jejich zkopírování do `search_logs/`:

```bash
just eval-mapping
```

**Kapitola 2.5.4 — unit testy.** Nevyžaduje žádný hardware, veškerá hardwarová
rozhraní jsou v testech nahrazena mock objekty:

```bash
just test
just coverage
```

## Poznámky a řešení problémů

**`setup_env.sh` je nutné spustit přes `source`**, ne jako `./setup_env.sh` —
jinak se nastavení proměnných neprojeví v aktuálním terminálu.

**Kamera je namontovaná otočená o 180°.** Nástroje s náhledem (`just capture`,
`just review`) proto obraz při zobrazení automaticky otáčejí; volba `--no-flip`
to vypne. Ukládaná data se nikdy neotáčejí — zůstávají v souřadnicovém prostoru,
ve kterém byl model trénován.

**Grafická okna pod Waylandem.** `justfile` nastavuje `QT_QPA_PLATFORM=xcb`,
protože Qt backend OpenCV nenajde nativní Wayland plugin. Při spouštění skriptů
mimo `just` může být nutné tuto proměnnou nastavit ručně.

**Ukončování oken klávesou `q`, nikoli ESC.** V tomto prostředí může nově zaměřené
okno generovat trvalý falešný stisk ESC, který by okno okamžitě zavřel.
Skripty proto ESC záměrně nepoužívají — funguje `q` nebo tlačítko pro zavření okna.

**Sonar nevidí překážky s mezerou pod podstavou.** Ultrazvukový paprsek prochází
pod nábytkem s vysokou podnoží; jde o známé omezení popsané v kapitole 2.5.2.
Při testování s takovým nábytkem počítejte s ruční asistencí.
