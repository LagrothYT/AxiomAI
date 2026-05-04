import os
import sys


if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass


def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')


def spaced_title(text):
    return " ".join(text.upper())


def print_run_banner(title, rows, width=54):
    print("╔" + "═" * width + "╗")
    print("║" + spaced_title(title).center(width) + "║")
    print("╚" + "═" * width + "╝")
    print()
    for label, value in rows:
        print(f"  {label:<12} {value}")
    print("─" * (width + 2))


def print_step(label, value=None):
    if value is None:
        print(f"  {label}")
    else:
        print(f"  {label:<12} {value}")


def print_done(rows, width=54):
    print("─" * (width + 2))
    for label, value in rows:
        print(f"  {label:<12} {value}")
