"""
Test pro Differential Evolution (DE/rand/1/bin) s vizualizací.
"""
import sys
import os

# Nastavení cesty pro import z kořene projektu
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import matplotlib
matplotlib.use('TkAgg')  # Explicitní backend pro zobrazení oken

from main import differential_evolution, ackley, sphere, rastrigin, get_default_bounds


def test_de_ackley_with_visualization():
    """
    Spustí Differential Evolution na Ackley funkci s heatmap vizualizací.
    """
    print("=" * 60)
    print("DIFFERENTIAL EVOLUTION - Ackley funkce (2D)")
    print("=" * 60)
    
    # Použijeme užší bounds pro lepší vizualizaci
    bounds_viz = [(-10.0, 10.0), (-10.0, 10.0)]
    
    best_x, best_f = differential_evolution(
        objective=ackley,
        bounds=bounds_viz,
        NP=20,
        F=0.5,
        CR=0.5,
        max_gen=50,
        seed=42,
        visualize=True,
        num_points=200,
    )
    
    print(f"\nVýsledek:")
    print(f"  Nejlepší bod: x = {best_x}")
    print(f"  Hodnota funkce: f(x) = {best_f:.8f}")
    print(f"  Globální optimum: (0, 0), f = 0")
    print()


def test_de_sphere():
    """
    Rychlý test DE na Sphere bez vizualizace.
    """
    print("=" * 60)
    print("DIFFERENTIAL EVOLUTION - Sphere funkce (test)")
    print("=" * 60)
    
    bounds = [(-5.0, 5.0), (-5.0, 5.0)]
    
    best_x, best_f = differential_evolution(
        objective=sphere,
        bounds=bounds,
        NP=20,
        F=0.5,
        CR=0.5,
        max_gen=50,
        seed=123,
        visualize=False,
    )
    
    print(f"Výsledek: x = {best_x}, f(x) = {best_f:.8f}")
    
    # Kontrola: pro Sphere by mělo být blízko nuly
    assert best_f < 0.01, f"DE na Sphere selhalo: f = {best_f} > 0.01"
    print("✓ Test prošel!")
    print()


def test_de_rastrigin_with_viz():
    """
    DE na Rastrigin s vizualizací.
    """
    print("=" * 60)
    print("DIFFERENTIAL EVOLUTION - Rastrigin funkce (2D)")
    print("=" * 60)
    
    bounds = [(-5.12, 5.12), (-5.12, 5.12)]
    
    best_x, best_f = differential_evolution(
        objective=rastrigin,
        bounds=bounds,
        NP=25,
        F=0.8,
        CR=0.7,
        max_gen=100,
        seed=999,
        visualize=True,
        num_points=180,
    )
    
    print(f"\nVýsledek:")
    print(f"  Nejlepší bod: x = {best_x}")
    print(f"  Hodnota funkce: f(x) = {best_f:.8f}")
    print(f"  Globální optimum: (0, 0), f = 0")
    print()


def test_de_rosenbrock_with_viz():
    """
    DE na Rosenbrock s vizualizací.
    """
    from main import rosenbrock
    
    print("=" * 60)
    print("DIFFERENTIAL EVOLUTION - Rosenbrock funkce (2D)")
    print("=" * 60)
    
    bounds = [(-5.0, 10.0), (-5.0, 10.0)]
    
    best_x, best_f = differential_evolution(
        objective=rosenbrock,
        bounds=bounds,
        NP=30,
        F=0.7,
        CR=0.8,
        max_gen=150,
        seed=555,
        visualize=True,
        num_points=200,
    )
    
    print(f"\nVýsledek:")
    print(f"  Nejlepší bod: x = {best_x}")
    print(f"  Hodnota funkce: f(x) = {best_f:.8f}")
    print(f"  Globální optimum: (1, 1), f = 0")
    print()


def test_de_comparison():
    """
    Porovnání DE s různými parametry na Sphere funkci.
    """
    print("=" * 60)
    print("DIFFERENTIAL EVOLUTION - Porovnání parametrů")
    print("=" * 60)
    
    bounds = [(-5.0, 5.0), (-5.0, 5.0)]
    configs = [
        {"NP": 10, "F": 0.5, "CR": 0.5, "max_gen": 30},
        {"NP": 20, "F": 0.5, "CR": 0.5, "max_gen": 30},
        {"NP": 20, "F": 0.8, "CR": 0.9, "max_gen": 30},
    ]
    
    for i, config in enumerate(configs, 1):
        print(f"\nKonfigurace {i}: NP={config['NP']}, F={config['F']}, CR={config['CR']}, gen={config['max_gen']}")
        
        best_x, best_f = differential_evolution(
            objective=sphere,
            bounds=bounds,
            seed=42,
            visualize=False,
            **config
        )
        
        print(f"  Výsledek: x = [{best_x[0]:.6f}, {best_x[1]:.6f}], f(x) = {best_f:.8f}")
    
    print("\n✓ Porovnání dokončeno!")
    print()


if __name__ == "__main__":
    # Hlavní vizualizace na Ackley
    test_de_rastrigin_with_viz()
    
    # Rychlý test bez vizualizace
    test_de_sphere()
    
    # Další vizualizace na Rastrigin (volitelně - odkomentujte pro spuštění)
    # test_de_rastrigin_with_viz()
    
    # Vizualizace na Rosenbrock (volitelně - odkomentujte pro spuštění)
    # test_de_rosenbrock_with_viz()
    
    # Porovnání různých konfigurací
    test_de_comparison()
    
    print("=" * 60)
    print("VŠECHNY TESTY DOKONČENY")
    print("=" * 60)
