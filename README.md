# BIA – Cvičení 5 (Biologicky inspirované algoritmy)

Tento repozitář obsahuje řešení 5. cvičení z předmětu Biologicky inspirované algoritmy.
Cílem je implementace algoritmu Differential Evolution (DE/rand/1/bin) a vygenerování figury s heatmapou výsledné funkce.

https://michaelmachu.eu/data/pdf/bia/Exercise5.pdf

## Obsah

- `main.py` – modulové implementace funkcí a algoritmů:
  - **Testovací funkce (minimizační)**: Sphere, Schwefel, Rosenbrock, Rastrigin, Griewank, Levy, Michalewicz, Zakharov, Ackley
  - **Differential Evolution**: `differential_evolution(objective, bounds, NP=20, F=0.5, CR=0.5, max_gen=50, seed=None, visualize=False, num_points=200)`
- `tests/` – testovací skripty pro oba algoritmy s vizualizací

## Jak spustit testy (Windows / PowerShell)

```powershell
# Differential Evolution
python tests\differential_evolution_test.py
```

## Differential Evolution – stručně

- **Populační algoritmus** pracující s více řešeními současně (typicky NP = 20 jedinců).
- Začíná s náhodnou populací uniformně rozloženou v zadaných mezích.
- V každé generaci pro každého jedince:
  1. **Mutace**: Vybere 3 náhodné různé jedince a vytvoří mutační vektor `v = x_r3 + F * (x_r1 - x_r2)`
  2. **Křížení**: Kombinuje mutační vektor s cílovým jedincem podle pravděpodobnosti `CR`
  3. **Selekce**: Pokud je potomek (trial vektor) lepší nebo roven, nahradí původního jedince
- Algoritmus končí po `max_gen` generacích.
- Celá populace konverguje směrem k optimu díky sdílení informací mezi jedinci.
- Výhoda: Dobrá schopnost globálního prohledávání a escapování z lokálních minim.

## Parametry Differential Evolution

- **NP** (20) – Velikost populace (počet jedinců)
- **F** (0.5) – Mutační konstanta / scaling factor (typicky 0.4–1.0)
- **CR** (0.5) – Pravděpodobnost crossoveru (0–1)
- **max_gen** (50) – Maximální počet generací

## Poznámky

- Všechny implementované funkce jsou chápány jako minimalizační (menší je lepší). Pro maximalizaci lze předat `objective=lambda x: -g(x)`.
- Vizualizace je k dispozici pouze pro 2D funkce (2 parametry).
- Oba algoritmy umožňují nastavení `seed` pro reprodukovatelnost výsledků.
