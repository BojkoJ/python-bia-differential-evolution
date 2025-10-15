import numpy as np
import math
from typing import Callable, List, Tuple, Optional

# Výchozí hranice (výřezy) pro 2D vizualizaci (X,Y).
DEFAULT_BOUNDS_2D = {
    "sphere": (-5.0, 5.0),          
    "schwefel": (-500.0, 500.0),    
    "rosenbrock": (-5.0, 10.0),   
    "rastrigin": (-5.12, 5.12),
    "griewank": (-6.0, 6.0),
    "levy": (-3.0, 3.0),
    "michalewicz": (0.0, float(math.pi)), 
    "zakharov": (-5.0, 5.0),
    "ackley": (-32.768, 32.768),
}

def get_default_bounds(func_name: str, dim: int = 2) -> List[Tuple[float, float]]:
        """
        Vrátí seznam hranic pro každou dimenzi.

        Parametry:
            func_name : název funkce
            dim       : kolik dimenzí (např. 2 > chceme dvě dvojice hranic).

        Návrat:
            list dvojic (low, high). Každá dvojice jsou float hodnoty dolní a horní meze.
        """
        # Převedeme název na malá písmena
        name_lower = func_name.lower()

        # Zkusíme v tabulce DEFAULT_BOUNDS_2D najít položku podle klíče name_lower, pokud není dáme default
        bounds_pair = DEFAULT_BOUNDS_2D.get(name_lower, (-5.0, 5.0))

        # Rozbalíme dvojici (low, high) do dvou proměnných pro přehlednost.
        # Příklad: bounds_pair = (-5.0, 5.0) -> low = -5.0, high = 5.0
        low = bounds_pair[0]
        high = bounds_pair[1]

        result: List[Tuple[float, float]] = []

        for i in range(dim):
                result.append((low, high))

        return result


def generate_grid(bounds_2d: List[Tuple[float, float]], num_points: int = 100):
    """
    Vytvoří mřížku (X, Y) v zadaných 2D hranicích. Každá z os má `num_points` vzorků.
    """
    (x_min, x_max), (y_min, y_max) = bounds_2d
    
    # np.linspace vytvoří jednorozměrné pole s rovnoměrně rozloženými body
    # mezi zadanými hranicemi. Např. linspace(0, 10, 5) → [0, 2.5, 5, 7.5, 10]
    x = np.linspace(x_min, x_max, num_points) # Vytvoří num_points bodů na x-ose
    y = np.linspace(y_min, y_max, num_points) # Vytvoří num_points bodů na y-ose
    
    # np.meshgrid vezme dva 1D vektory a vytvoří z nich 2D mřížku souřadnic
    # X obsahuje x-souřadnice pro každý bod mřížky
    # Y obsahuje y-souřadnice pro každý bod mřížky
    # Výsledek: každý bod [X[i,j], Y[i,j]] reprezentuje jeden bod v 2D mřížce
    X, Y = np.meshgrid(x, y)
    return X, Y

def evaluate_surface_2d(objective: Callable[[List[float]], float],
                            bounds_2d: List[Tuple[float, float]],
                            num_points: int = 100):
        """
        Vyhodnotí funkci na 2D mřížce: funkce dostane 2 parametry (x, y) a vrátí třetí (z).
        Používáme numpy.vectorize pro čitelnost.
        """
        # Vytvoříme 2D mřížku bodů (X, Y) v zadaných hranicích
        X, Y = generate_grid(bounds_2d, num_points)
        
        # np.vectorize umožňuje aplikovat obyčejnou funkci na numpy pole element po elementu
        # lambda x, y: objective([float(x), float(y)]) převede každou dvojici (x,y) z mřížky
        # na seznam [x, y] a předá ho cílové funkci
        f_vec = np.vectorize(lambda x, y: objective([float(x), float(y)]))
        
        # Aplikujeme vektorizovanou funkci na celé pole X a Y najednou
        # Výsledek Z obsahuje hodnotu funkce pro každý bod mřížky
        # Z[i,j] = objective([X[i,j], Y[i,j]])
        Z = f_vec(X, Y)
        
        return X, Y, Z


def plot_surface_2d(objective: Callable[[List[float]], float],
                    bounds_2d: Optional[List[Tuple[float, float]]] = None,
                    num_points: int = 100,
                    title: Optional[str] = None):
    """
    Jednoduchá 3D vizualizace povrchu funkce (2 vstupy -> 3D graf X,Y,Z).
    Importujeme matplotlib až zde, aby to nebylo povinné při běžných výpočtech.
    """
    if bounds_2d is None:                               # Pokud uživatel nezadal vlastní hranice tak default
        bounds_2d = get_default_bounds("sphere", 2)

    # Vyhodnotíme funkci na pravidelné mřížce bodů (dostaneme X, Y souřadnice a Z hodnoty funkce)
    X, Y, Z = evaluate_surface_2d(objective, bounds_2d, num_points)

    import matplotlib.pyplot as plt                      # Import knihovny pro vykreslování

    fig = plt.figure(figsize=(8, 6))                     # Vytvoří novou figuru s danou velikostí
    ax = fig.add_subplot(111, projection='3d')           # Přidá 3D subplot

    # Samotné vykreslení 3D povrchu (surface). Jednotlivé parametry určují barvy a mřížku.
    ax.plot_surface(
        X,                                             # 2D pole x souřadnic
        Y,                                             # 2D pole y souřadnic
        Z,                                             # 2D pole hodnot funkce f(x,y)
        cmap='jet',                                    # Barevná mapa
        edgecolor='k',                                 # Černé hrany každého malého polygonu
        linewidth=0.2,                                 # Tloušťka čar hran
        antialiased=True,                              # Vyhlazení hran pro hezčí vzhled
        rstride=1,                                     # Vykreslit každou řádku mřížky
        cstride=1,                                     # Vykreslit každý sloupec mřížky
    )
    
    ax.set_xlabel('x1')                                 # Popisek osy X
    ax.set_ylabel('x2')                                 # Popisek osy Y
    ax.set_zlabel('f(x)')                               # Popisek osy Z
    
    if title:                                           # Pokud je předán titulek nastavíme ho nad graf
        ax.set_title(title)                             

    plt.tight_layout()                                  # Úprava rozložení (aby se popisky nepřekrývaly)
    plt.show()                                          # Zobrazení okna s grafem

# Jednotlivé testovací funkce:

def sphere(params):
    """
    Definice:
        f(x) = sum(x_i^2) pro i = 1..n
    
    - n je rozměr (počet prvků vektoru x)
    - Obvyklá doména: x_i v intervalu [-5.12, 5.12] pro všechna i = 1, ..., d
    - Globální minimum: x* = (0, 0, ..., 0) s hodnotou f(x*) = 0

    Parametry
    ---------
    params : seznam nebo jiné pole čísel
        Vstupní vektor x.

    Návratová hodnota
    -----------------
    float
        Hodnota funkce Sphere pro zadané params
    """
    total = 0.0
    for value in params:
        total += value * value
    return float(total)

def schwefel(params):
    """
    Definice:
        f(x) = 418.9829 * n - sum_{i=1..n}  x_i * sin(sqrt(|x_i|))]

    - n je rozměr (počet prvků vektoru x)
    - Obvyklá doména: x_i v intervalu [-500, 500]
    - Globální minimum: x_i ≈ 420.968746... pro všechny i, hodnota f(x*) ≈ 0

    Parametry
    ---------
    params : seznam nebo jiné pole čísel
        Vstupní vektor x.

    Návratová hodnota
    -----------------
    float
        Hodnota funkce Schwefel pro zadané params
    """
    n = 0
    suma = 0.0
    for value in params:
        n += 1
        # Použijeme absolutní hodnotu uvnitř odmocniny podle definice.
        term = value * math.sin(math.sqrt(abs(value)))
        suma += term

    konst = 418.9829
    result = konst * n - suma
    return float(result)


def rosenbrock(params):
    """
    Definice:
        f(x) = sum_{i=1..n-1} [ 100 * (x_{i+1} - x_i^2)^2 + (1 - x_i)^2 ]

    - n je rozměr vektoru x (pro n < 2 je součet prázdný -> 0)
    - Obvyklá doména: x_i v intervalu přibližně [-2.5, 2.5] (často uváděno), někdy [-5, 10]
    - Globální minimum: x* = (1, 1, ..., 1) s hodnotou f(x*) = 0

    Parametry
    ---------
    params : seznam nebo jiné pole čísel
        Vstupní vektor x.

    Návratová hodnota
    -----------------
    float
        Hodnota funkce Rosenbrock pro zadané params
    """
    total = 0.0
    # Součet jde od i=0 do i=n-2 (tj. pracujeme vždy s dvojicí x_i a x_{i+1})
    n = len(params)
    for i in range(0, n - 1):
        xi = params[i]
        x_next = params[i + 1]
        # 100 * (x_{i+1} - x_i^2)^2
        first = 100.0 * (x_next - (xi * xi)) ** 2
        # (1 - x_i)^2
        second = (1.0 - xi) ** 2
        total += first + second

    return float(total)

def rastrigin(params):
    """
    Definice pro n-rozměrný vektor x:
        f(x) = 10 * n + sum_{i=1..n} [ x_i^2 - 10 * cos(2π x_i) ]

    - Obvyklá doména: x_i v intervalu [-5.12, 5.12]
    - Globální minimum: x* = (0, ..., 0) s hodnotou f(x*) = 0

    Parametry
    ---------
    params : seznam nebo jiné pole čísel
        Vstupní vektor x.

    Návratová hodnota
    -----------------
    float
        Hodnota funkce Rastrigin pro zadané params
    """
    n = 0
    total = 0.0
    for value in params:
        n += 1
        total += (value * value) - 10.0 * math.cos(2.0 * math.pi * value)

    result = 10.0 * n + total
    return float(result)


def griewank(params):
    r"""
    Definice pro n-rozměrný vektor x:
        f(x) = 1 + \sum_{i=1..n} (x_i^2 / 4000) - \prod_{i=1..n} cos\left(\frac{x_i}{\sqrt{i}}\right)

    - Obvyklá doména: x_i v intervalu [-600, 600]
    - Globální minimum: x* = (0, ..., 0) s hodnotou f(x*) = 0

    Parametry
    ---------
    params : seznam nebo jiné pole čísel
        Vstupní vektor x.

    Návratová hodnota
    -----------------
    float
        Hodnota funkce Griewank pro zadané `params`.
    """
    sum_term = 0.0
    prod_term = 1.0

    # i číslujeme od 1 kvůli definici s odmocninou i
    i = 1
    for value in params:
        sum_term += (value * value) / 4000.0
        prod_term *= math.cos(value / math.sqrt(i))
        i += 1

    result = 1.0 + sum_term - prod_term
    return float(result)


def levy(params):
    r"""
    Standardní (vícerozměrná) Levy funkce.

    Definice:
        Nejprve se provede transformace
            w_i = 1 + (x_i - 1) / 4

        f(x) = sin^2(π w_1)
               + Σ_{i=1..n-1} (w_i - 1)^2 * [ 1 + 10 * sin^2(π w_i + 1) ]
               + (w_n - 1)^2 * [ 1 + sin^2(2π w_n) ]

    Vlastnosti:
        - Doména obvykle x_i ∈ [-10, 10]
        - Globální minimum: x* = (1, ..., 1) → f(x*) = 0

    Parametry
    ---------
    params : list[float]
        Vstupní vektor x.

    Návratová hodnota
    -----------------
    float
        Hodnota Levy funkce pro zadané `params`.
    """
    n = len(params)
    if n == 0:
        return 0.0

    # Transformace w_i
    w = []
    for x in params:
        w.append(1.0 + (x - 1.0) / 4.0)

    # První člen
    total = math.sin(math.pi * w[0]) ** 2

    # Prostřední součet (i = 1..n-1 => indexy 0..n-2)
    for i in range(0, n - 1):
        wi = w[i]
        total += (wi - 1.0) * (wi - 1.0) * (1.0 + 10.0 * (math.sin(math.pi * wi + 1.0) ** 2))

    # Poslední člen
    wn = w[-1]
    total += (wn - 1.0) * (wn - 1.0) * (1.0 + (math.sin(2.0 * math.pi * wn) ** 2))

    return float(total)


def michalewicz(params):
    r"""
    Definice pro n-rozměrný vektor x (obvykle s parametrem m = 10):
        f(x) = - \sum_{i=1..n} [ sin(x_i) * ( sin( i * x_i^2 / π ) )^{2m} ]

    - Obvyklá doména: x_i v intervalu [0, π]
    - Typické nastavení: m = 10 (čím větší m, tím více lokálních minim)
    - Globální minimum pro n=2, m=10 je přibližně f(x*) ≈ -1.8013 v bodě x* ≈ (2.20, 1.57)

    Parametry
    ---------
    params : seznam nebo jiné pole čísel
        Vstupní vektor x.

    Návratová hodnota
    -----------------
    float
        Hodnota funkce Michalewicz pro zadané `params` (při m = 10).
    """
    m = 10
    total = 0.0
    i = 1
    for x in params:
        s1 = math.sin(x)
        s2 = math.sin(i * (x * x) / math.pi)
        term = s1 * (s2 ** (2 * m))
        total += term
        i += 1
    return float(-total)


def zakharov(params):
    r"""
    Funkce Zakharov (minimalizační úloha).

    Definice pro n-rozměrný vektor x:
        f(x) = \sum_{i=1..n} x_i^2
               + (\sum_{i=1..n} 0.5 * i * x_i)^2
               + (\sum_{i=1..n} 0.5 * i * x_i)^4

    - Obvyklá doména: x_i v intervalu [-5, 10] (různé zdroje uvádí mírně odlišně)
    - Globální minimum: x* = (0, ..., 0) s hodnotou f(x*) = 0

    Parametry
    ---------
    params : seznam nebo jiné pole čísel
        Vstupní vektor x.

    Návratová hodnota
    -----------------
    float
        Hodnota funkce Zakharov pro zadané `params`.
    """
    sum_sq = 0.0
    sum_lin = 0.0
    i = 1
    for x in params:
        sum_sq += x * x
        sum_lin += 0.5 * i * x
        i += 1

    result = sum_sq + (sum_lin ** 2) + (sum_lin ** 4)
    return float(result)


def ackley(params):
    """
    Funkce Ackley (minimalizační úloha).

    Pro n-rozměrný vektor x a konstanty a=20, b=0.2, c=2π:
        f(x) = -a * exp(-b * sqrt( (1/n) * sum(x_i^2) ))
               - exp( (1/n) * sum( cos(c * x_i) ) )
               + a + e

    - Obvyklá doména: x_i v intervalu [-32.768, 32.768]
    - Globální minimum: x* = (0, ..., 0) s hodnotou f(x*) = 0

    Parametry
    ---------
    params : seznam nebo jiné pole čísel
        Vstupní vektor x.

    Návratová hodnota
    -----------------
    float
        Hodnota funkce Ackley pro zadané `params`.
    """
    n = 0
    sum_sq = 0.0
    sum_cos = 0.0
    for x in params:
        n += 1
        sum_sq += x * x
        sum_cos += math.cos(2.0 * math.pi * x)

    if n == 0:
        return 0.0

    a = 20.0
    b = 0.2
    # c = 2*pi je použito přímo v cyklu výše

    term1 = -a * math.exp(-b * math.sqrt((1.0 / n) * sum_sq))
    term2 = -math.exp((1.0 / n) * sum_cos)
    result = term1 + term2 + a + math.e
    return float(result)


def differential_evolution(
    objective: Callable[[List[float]], float],
    bounds: List[Tuple[float, float]],
    NP: int = 20,
    F: float = 0.5,
    CR: float = 0.5,
    max_gen: int = 50,
    seed: Optional[int] = None,
    visualize: bool = False,
    num_points: int = 200,
) -> Tuple[List[float], float]:
    """
    Differential Evolution (DE/rand/1/bin) - evolučníalgoritmus pro minimalizaci.

    Algoritmus:
      1. Vygenerujeme populaci NP jedinců (náhodné vektory v daných mezích)
      2. Pro každou generaci (až do max_gen):
         - Pro každý jedinec (tzv. target vektor x_i):
           a) Vybereme 3 náhodné různé jedince: x_r1, x_r2, x_r3
           b) Vytvoříme mutační vektor: v = x_r3 + F * (x_r1 - x_r2)
           c) Křížením vytvoříme trial vektor u (kombinace v a x_i podle CR)
           d) Pokud je u lepší nebo rovno x_i, nahradíme x_i vektorem u
      3. Vrátíme nejlepšího jedince z finální populace

    Vizualizace (jen 2D):
      - Heatmapa funkce na pozadí
      - Cesta nejlepšího jedince v každé generaci (bílá čára)
      - Start (modrý bod), konec (zelená hvězda)

    Parametry
    ---------
    objective : Callable
        Cílová funkce k minimalizaci.
    bounds : List[Tuple[float, float]]
        Meze pro každou dimenzi.
    NP : int
        Velikost populace (počet jedinců).
    F : float
        Mutační konstanta (scaling factor), typicky 0.4 - 1.0.
    CR : float
        Pravděpodobnost crossoveru, rozsah [0, 1].
    max_gen : int
        Maximální počet generací.
    seed : Optional[int]
        Seed pro reprodukovatelnost.
    visualize : bool
        Má-li se vytvořit heatmap vizualizace (jen pro 2D).
    num_points : int
        Hustota mřížky pro heatmapu.

    Návrat
    ------
    (best_x, best_f) : Tuple[List[float], float]
        Nejlepší nalezené řešení a jeho hodnota.
    """
    import random
    import copy
    
    rng = random.Random(seed)
    dim = len(bounds)

    # Pomocná třída pro reprezentaci jedince (řešení)
    class Individual:
        def __init__(self, params: List[float], f_value: float):
            self.params = params  # vektor parametrů
            self.f = f_value      # hodnota fitness (cílové funkce)

    # Pomocná funkce: clamp hodnoty do mezí
    def clamp(value, low, high):
        if value < low:
            return low
        if value > high:
            return high
        return value

    # 1) Inicializace populace
    # Vygenerujeme NP náhodných jedinců v zadaných mezích
    population: List[Individual] = []
    first_individual_params = None  # Uložíme prvního jedince pro vizualizaci startu
    
    for idx in range(NP):
        params = []
        for low, high in bounds:
            value = rng.uniform(low, high)
            params.append(value)
        f_value = objective(params)
        population.append(Individual(params, f_value))
        
        # Uložíme prvního jedince jako reprezentanta počátečního stavu
        if idx == 0:
            first_individual_params = list(params)

    # Najdeme nejlepšího jedince v počáteční populaci
    best_idx = 0
    for i in range(1, NP):
        if population[i].f < population[best_idx].f:
            best_idx = i
    
    best_x = list(population[best_idx].params)
    best_f = float(population[best_idx].f)

    # Pro vizualizaci: ukládáme cestu nejlepšího jedince v každé generaci
    # Začínáme s prvním vygenerovaným jedincem (ne nejlepším), aby bylo vidět skutečný start
    path: List[Tuple[List[float], float]] = [(first_individual_params, objective(first_individual_params))]
    # Přidáme i nejlepšího z počáteční populace
    path.append((list(best_x), best_f))

    # 2) Hlavní smyčka - evoluce přes generace
    for g in range(max_gen):
        # Vytvoříme novou populaci (deep copy staré)
        new_population = copy.deepcopy(population)

        # Pro každého jedince v populaci
        for i in range(NP):
            # Cílový vektor (target vector)
            x_i = population[i]

            # a) Vybereme 3 náhodné různé indexy r1, r2, r3
            # Musí platit: r1 != r2 != r3 != i
            indices = list(range(NP))
            indices.remove(i)  # odstraníme index aktuálního jedince
            
            # Vybereme 3 různé indexy
            r1, r2, r3 = rng.sample(indices, 3)

            # b) Mutace: v = x_r3 + F * (x_r1 - x_r2)
            v = []
            for j in range(dim):
                low, high = bounds[j]
                # Mutační vektor podle vzorce
                v_j = population[r3].params[j] + F * (population[r1].params[j] - population[r2].params[j])
                # Ošetření hranic
                v_j = clamp(v_j, low, high)
                v.append(v_j)

            # c) Křížení (crossover): vytvoření trial vektoru u
            u = [0.0] * dim
            # j_rnd zajistí, že alespoň jeden parametr bude z mutačního vektoru
            # a to tak, že j_rnd je jeden z indexů dimenzí
            j_rnd = rng.randint(0, dim - 1)

            # CR je pravděpodobnost, že vezmeme parametr z mutačního vektoru
            # pro každou dimenzi rozhodneme podle CR
            for j in range(dim):
                # pokud je náhodné číslo menší než CR nebo je to j_rnd, vezmeme z mutačního vektoru v
                if rng.random() < CR or j == j_rnd:
                    # Vezmeme parametr z mutačního vektoru
                    u[j] = v[j]
                else:
                    # Vezmeme parametr z původního jedince
                    u[j] = x_i.params[j]

            # d) Vyhodnocení trial vektoru
            f_u = objective(u)

            # Selekce: pokud je trial vektor lepší nebo roven, nahradíme jím cílový vektor
            if f_u <= x_i.f:
                new_population[i] = Individual(list(u), f_u)
                
                # Aktualizace globálního nejlepšího
                if f_u < best_f:
                    best_x = list(u)
                    best_f = float(f_u)

        # Nahradíme starou populaci novou
        population = new_population
        
        # Uložíme nejlepšího jedince této generace pro vizualizaci
        path.append((list(best_x), best_f))

    # 3) Vizualizace (heatmapa pro 2D)
    if visualize and dim == 2:
        import matplotlib.pyplot as plt

        # Vytvoříme mřížku a vyhodnotíme funkci
        bounds_2d = [(bounds[0][0], bounds[0][1]), (bounds[1][0], bounds[1][1])]
        X, Y, Z = evaluate_surface_2d(objective, bounds_2d, num_points=num_points)

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111)

        # Heatmapa: vyplněné kontury
        levels = 50
        contour = ax.contourf(X, Y, Z, levels=levels, cmap='jet')
        
        # Přidáme colorbar
        cbar = fig.colorbar(contour, ax=ax)
        cbar.set_label('f(x)', rotation=270, labelpad=20)

        # Volitelně: přidáme contour čáry (izolinie)
        ax.contour(X, Y, Z, levels=20, colors='black', alpha=0.2, linewidths=0.5)

        ax.set_xlabel('x1')
        ax.set_ylabel('x2')
        ax.set_title('Differential Evolution (DE/rand/1/bin) – Heatmap')

        # Vykreslíme cestu nejlepšího jedince
        if len(path) > 1:
            path_x = [state[0][0] for state in path]
            path_y = [state[0][1] for state in path]
            
            # Cesta jako čára
            ax.plot(path_x, path_y, 'w-', linewidth=1.5, alpha=0.7, label='DE path')
            
            # Start bod (modrý)
            ax.scatter([path_x[0]], [path_y[0]], s=200, c='blue', 
                      edgecolor='white', linewidth=2, marker='o', 
                      label='Start', zorder=5)
            
            # Finální bod (zelený)
            ax.scatter([best_x[0]], [best_x[1]], s=200, c='lime', 
                      edgecolor='white', linewidth=2, marker='*', 
                      label='Best', zorder=5)

        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)

        msg = f"DE: best=({best_x[0]:.4f}, {best_x[1]:.4f}), f={best_f:.6g}, gen={max_gen}, NP={NP}"
        print(msg)
        ax.text(0.02, 0.98, msg, transform=ax.transAxes, 
               fontsize=10, verticalalignment='top',
               bbox=dict(facecolor='white', alpha=0.8, pad=5))

        plt.tight_layout()
        plt.show()

    return best_x, best_f