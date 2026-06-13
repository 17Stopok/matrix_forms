import math
import numpy as np
from fractions import Fraction

# ─────────────────────────────────────────────
# POMOCNÉ FUNKCIE
# ─────────────────────────────────────────────

def input_matrix():
    print("\nZadajte maticu riadok po riadku.")
    print("Každý riadok zadajte ako čísla oddelené medzerou (napr: 1 2 0 5 3).")
    print("Prázdny riadok ukončí zadávanie.\n")
    rows = []
    while True:
        line = input(f"  Riadok {len(rows)+1}: ").strip()
        if line == "":
            if len(rows) == 0:
                print("  Matica musí mať aspoň jeden riadok.")
                continue
            break
        try:
            row = list(map(int, line.split()))
            if rows and len(row) != len(rows[0]):
                print(f"  Chyba: očakávam {len(rows[0])} čísel, dostal som {len(row)}.")
                continue
            rows.append(row)
        except ValueError:
            print("  Chyba: zadajte iba celé čísla oddelené medzerou.")
    n = len(rows)
    if n != len(rows[0]):
        print(f"\nUpozornenie: matica nie je štvorcová ({n}x{len(rows[0])}).")
        print("Niektoré formy vyžadujú štvorcovú maticu.\n")
    return rows


def print_matrix(M, label=""):
    if label:
        print(f"\n{label}")
    n = len(M)
    col_widths = []
    for j in range(len(M[0])):
        w = max(len(str(M[i][j])) for i in range(n))
        col_widths.append(w)
    for row in M:
        formatted = "  ".join(str(row[j]).rjust(col_widths[j]) for j in range(len(row)))
        print(f"  [ {formatted} ]")


# ─────────────────────────────────────────────
# 1. JORDANOVA NORMÁLNA FORMA
# ─────────────────────────────────────────────

def jordan_form(M):
    try:
        import numpy as np
        A = np.array(M, dtype=float)
        n = A.shape[0]
        
        if A.shape[0] != A.shape[1]:
            print("\n  Chyba: Jordanova forma vyžaduje štvorcovú maticu.")
            return

        print("\n  Počítam... (môže chvíľu trvať)")
        
        raw_eigvals = np.linalg.eigvals(A)
        
        def format_num(x):
            if not np.isreal(x) and abs(np.imag(x)) >= 1e-5:
                r = np.round(np.real(x), 3)
                i = np.round(np.imag(x), 3)
                return f"{r}+{i}j" if i >= 0 else f"{r}{i}j"
            val = np.real(x)
            return int(np.round(val)) if abs(val - np.round(val)) < 1e-5 else np.round(val, 4)

        print("\n  Vlastné čísla:")
        unique_eigs, counts = np.unique(np.round(raw_eigvals, 5), return_counts=True)
        for e, mult in zip(unique_eigs, counts):
            print(f"    λ = {format_num(e)},  algebraická kratnosť = {mult}")

        eigenvalues, eigenvectors = np.linalg.eig(A)
        
        J = np.zeros((n, n), dtype=complex)
        P = np.zeros((n, n), dtype=complex)
        
        for i in range(n):
            J[i, i] = eigenvalues[i]
            P[:, i] = eigenvectors[:, i]

        for i in range(n - 1):
            if np.isclose(J[i, i], J[i+1, i+1], atol=1e-5):
                dot_prod = np.abs(np.dot(P[:, i], P[:, i+1]))
                if dot_prod > 0.99:  
                    J[i, i+1] = 1.0
                    M_lam = A - J[i, i] * np.eye(n)
                    try:
                        v_gen = np.linalg.lstsq(M_lam, P[:, i], rcond=None)[0]
                        P[:, i+1] = v_gen
                    except:
                        pass

        J_num = [[format_num(J[i,j]) for j in range(n)] for i in range(n)]
        P_num = [[format_num(P[i,j]) for j in range(n)] for i in range(n)]

        print_matrix(J_num, label="Jordanova normálna forma J:")
        print_matrix(P_num, label="Matica prechodu P (také že A ≈ P * J * P⁻¹):")

    except ImportError:
        print("\n  Chyba: vyžaduje sa knižnica numpy (pip install numpy)")
    except Exception as e:
        print(f"\n  Chyba pri výpočte: {e}")
        
# ─────────────────────────────────────────────
# 2. SMITHOVA NORMÁLNA FORMA
# ─────────────────────────────────────────────

def smith_normal_form(M):
    rows = len(M)
    cols = len(M[0])
    A = [row[:] for row in M]

    def row_op(i1, i2, factor, add=True):
        for j in range(cols):
            if add:
                A[i1][j] += factor * A[i2][j]
            else:
                A[i1][j] = factor * A[i2][j]

    def col_op(j1, j2, factor):
        for i in range(rows):
            A[i][j1] += factor * A[i][j2]

    pivot = 0
    for step in range(min(rows, cols)):
        found = False
        for i in range(step, rows):
            for j in range(step, cols):
                if A[i][j] != 0:
                    A[step], A[i] = A[i], A[step]
                    for k in range(rows):
                        A[k][step], A[k][j] = A[k][j], A[k][step]
                    found = True
                    break
            if found:
                break
        if not found:
            break

        changed = True
        while changed:
            changed = False
            for i in range(step + 1, rows):
                if A[i][step] != 0:
                    q = A[i][step] // A[step][step]
                    for j in range(cols):
                        A[i][j] -= q * A[step][j]
                    if A[i][step] != 0:
                        A[step], A[i] = A[i], A[step]
                    changed = True
            for j in range(step + 1, cols):
                if A[step][j] != 0:
                    q = A[step][j] // A[step][step]
                    for i in range(rows):
                        A[i][j] -= q * A[i][step]
                    if A[step][j] != 0:
                        for k in range(rows):
                            A[k][step], A[k][j] = A[k][j], A[k][step]
                    changed = True

        if A[step][step] < 0:
            for j in range(cols):
                A[step][j] = -A[step][j]

    print("\n  Smithova normálna forma:")
    print_matrix(A)
    diag = [A[i][i] for i in range(min(rows, cols))]
    print(f"\n  Invariantné faktory (uhlopriečka): {diag}")


# ─────────────────────────────────────────────
# 3. LAPLACEOVA MATICA
# ─────────────────────────────────────────────

def laplace_matrix():
    print("\n  Laplaceova matica sa počíta z grafu.")
    print("  Zadajte spojenie hranej vo formate:  i j  (vrcholy číslované od 0)")
    print("  Prázdny riadok ukončí zadávanie.\n")

    edges = []
    max_node = 0
    while True:
        line = input("  Hrana: ").strip()
        if line == "":
            break
        try:
            u, v = map(int, line.split())
            edges.append((u, v))
            max_node = max(max_node, u, v)
        except ValueError:
            print("  Chyba: zadajte dva celé čísla oddelené medzerou.")

    n = max_node + 1
    L = [[0] * n for _ in range(n)]
    for u, v in edges:
        L[u][u] += 1
        L[v][v] += 1
        L[u][v] -= 1
        L[v][u] -= 1

    print_matrix(L, label="  Laplaceova matica L = D - M:")

    try:
        import sympy
        eigvals = sympy.Matrix(L).eigenvals()
        zero_count = int(eigvals.get(sympy.Integer(0), 0))
        print(f"\n  Počet súvislých komponentov grafu: {zero_count}")
        print(f"  (= počet nulových vlastných čísel Laplaceovej matice)")
    except ImportError:
        pass


# ─────────────────────────────────────────────
# 4. DIAGONALIZÁCIA
# ─────────────────────────────────────────────

def diagonalize(M):
    n = len(M)
    A = np.array(M, dtype=float)

    def format_num(x):
        if not np.isreal(x) and abs(np.imag(x)) >= 1e-9:
            r = np.round(np.real(x), 3)
            i = np.round(np.imag(x), 3)
            return f"{r}+{i}j" if i >= 0 else f"{r}{i}j"
        
        val = np.real(x)
        return int(np.round(val)) if abs(val - np.round(val)) < 1e-9 else np.round(val, 4)

    try:
        eigenvalues, eigenvectors = np.linalg.eig(A)

        if np.linalg.matrix_rank(eigenvectors) < n:
            print("\n  Matica NIE JE diagonalizovateľná.")
            print("  Použite možnosť 1 pre Jordanovu normálnu formu.")
            return

        print("\n  Vlastné čísla:")
        unique, counts = np.unique(np.round(eigenvalues, 6), return_counts=True)
        for e, mult in zip(unique, counts):
            print(f"    λ = {format_num(e)},  kratnosť = {mult}")

        D_fmt = [[format_num(eigenvalues[j] if i == j else 0) for j in range(n)] for i in range(n)]
        P_fmt = [[format_num(eigenvectors[i, j]) for j in range(n)] for i in range(n)]

        print("\n  Matica je diagonalizovateľná.")
        print_matrix(D_fmt, label="Diagonálna forma D:")
        print_matrix(P_fmt, label="Matica prechodu P (také že A = P * D * P⁻¹):")

    except np.linalg.LinAlgError:
        print("\n  Chyba výpočtu: nepodarilo sa nájsť vlastné čísla.")

# ─────────────────────────────────────────────
# HLAVNÉ MENU
# ─────────────────────────────────────────────

def menu():
    print("=" * 50)
    print("   ŠPECIÁLNE TVARY MATÍC — interaktívny program")
    print("=" * 50)

    while True:
        print("\nČo chcete vypočítať?")
        print("  1  Jordanova normálna forma")
        print("  2  Smithova normálna forma")
        print("  3  Laplaceova matica (zo zadaného grafu)")
        print("  4  Diagonalizácia")
        print("  0  Koniec")

        choice = input("\nVáš výber: ").strip()

        if choice == "0":
            print("\nDovidenia!\n")
            break
        elif choice == "1":
            M = input_matrix()
            jordan_form(M)
        elif choice == "2":
            M = input_matrix()
            smith_normal_form(M)
        elif choice == "3":
            laplace_matrix()
        elif choice == "4":
            M = input_matrix()
            diagonalize(M)
        else:
            print("  Neplatná voľba, skúste znova.")

        input("\n  Stlačte Enter pre pokračovanie...")


if __name__ == "__main__":
    menu()
