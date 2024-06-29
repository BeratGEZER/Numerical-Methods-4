import numpy as np
from scipy.linalg import cholesky
from scipy.linalg import lu, solve
from tabulate import tabulate


# Coefficient matrix A
A = np.array([[2, -1, -1, 1],
              [2, 2, 1, 0],
              [2, -1, 2, 0],
              [2, -1, 2, 2]])

# Right-hand side vector B
B = np.array([5, 4, 8, 10])

# Initial guess
initial_guess = np.array([0, 0, 0, 0])

# Maximum number of iterations
max_iterations = 4

# List to store values for each iteration
iteration_values = [0,0,0,0]
i=0
# Gauss-Seidel method
for iteration in range(1, max_iterations + 1):
    
    i+=1
    # Update all variables simultaneously using vectorized operations
    iteration_values[0] = (B[0] + iteration_values[1]+iteration_values[2]-iteration_values[3])/2
    iteration_values[1] = (B[1] -2*iteration_values[0]-iteration_values[2])/2
    iteration_values[2] = (B[2] + iteration_values[1]- 2*iteration_values[0])/2
    iteration_values[3] = (B[3] - 2*iteration_values[2]+iteration_values[1]-2*iteration_values[0])/2

    print(f"iteration step {i} | x1= {iteration_values[0]},x2={-1*iteration_values[2]},x3={-1*iteration_values[1]},x4={iteration_values[3]}")

    


import numpy as np

# Katsayı matrisi (sol tarafındaki katsayılar)
coefficients = np.array([[2, -1, 2, 0],
                        [2, -1, -1, 1],
                        [2, 2, 1, 0],
                        [2, -1, 2, 2]])

# Doğru terim vektörü (sağ tarafındaki sayılar)
constants = np.array([8, 5, 4, 10])

# Denklem sistemi çözümü
solution = np.linalg.solve(coefficients, constants)

# Çözümü ekrana yazdırma
print("Solution using solve mathod:")
for i in range(len(solution)):
    print(f"x{i+1} = {solution[i]}")

import numpy as np

# Coefficient matrix A
coefficients = np.array([[2, -1, 2, 0],
                        [2, -1, -1, 1],
                        [2, 2, 1, 0],
                        [2, -1, 2, 2]])

# Right-hand side vector B
constants = np.array([8, 5, 4, 10])

# Find the inverse of the coefficient matrix
inverse_coefficients = np.linalg.inv(coefficients)

# Multiply the inverse matrix by the right-hand side vector to get the solution
solution = np.dot(inverse_coefficients, constants)

# Print the solution
print("Solution using matrix inverse:")
for i in range(len(solution)):
    print(f"x{i + 1} = {solution[i]}")




# Given data
x1_Gauss_Seidel = 2.5
x2_Gauss_Seidel = -0.84375
x3_Gauss_Seidel = 1.3125
x4_Gauss_Seidel = 1.0

x1_solve_method = 2.2222222222222223
x2_solve_method = -0.888888888888889
x3_solve_method = 1.3333333333333333
x4_solve_method = 1.0

x1_inverse_method = 2.2222222222222223
x2_inverse_method = -0.8888888888888891
x3_inverse_method = 1.3333333333333333
x4_inverse_method = 1.0

# Calculate relative errors
relative_error_solve_method_x1 = np.abs((x1_Gauss_Seidel - x1_solve_method) / x1_Gauss_Seidel) * 100
relative_error_solve_method_x2 = np.abs((x2_Gauss_Seidel - x2_solve_method) / x2_Gauss_Seidel) * 100
relative_error_solve_method_x3 = np.abs((x3_Gauss_Seidel - x3_solve_method) / x3_Gauss_Seidel) * 100
relative_error_solve_method_x4 = np.abs((x4_Gauss_Seidel - x4_solve_method) / x4_Gauss_Seidel) * 100


# Create a table
table = [["Method", "x1", "x2", "x3", "x4"],
         ["Gauss Seidel", x1_Gauss_Seidel, x2_Gauss_Seidel, x3_Gauss_Seidel, x4_Gauss_Seidel],
         ["Solve Method", x1_solve_method, x2_solve_method, x3_solve_method, x4_solve_method],
         ["İnvers Method",x1_inverse_method,x2_inverse_method,x3_inverse_method,x4_inverse_method],
         ["Relative Error (%)", relative_error_solve_method_x1, relative_error_solve_method_x2, relative_error_solve_method_x3, relative_error_solve_method_x4]]

# Print the table using tabulate
print(tabulate(table, headers="firstrow", tablefmt="rst", numalign="center"))




                                                ### Cramer Rule ###
# Katsayı matrisi (A)
A = np.array([[2, 3, -1],
              [1, -2, 2],
              [-1, 4, 1]])

# Bağımsız terim matrisi (B)
B = np.array([4, 6, 5])

# Ana determinant
det_A = np.linalg.det(A)

# x1 için
A1 = A.copy()
A1[:, 0] = B
det_A1 = np.linalg.det(A1)
x1 = det_A1 / det_A

# x2 için
A2 = A.copy()
A2[:, 1] = B
det_A2 = np.linalg.det(A2)
x2 = det_A2 / det_A

# x3 için
A3 = A.copy()
A3[:, 2] = B
det_A3 = np.linalg.det(A3)
x3 = det_A3 / det_A

# Sonuçları belirli bir ondalık basamak sayısına yuvarla (örneğin, 2 ondalık basamak)
x1_rounded = round(x1, 2)
x2_rounded = round(x2, 2)
x3_rounded = round(x3, 2)

# Yuvarlanmış sonuçları yazdırma
print("Çözüm:")
print("x1 =", x1_rounded)
print("x2 =", x2_rounded)
print("x3 =", x3_rounded)
##### 3 


# Denklem sistemini tanımla
A = np.array([[2, -1, 1], [3, 3, 9], [3, 3, 5]])
B = np.array([-1, 0, 4])

# LU faktörizasyonunu yap
P, L, U = lu(A) # type: ignore

np.set_printoptions(precision=0, suppress=True, linewidth=100)
# Pb = LUx formülünü çözerek x'i bul
y = solve(L, np.dot(P, B))
x = solve(U, y)

print("P matrisi:")
print(P)
print("\nL matrisi:")
print(L)
print("\nU matrisi:")
print(U)
print("\nÇözüm:")
print(x)

#### 4

A = np.array([[4, 2, 2], [2, 5, 1], [2, 1, 6]])

L = np.linalg.cholesky(A)

print("Cholesky faktörü (NumPy):\n", L)




A = np.array([[4, 2, 2], [2, 5, 1], [2, 1, 6]])

L = cholesky(A, lower=True)

print("Cholesky faktörü (SciPy):\n", L)


import numpy as np
from scipy.linalg import cholesky

# Matrisi tanımla
A = np.array([[4, 2, 2], [2, 5, 1], [2, 1, 6]])

# NumPy'deki Cholesky fonksiyonunu kullan
L_numpy = np.linalg.cholesky(A)

# SciPy'deki Cholesky fonksiyonunu kullan
L_scipy = cholesky(A, lower=True)


# İki matris arasındaki farkı kontrol et 
are_close = np.allclose(L_numpy, L_scipy)

# Sonucu yazdır
print("Do NumPy and SciPy Cholesky functions produce the same result?", are_close)

# Bu kod, NumPy ve SciPy Cholesky fonksiyonları tarafından üretilen matrislerin eşit olup olmadığını kontrol eder ve are_close değişkeni üzerinden sonucu ekrana yazdırır.
# Eğer her iki kütüphane de aynı sonucu üretiyorsa, are_close True olacaktır.
import numpy as np

# Matris A1_5 için
A1_5 = np.array([[1, 2], [-1, 2]])

# Matris A2_5 için
A2_5 = np.array([[-1, 2, 0], [0, 3, 4], [0, 0, 7]])

# Özdeğerler ve özvektörler için eig fonksiyonunu kullanma
eigenvalues_A1_5, eigenvectors_A1_5 = np.linalg.eig(A1_5)
eigenvalues_A2_5, eigenvectors_A2_5 = np.linalg.eig(A2_5)

# Spektral yarıçap hesaplama
spectral_radius_A1_5 = np.max(np.abs(eigenvalues_A1_5))
spectral_radius_A2_5 = np.max(np.abs(eigenvalues_A2_5))


# Sonuçları yazdırma
print("Matris A1_5 için Özdeğerler:")
print(eigenvalues_A1_5)
print("\nMatris A1_5 için Özvektörler:")
print(eigenvectors_A1_5)
print("\nMatris A1_5 için Spektral Yarıçap:")
print(spectral_radius_A1_5)

# A2_5
print("Matris A2_5 için Özdeğerler:")
print(eigenvalues_A2_5)
print("\nMatris A2_5 için Özvektörler:")
print(eigenvectors_A2_5)
print("\nMatris A2_5 için Spektral Yarıçap:")
print(spectral_radius_A2_5)



