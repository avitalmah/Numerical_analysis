import sympy as sp
import math
from sympy.utilities.lambdify import lambdify
import sympy
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime


def final_result(actual_result,now):
    current_time = now.strftime("%d%H%M")
    return f"{str(actual_result)}00000{current_time}"

def find_derivative(my_f):

    x = sp.symbols('x')
    f_prime = my_f.diff(x)
    return f_prime

def print_result(A):
    B = [-1] * 20
    index = 0
    check = True
    for num in A:
        check = True
        for x in B:
            if  round(float(num),2) ==  round(float(x),2):
                check = False
        if check:
            B[index] = num
            index+=1

    if A[0]== -1:
        print("There are no intersections in the field")
    else:
        for num in B:
            if num != -1:
                print(final_result(round(float(num), 4),datetime.now()))





def Newton_Raphson (my_f,start_point,end_point,epsilon):
    x = sp.symbols('x')
    f = my_f
    f = lambdify(x, f)
    A = [-1] * 20
    index = 0
    i = 1
    x = 0.1
    temp_start = start_point
    temp_end = start_point + x
    while round(temp_end,2) <= round(end_point,2):
        result = (temp(my_f, temp_start, temp_end, epsilon))
        if result == 0:
            print("The digit 0 cannot be calculated within the ln function")
        else:
            if result >= temp_start and result <= temp_end:
                A[index] = result
                index += 1

        temp_start = temp_end
        temp_end = temp_end+x
        i += 1
    print("\nAll the intersection points : ")
    print_result(A)





def temp (my_f,start_point,end_point,epsilon):

    i = 1
    x = sp.symbols('x')
    f = my_f
    f_prime = find_derivative(my_f)
    f = lambdify(x, f)
    f_prime = lambdify(x, f_prime)
    xr = (start_point + end_point) / 2
    if f_prime(xr) == 0.0:
        return -1
    xr1 = xr - (f(xr) / f_prime(xr))
    print("i    xr                      f(xr)                  f'(xr)")
    while abs(xr1-xr) > epsilon:
        if (f'{xr:.2f}'[:-1]) == '0.0' or (f'{xr:.2f}'[:-1]) == '-0.0':
            return 0
        xr = xr1
        print(i,"   ",xr,"                  ",f(xr),"                  ", f_prime(xr))
        xr1 = xr - (f(xr) / f_prime(xr))
        i += 1
    return xr





def secant_method(my_f, start_point, end_point, epsilon):
    x = sp.symbols('x')
    f = my_f
    f = lambdify(x, f)
    A = [-1] * 20
    index = 0
    i = 1
    x = 0.1
    temp_start = start_point
    temp_end = start_point + x
    while temp_end <= end_point:
        result = (temp2(my_f, temp_start, temp_end, epsilon))
        if result == 0 :
            print("The digit 0 cannot be calculated within the ln function")
        else:
            if result != 0:
                if result >= temp_start and result <= temp_end :
                    A[index] = result
                    index += 1


        temp_start = temp_end
        temp_end = temp_end + x
        i += 1
    print("\nAll the intersection points : ")
    print_result(A)

def temp2 (my_f,start_point,end_point,epsilon):
    i = 1
    x = sp.symbols('x')
    f = my_f
    f = lambdify(x, f)
    xr = start_point
    xr1 = end_point
    print("i    xr                                xr1                                               f(xr)")
    while  abs(xr1-xr) > epsilon:

        if (f'{xr:.2f}'[:-1]) == '0.0' or (f'{xr:.2f}'[:-1]) == '-0.0' :
            return 0
        print(i,"   ",xr,"                         ",xr1,"                                  ", f(xr))
        temp = xr1
        xr1 = (xr * f(xr1) - xr1 * f(xr))/(f(xr1) - f(xr))
        xr = temp

        i += 1

    return xr


# trapezoidal rule
def trapezoid(my_f,a,b,n):
    x = sp.symbols('x')
    f = lambdify(x, my_f)
    h = (b-a)/n
    xi = np.linspace(a,b,n+1)
    fi = f(xi)
    s = 0.0
    for i in range(1,n):
        s = s + fi[i]
    s = (h/2)*(fi[0] + fi[n]) + h*s
    return s

# romberg method
def romberg(my_f,a,b,eps,n):
# my_f - The function performed on it is integral
# [a,b] - The range in which we find the integral value
# eps - The level of accuracy we want
# n - The maximum order of the Romberg method
    Q = np.zeros((n,n),float)
    for i in range(0,n):
        N = 2**i
        Q[i,0] = trapezoid(my_f,a,b,N)
        for j in range(0,i):
            x = j + 2
            Q[i,j+1] = 1.0/(4**(x-1)-1)*(4**(x-1)*Q[i,j] - Q[i-1,j])
            print("\n","Q",i,j+1,":", Q[i, j + 1])
        if i > 0:
            if (abs(Q[i,j+1] - Q[i,j]) < eps):
               break
    print("\nThe final resulr:",Q[i,j+1])


def simpson(my_f, a, b, n):
    # my_f - The function performed on it is integral
    # [a,b] - The range in which we find the integral value
    # n - How much Simpson needs to calculate
    x = sp.symbols('x')
    f = lambdify(x, my_f)
    h = (b-a)/n
    y = 0.0
    x = a + h
    for i in range(1,n//2 + 1):
        y += 4*f(x)
        x += 2*h

    x = a + 2*h
    for i in range(1,n//2):
        y += 2*f(x)
        x += 2*h
    return (h/3)*(f(a)+f(b)+y)

def main():
    x = sp.symbols('x')

    start_point = 0
    end_point = 1.5
    epsilon = 0.00001
    my_f = (2 * x * math.e ** -x + sp.ln(2 * x ** 2)) * (2 * x ** 3 + 2 * x ** 2 - 3 * x - 5)
    check = input("choose option: 1 to Newton_Raphson, 2 to secant_method ")
    if check == '1':
        Newton_Raphson(my_f, start_point, end_point, epsilon)
    else:
        secant_method(my_f, start_point, end_point, epsilon)
    # integration interval [a,b]
    a = 0.5;
    b = 1.0
    print("\nAccording to the Romberg method, the value of the integral is between 0.5 and 1")
    romberg(my_f, a, b, 1.0e-12, 4)
    print("\nAccording to the Simpson  method, the value of the integral is between 0.5 and 1")
    print(simpson(my_f, a, b, 4))
main()