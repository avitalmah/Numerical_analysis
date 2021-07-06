import math
from sympy.utilities.lambdify import lambdify
import sympy as sp

def Richardson(f, x0 , min_h , k ):
    t = 2
    h = min_h*t**(k-1)
    A = lambda step : centeral(f,x0,step)
    A_list = []
    while h >= min_h:
        A_list.append(A(h))
        print("h:"+str(h))
        print("A:" + str(A_list[-1]))
        h/=t
    j = 2
    while (len(A_list) != 1):
        print(A_list)
        A_list = inter(A_list,j)
        j += 1

    return A_list[0]

def inter(A_list,j):
    next_A=[]
    for i in range (len(A_list)-1):
        a = A_list[i]
        b = A_list[i+1]
        v = b+(b-a)/(4**(j-1)-1)
        next_A.append(v)
    return next_A

def centeral(f,x0,h):
    return (f(x0+h)-f(x0-h))/(2*h)

def sin(x):
    return math.sin(x*math.pi/180)

def tan(x):
    return math.tan(x*math.pi/180)

def cos(x):
    return math.cos(x*math.pi/180)

def main():

    f = (input("enter function of x: "))
    x = sp.symbols('x')

    f = lambdify(x, f,modules=[{"sin":sin , "cos":cos , "tan":tan },math])
    x0 = 1#float(input("Calc derivative at : "))
    k = 4#float(input("Order of error magnitude : "))
    min_h = 0.25#float(input("Set min step size :"))
    derivative = Richardson(f, x0 , min_h , k)

    print("The result by Richardson Extrapolation:",derivative)

main()