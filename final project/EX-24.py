stopCalc = 0.00001
from datetime import datetime
from itertools import zip_longest

def gauss_Seidel(a, b):
    """
    find the solution with Gauss Seidel methode only for 3*3 matrix
    :param a: coefficient matrix
    :param b: result matrix
    :return: result of x,y,z
    """
    count = 1
    x0, y0, z0 = 0, 0, 0
    x1, y1, z1 = 0, 0, 0
    # chek if There is a dominant diagonal
    if a[0][0] > abs(a[0][1]) + abs(a[0][2]) and \
            a[1][1] > abs(a[1][0]) + abs(a[1][2]) and \
            a[2][2] > abs(a[2][0]) + abs(a[2][1]):
        stopCondition = True
        f1 = lambda y, z: (b[0] - y * a[0][1] - z * a[0][2]) / a[0][0]
        f2 = lambda x, z: (b[1] - x * a[1][0] - z * a[1][2]) / a[1][1]
        f3 = lambda x, y: (b[2] - x * a[2][0] - y * a[2][1]) / a[2][2]
        while stopCondition:
            x1 = f1(y0, z0)
            y1 = f2(x1, z0)
            z1 = f3(x1, y1)
            print('Iteration num %d:\tx=%0.7f\ty=%0.7f\tz=%0.7f\n' % (count, x1, y1, z1))
            error1 = abs(x0 - x1)
            error2 = abs(y0 - y1)
            error3 = abs(z0 - z1)
            count += 1
            x0 = x1
            y0 = y1
            z0 = z1
            stopCondition = error1 > stopCalc and \
                            error2 > stopCalc and \
                            error3 > stopCalc
    else:
        print("In the matrix there is no dominant diagonal")
    print('Total iteration for find the result %d\n' % count)
    return "x1:   "+final_result(x1,datetime.now()) +"  ,  y1:    " \
                                                     "" +final_result(y1,datetime.now())+"  ,  z1:    " \
                                                                                         " " +final_result(z1,datetime.now())


def Jacobi(a, b):
    """
    find the solution with jacobi methode only for 3*3 matrix
    :param a: coefficient matrix
    :param b: result matrix
    :return: result of x,y,z
    """
    x0, y0, z0 = 0, 0, 0
    # chek if There is a dominant diagonal
    x1, y1, z1 = 0, 0, 0
    if a[0][0] > abs(a[0][1]) + abs(a[0][2]) and \
            a[1][1] > abs(a[1][0]) + abs(a[1][2]) and \
            a[2][2] > abs(a[2][0]) + abs(a[2][1]):
        stopCondition = True
        f1 = lambda y, z: (b[0] - y * a[0][1] - z * a[0][2]) / a[0][0]
        f2 = lambda x, z: (b[1] - x * a[1][0] - z * a[1][2]) / a[1][1]
        f3 = lambda x, y: (b[2] - x * a[2][0] - y * a[2][1]) / a[2][2]
        count = 0
        while stopCondition:
            x1 = f1(y0, z0)
            y1 = f2(x0, z0)
            z1 = f3(x0, y0)
            print('Iteration num %d:\tx=%0.7f\ty=%0.7f\tz=%0.7f\n' % (count, x1, y1, z1))
            error1 = abs(x0 - x1)
            error2 = abs(y0 - y1)
            error3 = abs(z0 - z1)
            count += 1
            x0 = x1
            y0 = y1
            z0 = z1
            stopCondition = error1 > stopCalc and \
                            error2 > stopCalc and \
                            error3 > stopCalc
    return "x1:   "+final_result(x1,datetime.now()) +"  ,  y1:    " +final_result(y1,datetime.now())+"  ,  z1:     " +final_result(z1,datetime.now())

def final_result(actual_result,now):
    current_time = now.strftime("%d%H%M")
    return f"{str(actual_result)}00000{current_time}"


class Matrix:
    def __init__(self, list_of_rows, name=None,string_matrix=False):
        # example m=Matrix([[1,2,3],
        #                   [2,3,4],
        #                   [5,5,6]])
        self.list_of_rows = list_of_rows
        self.num_of_cols = self.validation(list_of_rows)
        if not string_matrix:
            self.rows = [list(map(float, row ))for row in list_of_rows]
        else:
            self.rows = [list(map(str, row)) for row in list_of_rows]
        self.num_of_rows = len(self.rows)
        self.det = None
        self.row_range = range(self.num_of_rows)
        self.col_range = range(self.num_of_cols)
        self.name = name
        self.side = None

    @classmethod
    def validation(cls, rows):
        if not isinstance(rows, list):
            raise TypeError("Matrix must to be list")
        if len(rows) == 0:
            raise TypeError("There is no rows")
        num_of_cols = None
        for row in rows:
            if not isinstance(row, list):
                raise TypeError("All rows must be list")
            if num_of_cols is None:
                num_of_cols = len(row)
                if  num_of_cols == 0:
                    raise TypeError("There is no cols")
            elif num_of_cols != len(row):
                raise TypeError("All rows must be in the same length")
            #for cell in row:
            #    if not (isinstance(cell, float) or  isinstance(cell, int)):
            #        raise ValueError("All values must be float")
        return num_of_cols


    def side_matrix(self,other):
        self.side = other

    def __str__(self):
        # Matrix painting
        if self.name:
            result = "{2} Matrix({1}x{0}):\n".format(self.num_of_cols, self.num_of_rows, self.name)
        else:
            result = "Matrix({1}x{0}):\n".format(self.num_of_cols, self.num_of_rows)
        for row in self.rows:
              result += str(list(map(str, row))) + "\n"

        if self.side :
            matrix_line = result.split("\n")
            side_line = str(self.side).split("\n")

            res_line  = [" | ".join([x.ljust(40),y.ljust(30)]) for x,y in zip_longest(matrix_line,side_line)]
            result = "\n".join(res_line)

        return result

    def final_res(self):
        fmt = lambda v: final_result(v, datetime.now())
        result = [list(map(fmt, row)) for row in self.rows]
        name = ""
        if self.name :
            name = self.name + " "
        name += "final"
        return Matrix(result,name,True)

    def __multiplication(self, other):
        # Function for multipcation between matrices.
        if not isinstance(other, Matrix):
            raise TypeError("multiplication is between two matrices")
        if not self.num_of_cols == other.num_of_rows:
            raise ValueError("can't multiplication this matrices")
        result = [[0 for j in other.col_range] for i in self.row_range]
        for i in self.row_range:
            for j in other.col_range:
                for k in range(other.num_of_rows):
                    result[i][j] += self.rows[i][k]*other.rows[k][j]
        if self.num_of_rows == other.num_of_cols:
            # Checking if the matrix is a SquareMatrix.
            return SquareMatrix(result)
        return Matrix(result)

    def __mul__(self, other):
        # A function that when there is a multiplication operator returs __multiplication function.
         return self.__multiplication(other)

    def __truediv__(self, other):
        # A function that when there is a divide operator returs __truediv__ function.
        if (isinstance(other, int) or isinstance(other, float)):
            return self._div_by_scalar(other)

    def _div_by_scalar(self, scalar):
        # Function that divied between matrix and scalar.
        # and return the new matrix.
        result = [[0 for j in self.col_range] for i in self.row_range]
        for i in self.row_range:
            for j in self.col_range:
                result[i][j] = self.rows[i][j]/scalar
        if self.num_of_rows == self.num_of_cols:
            # Checking if the matrix is a SquareMatrix.
            return SquareMatrix(result)
        return Matrix(result)

    def swap(self, i1, i2):
        # Function that swap beetween two rows.
        if i1 == i2:
            return
        self.rows[i1], self.rows[i2] = self.rows[i2], self.rows[i1]



class SquareMatrix(Matrix):
    # A class that representing a square matrix that heiress from Matrix class.
    @classmethod
    def validation(cls, rows):
        num_of_cols = Matrix.validation(rows)
        if num_of_cols != len(rows):
            raise TypeError("Num of cols and rows must be equals")
        return num_of_cols

    def is_invertible(self):
        # A function that return if the matrix is invertible or not.
        return self.deteminante() != 0

    def deteminante(self):
        if self.det is not None:
            # If the determinant has already been calculated return it.
            return self.det
        if self.num_of_cols == 1:
            self.det = self.rows[0][0]
            return self.det
        sign = 1
        # First line development
        i = 0
        result = 0
        for j in self.col_range:
            factor = self.rows[i][j]
            if factor != 0:
                # Calculation by minor.
                result += sign*factor*self.minor(i, j).deteminante()
            sign *= -1
        self.det = result
        return self.det

    def minor(self, row_index, col_index):
        if not (0 <= row_index < self.num_of_rows):
            raise ValueError("Row index not valid")
        if not (0 <= col_index < self.num_of_cols):
            raise ValueError("Col index not valid")
        result =[]
        for i in self.row_range:
            if i == row_index:
                continue
            row = []
            for j in self.col_range:
                if j == col_index:
                    continue
                row.append(self.rows[i][j])
            result.append(row)
        return SquareMatrix(result)

    def gauss_method(self,b):
        return self._gauss_methodA(b)

    def _gauss_methodA(self, b):
        # Calculation b*inverse(A) according to Gauss.
        x = self.inverse()*b
        x.name = "x"
        return x


    def inverse(self):
        if self.is_invertible():
            return self._inverseA()
        else:
            raise ValueError("Matrix must be invertible")

    def _inverseA(self):
        # A function that inverse the matrix
        unit_matrix = ConstDiagonalMatrix(self.num_of_rows, 1, "A inverse") # Creating unit metrix.
        temp_matrix = SquareMatrix((self.list_of_rows.copy())) # Copy the matrix so as not to change the source.
        temp_matrix.side_matrix(unit_matrix)
        print(temp_matrix)
        for i in self.row_range:
            new_row_index, pivot_value = temp_matrix.pivot(i)
            if i != new_row_index:
                # Swap rows
                temp_matrix.swap(i, new_row_index)
                unit_matrix.swap(i, new_row_index)
            factors = [-row[i]/pivot_value for row in temp_matrix.rows]
            # All the elementary operations we do both on the temp matrix and on the unit matrix.
            for j in self.row_range:
                if i == j:
                    continue
                else:
                    factor = factors[j]
                    for k in self.col_range:
                        temp_matrix.rows[j][k] += factor*temp_matrix.rows[i][k]
                        unit_matrix.rows[j][k] += factor * unit_matrix.rows[i][k]

            for k in self.col_range:
                    temp_matrix.rows[i][k] /= pivot_value
                    unit_matrix.rows[i][k] /= pivot_value
            print(temp_matrix)

        # Returning the inverse matrix.
        temp_matrix.side_matrix(None)
        return unit_matrix

    def pivot(self, i):
        # Function for finding pivot.
        max_value = self.rows[i][i]
        max_row = i
        col = [row[i] for row in self.rows]
        for j in self.col_range:
            if j <= i:
                continue
            value = abs(col[j])
            if value > abs(max_value):
                max_value = value
                max_row = j
        if max_value == 0:
            raise ValueError("can't find pivot")  # if in the whole cols there are zeros.
        return max_row, max_value

# Another way to do n inverse matrix. (_inverseB , adjoint)
    def _inverseB(self):
        return self.adjoint() / self.deteminante()
    def adjoint(self):
        result = [[0 for j in self.col_range] for i in self.row_range]
        for i in self.row_range:
            for j in self.col_range:
                result[i][j] = ((-1)**(i+j))*self.minor(i, j).deteminante()
        return SquareMatrix(result)


class ConstDiagonalMatrix(SquareMatrix):
    # A class describing a digonal matrix.
    def __init__(self, size, value, name = None):
        Range = range(size)
        result = [[0 if j != i else value for j in Range ] for i in Range]
        SquareMatrix.__init__(self, result, name)

"""----------------main----------------------"""
A = [[0.04, 0.01, -0.01], [0.2, 0.5, -0.2], [1, 2, 4]]
B = [0.06, 0.3, 11]
x = input("Which way would you like to use?\nGauss Seidel enter 1\nYaakobi enter 2\n")
if int(x) == 1:
    print('the result with Gauss Seidel is: %s' % (gauss_Seidel(A, B)))
elif int(x) == 2:
    print('the result with Gauss Seidel is: %s' % (Jacobi(A, B)))
else:
    print("Wrong choice.\nRun again")


print ("The result with the Elimianation")

A_ = SquareMatrix([[0.04, 0.01, -0.01],
                  [0.2, 0.5, -0.2],
                  [1, 2, 4], ], "A")
b_ = Matrix([[0.06], [0.3], [11]], "b")


print(A_.gauss_method(b_).final_res() )