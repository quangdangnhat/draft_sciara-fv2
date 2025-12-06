void calSetActiveCellsBuffer2Dr(double* M, double value,struct CALModel2D* ca2D);
/*
 * Copyright (c) 2016 OpenCALTeam (https://github.com/OpenCALTeam),
 * Telesio Research Group,
 * Department of Mathematics and Computer Science,
 * University of Calabria, Italy.
 *
 * This file is part of OpenCAL (Open Computing Abstraction Layer).
 *
 * OpenCAL is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as
 * published by the Free Software Foundation, either version 3 of
 * the License, or (at your option) any later version.
 *
 * OpenCAL is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with OpenCAL. If not, see <http://www.gnu.org/licenses/>.
 */

#ifndef cal2DBuffer_h
#define cal2DBuffer_h

//#include "calCommon.h"
//#include cal2D.h


/*! \brief Allocates a byte linearized matrix.
*/
bool* calAllocBuffer2Db(int rows, int columns);

/*! \brief Allocates an int linearized matrix.
*/
int* calAllocBuffer2Di(int rows, int columns);

/*! \brief Allocates a real (floating point) linearized matrix.
*/
double* calAllocBuffer2Dr(int rows, int columns);



/*! \brief Deletes the memory associated to a byte linearized matrix.
*/
void calDeleteBuffer2Db(bool* M);

/*! \brief Deletes the memory associated to an int linearized matrix.
*/
void calDeleteBuffer2Di(int* M);

/*! \brief Deletes the memory associated to a real (floating point) linearized matrix.
*/
void calDeleteBuffer2Dr(double* M);



/*! \brief Byte linearized matrix copy function.
*/
void calCopyBuffer2Db(bool* M_src, bool* M_dest, int rows, int columns);

/*! \brief Int linearized matrix copy function.
*/
void calCopyBuffer2Di(int* M_src, int* M_dest, int rows, int columns);

/*! \brief Real (floating point) linearized matrix copy function.
*/
void calCopyBuffer2Dr(double* M_src, double* M_dest, int rows, int columns);



/*! \brief Byte linearized matrix copy function.
*/
void calAddBuffer2Db(bool* M_op1, bool* M_op2,  bool* M_dest, int rows, int columns);

/*! \brief Int linearized matrix copy function.
*/
void calAddBuffer2Di(int* M_op1, int* M_op2,  int* M_dest, int rows, int columns);

/*! \brief Real (floating point) linearized matrix copy function.
*/
void calAddBuffer2Dr(double* M_op1, double* M_op2,  double* M_dest, int rows, int columns);



/*! \brief Byte linearized matrix subtract function.
*/
void calSubtractBuffer2Db(bool* M_op1, bool* M_op2,  bool* M_dest, int rows, int columns);

/*! \brief Int linearized matrix subtract function.
*/
void calSubtractBuffer2Di(int* M_op1, int* M_op2,  int* M_dest, int rows, int columns);

/*! \brief Real (floating point) linearized matrix subtract function.
*/
void calSubtractBuffer2Dr(double* M_op1, double* M_op2,  double* M_dest, int rows, int columns);



/*! \brief Sets a byte matrix to a constant value.
*/
void calSetBuffer2Db(bool* M, int rows, int columns, bool value);

/*! \brief Sets an int matrix to a constant value.
*/
void calSetBuffer2Di(int* M, int rows, int columns, int value);

/*! \brief Sets a real (floating point) matrix to a constant value.
*/
void calSetBuffer2Dr(double* M, int rows, int columns, double value);



/*! \brief Sets the value of the cell (i, j) of the matrix M.
*/
#define calSetMatrixElement(M, columns, i, j, value) ( (M)[(((i)*(columns)) + (j))] = (value) )


/*! \brief Returns the value of the cell (i, j) of the matrix M.
*/
#define calGetMatrixElement(M, columns, i, j) ( M[ (((i)*(columns)) + (j))] )



#endif
