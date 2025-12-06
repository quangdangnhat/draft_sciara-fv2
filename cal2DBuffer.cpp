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

#include "cal2DBuffer.h"
#include <stdlib.h>
#include <string.h>



bool* calAllocBuffer2Db(int rows, int columns) {
    return (bool*)malloc(sizeof(bool)*rows*columns);
}
int* calAllocBuffer2Di(int rows, int columns) {
    return (int*)malloc(sizeof(int)*rows*columns);
}
double* calAllocBuffer2Dr(int rows, int columns) {
    return (double*)malloc(sizeof(double)*rows*columns);
}



void calDeleteBuffer2Db(bool* M) {
    free(M);
}
void calDeleteBuffer2Di(int* M) {
    free(M);
}
void calDeleteBuffer2Dr(double* M) {
    free(M);
}



void calCopyBuffer2Db(bool* M_src, bool* M_dest, int rows, int columns)
{
    memcpy(M_dest, M_src, sizeof(bool)*rows*columns);
}
void calCopyBuffer2Di(int* M_src, int* M_dest, int rows, int columns)
{
    memcpy(M_dest, M_src, sizeof(int)*rows*columns);
}
void calCopyBuffer2Dr(double* M_src, double* M_dest, int rows, int columns)
{
    memcpy(M_dest, M_src, sizeof(double)*rows*columns);
}


void calAddBuffer2Db(bool* M_op1, bool* M_op2,  bool* M_dest, int rows, int columns) {
    int size = rows * columns;
    int i;

    for (i=0; i<size; i++)
        M_dest[i] = M_op1[i] + M_op2[i];
}
void calAddBuffer2Di(int* M_op1, int* M_op2,  int* M_dest, int rows, int columns) {
    int size = rows * columns;
    int i;

    for (i=0; i<size; i++)
        M_dest[i] = M_op1[i] + M_op2[i];
}
void calAddBuffer2Dr(double* M_op1, double* M_op2,  double* M_dest, int rows, int columns) {
    int size = rows * columns;
    int i;

    for (i=0; i<size; i++)
        M_dest[i] = M_op1[i] + M_op2[i];
}



void calSubtractBuffer2Db(bool* M_op1, bool* M_op2,  bool* M_dest, int rows, int columns) {
    int size = rows * columns;
    int i;

    for (i=0; i<size; i++)
        M_dest[i] = M_op1[i] - M_op2[i];
}
void calSubtractBuffer2Di(int* M_op1, int* M_op2,  int* M_dest, int rows, int columns) {
    int size = rows * columns;
    int i;

    for (i=0; i<size; i++)
        M_dest[i] = M_op1[i] - M_op2[i];
}
void calSubtractBuffer2Dr(double* M_op1, double* M_op2,  double* M_dest, int rows, int columns) {
    int size = rows * columns;
    int i;

    for (i=0; i<size; i++)
        M_dest[i] = M_op1[i] - M_op2[i];
}



void calSetBuffer2Db(bool* M, int rows, int columns, bool value)
{
    memset(M, value, sizeof(bool)*rows*columns);
}
void calSetBuffer2Di(int* M, int rows, int columns, int value)
{
    int size = rows * columns;
    int i;

    for (i=0; i<size; i++)
            M[i] = value;
}
void calSetBuffer2Dr(double* M, int rows, int columns, double value)
{
    int size = rows * columns;
    int i;

    for (i=0; i<size; i++)
        M[i] = value;
}

