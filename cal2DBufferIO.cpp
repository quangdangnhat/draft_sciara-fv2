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

//#include "cal2DBuffer.h"
#include "cal2DBufferIO.h"


#include <stdlib.h>

#define STRLEN 256



void calfLoadMatrix2Db(bool* M, int rows, int columns, FILE* f)
{
  char str[STRLEN];
  int i, j;

  for (i=0; i<rows; i++)
    for (j=0; j<columns; j++){
      fscanf(f, "%s", str);
      //calSetMatrixElement(M, columns, i, j, atoi(str));
      M[i*columns+j] = atoi(str);
    }
}

void calfLoadMatrix2Di(int* M, int rows, int columns, FILE* f)
{
  char str[STRLEN];
  int i, j;

  for (i=0; i<rows; i++)
    for (j=0; j<columns; j++){
      fscanf(f, "%s", str);
      //calSetMatrixElement(M, columns, i, j, atoi(str));
      M[i*columns+j] = atoi(str);
    }
}

void calfLoadMatrix2Dr(double* M, int rows, int columns, FILE* f)
{
  char str[STRLEN];
  int i, j;

  for (i=0; i<rows; i++)
    for (j=0; j<columns; j++){
      fscanf(f, "%s", str);
      //calSetMatrixElement(M, columns, i, j, atof(str));
      M[i*columns+j] = atof(str);
    }
}



bool calLoadMatrix2Db(bool* M, int rows, int columns, char* path)
{
  FILE *f = NULL;
  f = fopen(path, "r");

  if ( !f )
    return false;

  calfLoadMatrix2Db(M, rows, columns, f);

  fclose(f);

  return true;
}

bool calLoadMatrix2Di(int* M, int rows, int columns, char* path)
{
  FILE *f = NULL;
  f = fopen(path, "r");

  if ( !f )
    return false;

  calfLoadMatrix2Di(M, rows, columns, f);

  fclose(f);

  return true;
}

bool calLoadMatrix2Dr(double* M, int rows, int columns, char* path)
{
  FILE *f = NULL;
  f = fopen(path, "r");

  if ( !f )
    return false;

  calfLoadMatrix2Dr(M, rows, columns, f);

  fclose(f);

  return true;
}



void calfSaveMatrix2Db(bool* M, int rows, int columns, FILE* f)
{
  char str[STRLEN];
  int i, j;

  for (i=0; i<rows; i++) {
    for (j=0; j<columns; j++) {
      sprintf(str, "%d ", M[i*columns+j]);
      fprintf(f,"%s ",str);
    }
    fprintf(f,"\n");
  }
}

void calfSaveMatrix2Di(int* M, int rows, int columns, FILE* f)
{
  char str[STRLEN];
  int i, j;

  for (i=0; i<rows; i++) {
    for (j=0; j<columns; j++) {
      sprintf(str, "%d ", M[i*columns+j]);
      fprintf(f,"%s ",str);
    }
    fprintf(f,"\n");
  }
}

void calfSaveMatrix2Dr(double* M, int rows, int columns, FILE* f)
{
  char str[STRLEN];
  int i, j;

  for (i=0; i<rows; i++) {
    for (j=0; j<columns; j++) {
	    sprintf(str, "%f ", M[i*columns+j]);
      fprintf(f,"%s ",str);
    }
    fprintf(f,"\n");
  }
}



bool calSaveMatrix2Db(bool* M, int rows, int columns, char* path)
{
  FILE *f;
  f = fopen(path, "w");

  if ( !f )
    return false;

  calfSaveMatrix2Db(M, rows, columns, f);

  fclose(f);

  return true;
}

bool calSaveMatrix2Di(int* M, int rows, int columns, char* path)
{
  FILE *f;
  f = fopen(path, "w");

  if ( !f )
    return false;

  calfSaveMatrix2Di(M, rows, columns, f);

  fclose(f);

  return true;
}

bool calSaveMatrix2Dr(double* M, int rows, int columns, char* path)
{
  FILE *f;
  f = fopen(path, "w");

  if ( !f )
    return false;

  calfSaveMatrix2Dr(M, rows, columns, f);

  fclose(f);

  return true;
}
