#ifndef MATRIX__H
#define MATRIX__H

#include <stdlib.h>
#include <stdio.h>


typedef struct matStruct{

    int ** mat;
    int * storage;

}matrix;


void deleteMatrix(matrix * matr);
void printMatrix(matrix * m, int x, int y);
matrix * initMatrix(int height, int width);

#endif
