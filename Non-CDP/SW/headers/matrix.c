#include "matrix.h"


matrix * initMatrix(int height, int width ){


    matrix * matr = malloc(sizeof(matrix));
  
    int** mat = (int **) malloc(sizeof(int *) *(height + 1));
    int* storage = (int *) calloc(sizeof(int), (height + 1) * (width + 1));

    if(mat == NULL) return NULL;
    if(storage == NULL) return NULL;

    matr->mat = mat;
    matr->storage = storage; 

    for(int i = 0; i <= height; i++){
	mat[i] = &storage[i * (width + 1)];
    }

    mat[0][0] = 0;
    
    for(int i = 1; i <= width; i ++){
	    mat[0][i] = 0;
    }
    
    for(int i = 1; i <= height; i++){
	    mat[i][0] = 0;
    }
    return matr;
}

void deleteMatrix(matrix * matr){
	
	free(matr->storage);
	free(matr->mat);
	free(matr);
	
}

void printMatrix(matrix * m, int x, int y){

	int ** mat = m->mat;

	for(int i = 0; i <= x; i ++){
		for(int j = 0; j<= y; j++){
			printf("%d ", mat[i][j]);
		}printf("\n");
	}printf("\n");
}




