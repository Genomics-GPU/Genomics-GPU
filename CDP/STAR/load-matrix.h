#ifndef _LOAD_MATRIX_H_
#define _LOAD_MATRIX_H_

#include <algorithm> 
#include <functional> 
#include <cctype>
#include <locale>

#include <fstream>
#include <iostream>
#include <string>

#include <ctype.h>

using namespace std;

//typedef struct {
//	char idx_mat[25];
//	int val_mat[25][25];
//} ScoreMatrix;
//typedef struct {
//	char* idx_mat;
//	int* val_mat;
//} ScoreMatrix;

extern char* idx_mat[25];
extern int val_mat[25][25];


// load score matrix
int load_matrix(const char *path);

// search score in score matrix by two amino acids (eg. A and R)
int searchScore(char x, char y);

#endif