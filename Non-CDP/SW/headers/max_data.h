#ifndef MAX_DATA__H
#define MAX_DATA__H
#define max2(A,B) ((A) > (B) ? (A) : (B))
#define max4(A,B,C,D) (max2((A) , max2( (B) , max2( (C) , (D)))))
#include <stdio.h>
#include <stdlib.h>



typedef struct max_entry{
	int x;
	int y;
	struct max_entry * next;

}maxEntry;

typedef struct linkedList{
	maxEntry * head;
	maxEntry * tail;
	int size;
}linkedList;


void printEntry(maxEntry * entry);
linkedList * initList();
maxEntry * append(linkedList * list, int x, int y);
int clearList(linkedList * list);
void freeList(linkedList * list);

#endif
