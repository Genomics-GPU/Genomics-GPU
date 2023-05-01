#include <stdio.h>
#include <stdlib.h>
#include "../max_data.h"

int main(){

	linkedList * list = initList();
	int max[100];
	int m = 0;

	for(int i = 0; i < 100; i++){
		max[i] = rand();
	}

	for(int i = 0; i < 100; i++){
		if(max[i] > m){
		    clearList(list);
		    append(list, max[i], max[i]);
		    m = max[i];
		}else if(max[i] == m){
		    append(list, max[i], max[i]);
		}
		printList(list);
	}
	
	printList(list);
	printList(list);
	freeList(list);
	return 0;
}
