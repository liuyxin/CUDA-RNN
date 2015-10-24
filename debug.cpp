#include "debug.h"
void printmatrix(cuMatrix<double>* m,int i) {
	if (i == 0) {
		m->toCpu();
	}
	for (int i = 0; i < m->rows; i++) {
		for (int j = 0; j < m->cols; j++) {
			printf("%f,", m->get(i, j, 0));
		}
		printf("\n");
	}
}

void elemul(cuMatrix<double>* m, cuMatrix<double>* y) {
	m->toCpu();
	y->toCpu();
	for (int i = 0; i < m->rows; i++) {
		for (int j = 0; j < m->cols; j++) {
			printf("%f,", m->get(i, j, 0) * y->get(i, j, 0));
		}
		printf("\n");
	}
}
