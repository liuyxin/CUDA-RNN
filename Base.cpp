#include "Base.h"

cublasHandle_t& getHandle()
{
	static cublasHandle_t handle = NULL;
	if(handle == NULL){
		cublasStatus_t stat;
		stat = cublasCreate(&handle);
		if(stat != CUBLAS_STATUS_SUCCESS) {
			printf ("init: CUBLAS initialization failed\n");
			exit(0);
		}
	}
	return handle;
}
/*matrix multiply*/
/*z = x * y*/
/* if uses #define  IDX2C(i,j,leading) (((j)*(leading))+(i))
cublasDgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,mat1->_rows,mat2->_cols,
            mat2->_rows,&alpha,d_a,mat1->_rows,d_b,mat2->_rows,&beta,d_c,mat1->_rows);*/
void matrixMul(cuMatrix<double>* x, cuMatrix<double>*y, cuMatrix<double>*z)
{
	if(x->channels != 1 || y->channels != 1 || z->channels != 1){
		printf("matrix mul chanels != 1\n");
		exit(0);
	}
	if(x->cols != y->rows || z->rows != x->rows || z->cols != y->cols){
		printf("matrix mul chanels != 1\n");
		exit(0);
	}
	double alpha = 1.0;
	double beta = 0.0;
	cublasStatus_t stat;
	stat = cublasDgemm(
		getHandle(),
		CUBLAS_OP_N,
		CUBLAS_OP_N,
		y->cols,
		x->rows,
		y->rows,
		&alpha,
		y->getDev(),
		y->cols,
		x->getDev(),
		x->cols,
		&beta,
		z->getDev(),
		z->cols);
	cudaStreamSynchronize(0);
	if(stat != CUBLAS_STATUS_SUCCESS) {
		printf("matrixMul cublasDgemm error\n");
		cudaFree(x->getDev());
		cudaFree(y->getDev());
		cudaFree(z->getDev());
		exit(0);
	}
}

/*z = T(x) * y*/
void matrixMulTA(cuMatrix<double>* x, cuMatrix<double>*y, cuMatrix<double>*z)
{
	if(x->channels != 1 || y->channels != 1 || z->channels != 1){
		printf("matrix mul chanels != 1\n");
	}

	if(x->rows != y->rows || z->rows != x->cols || z->cols != y->cols){
		printf("matrix mul chanels != 1\n");
		exit(0);
	}
	cublasStatus_t stat;
	double alpha = 1.0;
	double beta = 0.0;
	stat = cublasDgemm(
		getHandle(),
		CUBLAS_OP_N,
		CUBLAS_OP_T,
		y->cols,
		x->cols,
		y->rows,
		&alpha,
		y->getDev(),
		y->cols,
		x->getDev(),
		x->cols,
		&beta,
		z->getDev(),
		z->cols);
	cudaStreamSynchronize(0);
	if(stat != CUBLAS_STATUS_SUCCESS) {
		printf( "matrixMulTA cublasDgemm error\n");
		exit(0);
	}
}

/*z = x * T(y)*/
void matrixMulTB(cuMatrix<double>* x, cuMatrix<double>*y, cuMatrix<double>*z)
{
	if(x->channels != 1 || y->channels != 1 || z->channels != 1){
		printf("matrix mul chanels != 1\n");
		exit(0);
	}
	if(x->cols != y->cols || z->rows != x->rows || z->cols != y->rows){
		printf("matrix mul size not true!");
		exit(0);
	}
	cublasStatus_t stat;
	double alpha = 1.0;
	double beta = 0.0;
	stat = cublasDgemm(
		getHandle(),
		CUBLAS_OP_T,
		CUBLAS_OP_N,
		y->rows,
		x->rows,
		y->cols,
		&alpha,
		y->getDev(),
		y->cols,
		x->getDev(),
		x->cols,
		&beta,
		z->getDev(),
		z->cols);
	cudaStreamSynchronize(0);
	if(stat != CUBLAS_STATUS_SUCCESS) {
		printf("matrixMulTB cublasDgemm error\n");
		exit(0);
	}
}

