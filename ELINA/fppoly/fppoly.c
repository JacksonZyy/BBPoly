/*
 *
 *  This source file is part of ELINA (ETH LIbrary for Numerical Analysis).
 *  ELINA is Copyright © 2019 Department of Computer Science, ETH Zurich
 *  This software is distributed under GNU Lesser General Public License Version 3.0.
 *  For more information, see the ELINA project website at:
 *  http://elina.ethz.ch
 *
 *  THE SOFTWARE IS PROVIDED "AS-IS" WITHOUT ANY WARRANTY OF ANY KIND, EITHER
 *  EXPRESS, IMPLIED OR STATUTORY, INCLUDING BUT NOT LIMITED TO ANY WARRANTY
 *  THAT THE SOFTWARE WILL CONFORM TO SPECIFICATIONS OR BE ERROR-FREE AND ANY
 *  IMPLIED WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE,
 *  TITLE, OR NON-INFRINGEMENT.  IN NO EVENT SHALL ETH ZURICH BE LIABLE FOR ANY     
 *  DAMAGES, INCLUDING BUT NOT LIMITED TO DIRECT, INDIRECT,
 *  SPECIAL OR CONSEQUENTIAL DAMAGES, ARISING OUT OF, RESULTING FROM, OR IN
 *  ANY WAY CONNECTED WITH THIS SOFTWARE (WHETHER OR NOT BASED UPON WARRANTY,
 *  CONTRACT, TORT OR OTHERWISE).
 *
 */

#include "backsubstitute.h"
#include <time.h>
#include <math.h>
#include <time.h>
#include "gurobi_c.h"
#include "relu_approx.h"
#include <setoper.h>
#include <cdd.h>

void handle_cddlib_error(dd_ErrorType err){
	if (err!=dd_NoError){
		dd_WriteErrorMessages(stdout,err);
		dd_free_global_constants();  /* At the end, this must be called. */
  		exit(1);
	}
}

bool revert_to_Real(mytype x, double * ax, long * ix)
{
	long ix1,ix2;
	*ax = dd_get_d(x);  
	ix1= (long) (fabs(*ax) * 10000. + 0.5);
	ix2= (long) (fabs(*ax) + 0.5);
	ix2= ix2*10000;
	if(ix1 == ix2){
		if(dd_Positive(x)) {
			*ix = (long)(*ax + 0.5);
		}else{
			*ix = (long)(-(*ax) + 0.5);
			*ix = -(*ix);
		}
		return true; //meaning that we should take from value ix
	}else{
		return false; //meaning that we should take from value ax
	}
}

void cddlib_2dint_testing(void){
	dd_PolyhedraPtr poly1, poly2, poly3;
	dd_MatrixPtr A, B, C, G_poly1, G_poly2, G_poly3;
	dd_rowrange m; 
	dd_colrange d;
	dd_ErrorType err;
	dd_set_global_constants();  /* First, this must be called to use cddlib. */
	m=3; d=3;
	A=dd_CreateMatrix(m,d);
	dd_set_si(A->matrix[0][0],1); dd_set_si(A->matrix[0][1],0); dd_set_si(A->matrix[0][2], -1);
	dd_set_si(A->matrix[1][0],0); dd_set_si(A->matrix[1][1], 1); dd_set_si(A->matrix[1][2], 0);
	dd_set_si(A->matrix[2][0],0); dd_set_si(A->matrix[2][1], -1); dd_set_si(A->matrix[2][2], 1);
	/*  1         -  x2   >= 0
		      x1          >= 0
		     -x1  +  x2   >= 0
	*/
	A->representation=dd_Inequality;
	poly1=dd_DDMatrix2Poly(A, &err);  /* compute the second (generator) representation */
	handle_cddlib_error(err);
	printf("\nInput is H-representation:\n");
	G_poly1=dd_CopyGenerators(poly1);
	dd_WriteMatrix(stdout,A);  printf("\n");
	dd_WriteMatrix(stdout,G_poly1); printf("\n");

	B=dd_CreateMatrix(m,d);
	dd_set_si(B->matrix[0][0],1); dd_set_si(B->matrix[0][1],-1); dd_set_si(B->matrix[0][2], 0);
	dd_set_si(B->matrix[1][0],0); dd_set_si(B->matrix[1][1], 0); dd_set_si(B->matrix[1][2], 1);
	dd_set_si(B->matrix[2][0],0); dd_set_si(B->matrix[2][1], 1); dd_set_si(B->matrix[2][2], -1);
	/*  1   - x1          >= 0
		      x2          >= 0
		      x1  -  x2   >= 0
	*/
	B->representation=dd_Inequality;
	poly2=dd_DDMatrix2Poly(B, &err);  /* compute the second (generator) representation */
	handle_cddlib_error(err);
	printf("\nInput is H-representation:\n");
	G_poly2=dd_CopyGenerators(poly2);
	dd_WriteMatrix(stdout,B);  printf("\n");
	dd_WriteMatrix(stdout,G_poly2); printf("\n");

	// Compute the combination of those four points
	C=dd_CreateMatrix(m * 2,d);
	long i,j,ix;
	double ax; 
	for (i=0; i < G_poly1->rowsize; i++) {
		for (j=0; j < G_poly1->colsize; j++) {
			if(revert_to_Real(G_poly1->matrix[i][j], &ax, &ix)){
				dd_set_si(C->matrix[i][j], ix);
			}
			else{
				dd_set_si(C->matrix[i][j], ax);
			}
		}
	}
	for (i=0; i < G_poly2->rowsize; i++) {
		for(j=0; j < G_poly2->colsize; j++){
			if(revert_to_Real(G_poly2->matrix[i][j], &ax, &ix)){
				dd_set_si(C->matrix[i+m][j], ix);
			}
			else{
				dd_set_si(C->matrix[i+m][j], ax);
			}
		}
	}
	C->representation=dd_Generator;
	poly3=dd_DDMatrix2Poly(C, &err);  /* compute the second (generator) representation */
	handle_cddlib_error(err);
	printf("\nInput is V-representation of two polys:\n");
	G_poly3=dd_CopyInequalities(poly3);
	dd_WriteMatrix(stdout,C);  printf("\n");
	dd_WriteMatrix(stdout,G_poly3); printf("\n");

	dd_FreeMatrix(A);
	dd_FreeMatrix(G_poly1);
	dd_FreeMatrix(B);
	dd_FreeMatrix(G_poly2);
	dd_FreeMatrix(C);
	dd_FreeMatrix(G_poly3);
	dd_FreePolyhedra(poly1);
	dd_FreePolyhedra(poly2);
	dd_FreePolyhedra(poly3);
}

void cddlib_2ddb_testing(void){
	dd_PolyhedraPtr poly1, poly2, poly3;
	dd_MatrixPtr A, B, C, G_poly1, G_poly2, G_poly3;
	dd_rowrange m; 
	dd_colrange d;
	dd_ErrorType err;
	dd_set_global_constants();  /* First, this must be called to use cddlib. */
	m=3; d=3;
	A=dd_CreateMatrix(m,d);
	dd_set_d(A->matrix[0][0],0.3); dd_set_d(A->matrix[0][1],0); dd_set_d(A->matrix[0][2], -1);
	dd_set_d(A->matrix[1][0],0); dd_set_d(A->matrix[1][1], 1); dd_set_d(A->matrix[1][2], 0);
	dd_set_d(A->matrix[2][0],0); dd_set_d(A->matrix[2][1], -1); dd_set_d(A->matrix[2][2], 1);
	/*  0.5       -  x2   >= 0
		      x1          >= 0
		     -x1  +  x2   >= 0
	*/
	A->representation=dd_Inequality;
	poly1=dd_DDMatrix2Poly(A, &err);  /* compute the second (generator) representation */
	handle_cddlib_error(err);
	printf("\nInput is H-representation:\n");
	G_poly1=dd_CopyGenerators(poly1);
	dd_WriteMatrix(stdout,A);  printf("\n");
	dd_WriteMatrix(stdout,G_poly1); printf("\n");

	B=dd_CreateMatrix(m,d);
	dd_set_d(B->matrix[0][0],0.3); dd_set_d(B->matrix[0][1],-1); dd_set_d(B->matrix[0][2], 0);
	dd_set_d(B->matrix[1][0],0); dd_set_d(B->matrix[1][1], 0); dd_set_d(B->matrix[1][2], 1);
	dd_set_d(B->matrix[2][0],0); dd_set_d(B->matrix[2][1], 1); dd_set_d(B->matrix[2][2], -1);
	/* 0.5  - x1          >= 0
		      x2          >= 0
		      x1  -  x2   >= 0
	*/
	B->representation=dd_Inequality;
	poly2=dd_DDMatrix2Poly(B, &err);  /* compute the second (generator) representation */
	handle_cddlib_error(err);
	printf("\nInput is H-representation:\n");
	G_poly2=dd_CopyGenerators(poly2);
	dd_WriteMatrix(stdout,B);  printf("\n");
	dd_WriteMatrix(stdout,G_poly2); printf("\n");

	// Compute the combination of those four points
	C=dd_CreateMatrix(m * 2,d);
	long i,j,ix;
	double ax; 
	for (i=0; i < G_poly1->rowsize; i++) {
		for (j=0; j < G_poly1->colsize; j++) {
			if(revert_to_Real(G_poly1->matrix[i][j], &ax, &ix)){
				dd_set_d(C->matrix[i][j], ix);
			}
			else{
				dd_set_d(C->matrix[i][j], ax);
			}
		}
	}
	for (i=0; i < G_poly2->rowsize; i++) {
		for(j=0; j < G_poly2->colsize; j++){
			if(revert_to_Real(G_poly2->matrix[i][j], &ax, &ix)){
				dd_set_d(C->matrix[i+m][j], ix);
			}
			else{
				dd_set_d(C->matrix[i+m][j], ax);
			}
		}
	}
	C->representation=dd_Generator;
	poly3=dd_DDMatrix2Poly(C, &err);  /* compute the second (generator) representation */
	handle_cddlib_error(err);
	printf("\nInput is V-representation of two polys:\n");
	G_poly3=dd_CopyInequalities(poly3);
	dd_WriteMatrix(stdout,C);  printf("\n");
	dd_WriteMatrix(stdout,G_poly3); printf("\n");

	dd_FreeMatrix(A);
	dd_FreeMatrix(G_poly1);
	dd_FreeMatrix(B);
	dd_FreeMatrix(G_poly2);
	dd_FreeMatrix(C);
	dd_FreeMatrix(G_poly3);
	dd_FreePolyhedra(poly1);
	dd_FreePolyhedra(poly2);
	dd_FreePolyhedra(poly3);
}

void cddlib_floaterror_testing(void){
	dd_PolyhedraPtr poly3;
	dd_MatrixPtr  C, G_poly3;
	dd_rowrange m; 
	dd_colrange d;
	dd_ErrorType err;
	dd_set_global_constants();  /* First, this must be called to use cddlib. */
	m=4; d=3;
	// Compute the combination of those four points
	C=dd_CreateMatrix(m,d);
	dd_set_d(C->matrix[0][0], 1); dd_set_d(C->matrix[0][1], -0.3); dd_set_d(C->matrix[0][2], 0);
	dd_set_d(C->matrix[1][0], 1); dd_set_d(C->matrix[1][1], 0); dd_set_d(C->matrix[1][2], -0.3);
	dd_set_d(C->matrix[2][0], 1); dd_set_d(C->matrix[2][1], 0); dd_set_d(C->matrix[2][2], 0);
	dd_set_d(C->matrix[3][0], 1); dd_set_d(C->matrix[3][1], -0.3); dd_set_d(C->matrix[3][2], -0.3);
	C->representation=dd_Generator;
	poly3=dd_DDMatrix2Poly(C, &err);  /* compute the second (generator) representation */
	handle_cddlib_error(err);
	printf("\nInput is V-representation:\n");
	G_poly3=dd_CopyInequalities(poly3);
	dd_WriteMatrix(stdout,C);  printf("\n");
	dd_WriteMatrix(stdout,G_poly3); printf("\n");

	dd_FreeMatrix(C);
	dd_FreeMatrix(G_poly3);
	dd_FreePolyhedra(poly3);
}

void cdd_numerical_inconsistent(void){
	dd_PolyhedraPtr poly3;
	dd_MatrixPtr  C, G_poly3;
	dd_rowrange m; 
	dd_colrange d;
	dd_ErrorType err;
	dd_set_global_constants();  /* First, this must be called to use cddlib. */
	m=5; d=4;
	// Compute the combination of those four points
	C=dd_CreateMatrix(m,d);
	dd_set_d(C->matrix[0][0], 1); dd_set_d(C->matrix[0][1], -316.5); dd_set_d(C->matrix[0][2], -109); dd_set_d(C->matrix[0][3], -94.97);
	dd_set_d(C->matrix[1][0], 1); dd_set_d(C->matrix[1][1], 245); dd_set_d(C->matrix[1][2], 228); dd_set_d(C->matrix[1][3], 79);
	dd_set_d(C->matrix[2][0], 1); dd_set_d(C->matrix[2][1], -160); dd_set_d(C->matrix[2][2], 132.3); dd_set_d(C->matrix[2][3], 109);
	dd_set_d(C->matrix[3][0], 1); dd_set_d(C->matrix[3][1], -377.3); dd_set_d(C->matrix[3][2], 63.6); dd_set_d(C->matrix[3][3], -99.1);
	dd_set_d(C->matrix[4][0], 1); dd_set_d(C->matrix[4][1], -376.5); dd_set_d(C->matrix[4][2], 81.2); dd_set_d(C->matrix[4][3], -93.4);
	C->representation=dd_Generator;
	poly3=dd_DDMatrix2Poly(C, &err);  /* compute the second (generator) representation */
	handle_cddlib_error(err);
	printf("\nInput is V-representation:\n");
	G_poly3=dd_CopyInequalities(poly3);
	dd_WriteMatrix(stdout,C);  printf("\n");
	dd_WriteMatrix(stdout,G_poly3); printf("\n");

	dd_FreeMatrix(C);
	dd_FreeMatrix(G_poly3);
	dd_FreePolyhedra(poly3);
}

void cdd_net_fragment_test(void){
	dd_PolyhedraPtr poly1, poly2, poly3, poly4, polyu;
	dd_MatrixPtr A, B, C, D, E, G_poly1, G_poly2, G_poly3, G_poly4, G_polyu;
	dd_rowrange m, count; 
	dd_colrange d;
	dd_ErrorType err;
	dd_set_global_constants();  /* First, this must be called to use cddlib. */
	// The first case where x3, x7 are both deactivated
	// The vars include x3 \in [-1,2], alpha1 \in [-1,1], x9 \in [0,2]
	m=10; d=4;
	A=dd_CreateMatrix(m,d);
	dd_set_d(A->matrix[0][0],1); dd_set_d(A->matrix[0][1], 1); dd_set_d(A->matrix[0][2], 0); dd_set_d(A->matrix[0][3], 0);
	dd_set_d(A->matrix[1][0],2); dd_set_d(A->matrix[1][1], -1); dd_set_d(A->matrix[1][2], 0); dd_set_d(A->matrix[1][3], 0);
	dd_set_d(A->matrix[2][0],1); dd_set_d(A->matrix[2][1], 0); dd_set_d(A->matrix[2][2], 1); dd_set_d(A->matrix[2][3], 0);
	dd_set_d(A->matrix[3][0],1); dd_set_d(A->matrix[3][1], 0); dd_set_d(A->matrix[3][2], -1); dd_set_d(A->matrix[3][3], 0);
	dd_set_d(A->matrix[4][0],0); dd_set_d(A->matrix[4][1], 0); dd_set_d(A->matrix[4][2], 0); dd_set_d(A->matrix[4][3], 1);
	dd_set_d(A->matrix[5][0],2); dd_set_d(A->matrix[5][1], 0); dd_set_d(A->matrix[5][2], 0); dd_set_d(A->matrix[5][3], -1);
	/*  1     +  x3   >= 0
		2     -  x3   >= 0
		1           +  alpha   >= 0   
		1           -  alpha   >= 0  
		                      x9   >= 0
		2					  -x9   >= 0
		variable interval constraint
	*/
	dd_set_d(A->matrix[6][0],0); dd_set_d(A->matrix[6][1], -1); dd_set_d(A->matrix[6][2], 0); dd_set_d(A->matrix[6][3], 0);
	dd_set_d(A->matrix[7][0],0); dd_set_d(A->matrix[7][1], 0); dd_set_d(A->matrix[7][2], -1); dd_set_d(A->matrix[7][3], 0);
	dd_set_d(A->matrix[8][0],0); dd_set_d(A->matrix[8][1], 0); dd_set_d(A->matrix[8][2], 0); dd_set_d(A->matrix[8][3], 1);
	dd_set_d(A->matrix[9][0],0); dd_set_d(A->matrix[9][1], 0); dd_set_d(A->matrix[9][2], 0); dd_set_d(A->matrix[9][3], -1);
	/*  - x3   >= 0  
		-  alpha   >= 0  
		x9   >= 0
		-x9   >= 0
		branch condition constraint
	*/
	A->representation=dd_Inequality;
	poly1=dd_DDMatrix2Poly(A, &err);  /* compute the second (generator) representation */
	handle_cddlib_error(err);
	printf("\nInput is H-representation:\n");
	G_poly1=dd_CopyGenerators(poly1);
	dd_WriteMatrix(stdout,A);  printf("\n");
	dd_WriteMatrix(stdout,G_poly1); printf("\n");

	// The second case where x3 deactivated, x7 activated
	B=dd_CreateMatrix(m,d);
	dd_set_d(B->matrix[0][0],1); dd_set_d(B->matrix[0][1], 1); dd_set_d(B->matrix[0][2], 0); dd_set_d(B->matrix[0][3], 0);
	dd_set_d(B->matrix[1][0],2); dd_set_d(B->matrix[1][1], -1); dd_set_d(B->matrix[1][2], 0); dd_set_d(B->matrix[1][3], 0);
	dd_set_d(B->matrix[2][0],1); dd_set_d(B->matrix[2][1], 0); dd_set_d(B->matrix[2][2], 1); dd_set_d(B->matrix[2][3], 0);
	dd_set_d(B->matrix[3][0],1); dd_set_d(B->matrix[3][1], 0); dd_set_d(B->matrix[3][2], -1); dd_set_d(B->matrix[3][3], 0);
	dd_set_d(B->matrix[4][0],0); dd_set_d(B->matrix[4][1], 0); dd_set_d(B->matrix[4][2], 0); dd_set_d(B->matrix[4][3], 1);
	dd_set_d(B->matrix[5][0],2); dd_set_d(B->matrix[5][1], 0); dd_set_d(B->matrix[5][2], 0); dd_set_d(B->matrix[5][3], -1);
	/* var intervals */
	dd_set_d(B->matrix[6][0],0); dd_set_d(B->matrix[6][1], -1); dd_set_d(B->matrix[6][2], 0); dd_set_d(B->matrix[6][3], 0);
	dd_set_d(B->matrix[7][0],0); dd_set_d(B->matrix[7][1], 0); dd_set_d(B->matrix[7][2], 1); dd_set_d(B->matrix[7][3], 0);
	dd_set_d(B->matrix[8][0],0); dd_set_d(B->matrix[8][1], 0); dd_set_d(B->matrix[8][2], -1); dd_set_d(B->matrix[8][3], 1);
	dd_set_d(B->matrix[9][0],0); dd_set_d(B->matrix[9][1], 0); dd_set_d(B->matrix[9][2], 1); dd_set_d(B->matrix[9][3], -1);
	/*  - x3   >= 0  
		alpha   >= 0  
		- alpha + x9   >= 0
		alpha - x9   >= 0
		branch condition constraint
	*/
	B->representation=dd_Inequality;
	poly2=dd_DDMatrix2Poly(B, &err);  /* compute the second (generator) representation */
	handle_cddlib_error(err);
	printf("\nInput is H-representation:\n");
	G_poly2=dd_CopyGenerators(poly2);
	dd_WriteMatrix(stdout,B);  printf("\n");
	dd_WriteMatrix(stdout,G_poly2); printf("\n");

	// The third case where x3 activated, x7 activated
	C=dd_CreateMatrix(m,d);
	dd_set_d(C->matrix[0][0],1); dd_set_d(C->matrix[0][1], 1);  dd_set_d(C->matrix[0][2], 0);  dd_set_d(C->matrix[0][3], 0);
	dd_set_d(C->matrix[1][0],2); dd_set_d(C->matrix[1][1], -1); dd_set_d(C->matrix[1][2], 0);  dd_set_d(C->matrix[1][3], 0);
	dd_set_d(C->matrix[2][0],1); dd_set_d(C->matrix[2][1], 0);  dd_set_d(C->matrix[2][2], 1);  dd_set_d(C->matrix[2][3], 0);
	dd_set_d(C->matrix[3][0],1); dd_set_d(C->matrix[3][1], 0);  dd_set_d(C->matrix[3][2], -1); dd_set_d(C->matrix[3][3], 0);
	dd_set_d(C->matrix[4][0],0); dd_set_d(C->matrix[4][1], 0);  dd_set_d(C->matrix[4][2], 0);  dd_set_d(C->matrix[4][3], 1);
	dd_set_d(C->matrix[5][0],2); dd_set_d(C->matrix[5][1], 0);  dd_set_d(C->matrix[5][2], 0);  dd_set_d(C->matrix[5][3], -1);
	/* var intervals */
	dd_set_d(C->matrix[6][0],0); dd_set_d(C->matrix[6][1], 1); dd_set_d(C->matrix[6][2], 0);  dd_set_d(C->matrix[6][3], 0);
	dd_set_d(C->matrix[7][0],0); dd_set_d(C->matrix[7][1], 1);  dd_set_d(C->matrix[7][2], 1);  dd_set_d(C->matrix[7][3], 0);
	dd_set_d(C->matrix[8][0],0); dd_set_d(C->matrix[8][1], -1);  dd_set_d(C->matrix[8][2], -1); dd_set_d(C->matrix[8][3], 1);
	dd_set_d(C->matrix[9][0],0); dd_set_d(C->matrix[9][1], 1);  dd_set_d(C->matrix[9][2], 1);  dd_set_d(C->matrix[9][3], -1);
	/*  x3   >= 0  
		x3 + alpha   >= 0  
		- x3 - alpha + x9   >= 0
		 x3 + alpha - x9   >= 0
		branch condition constraint
	*/
	C->representation=dd_Inequality;
	poly3=dd_DDMatrix2Poly(C, &err);  /* compute the second (generator) representation */
	handle_cddlib_error(err);
	printf("\nInput is H-representation:\n");
	G_poly3=dd_CopyGenerators(poly3);
	dd_WriteMatrix(stdout,C);  printf("\n");
	dd_WriteMatrix(stdout,G_poly3); printf("\n");

	// The forth case where x3 activated, x7 dectivated
	D=dd_CreateMatrix(m,d);
	dd_set_d(D->matrix[0][0],1); dd_set_d(D->matrix[0][1], 1);  dd_set_d(D->matrix[0][2], 0);  dd_set_d(D->matrix[0][3], 0);
	dd_set_d(D->matrix[1][0],2); dd_set_d(D->matrix[1][1], -1); dd_set_d(D->matrix[1][2], 0);  dd_set_d(D->matrix[1][3], 0);
	dd_set_d(D->matrix[2][0],1); dd_set_d(D->matrix[2][1], 0);  dd_set_d(D->matrix[2][2], 1);  dd_set_d(D->matrix[2][3], 0);
	dd_set_d(D->matrix[3][0],1); dd_set_d(D->matrix[3][1], 0);  dd_set_d(D->matrix[3][2], -1); dd_set_d(D->matrix[3][3], 0);
	dd_set_d(D->matrix[4][0],0); dd_set_d(D->matrix[4][1], 0);  dd_set_d(D->matrix[4][2], 0);  dd_set_d(D->matrix[4][3], 1);
	dd_set_d(D->matrix[5][0],2); dd_set_d(D->matrix[5][1], 0);  dd_set_d(D->matrix[5][2], 0);  dd_set_d(D->matrix[5][3], -1);
	/* var intervals */
	dd_set_d(D->matrix[6][0],0); dd_set_d(D->matrix[6][1], 1);  dd_set_d(D->matrix[6][2], 0);  dd_set_d(D->matrix[6][3], 0);
	dd_set_d(D->matrix[7][0],0); dd_set_d(D->matrix[7][1], -1); dd_set_d(D->matrix[7][2], -1); dd_set_d(D->matrix[7][3], 0);
	dd_set_d(D->matrix[8][0],0); dd_set_d(D->matrix[8][1], 0);  dd_set_d(D->matrix[8][2], -1); dd_set_d(D->matrix[8][3], 1);
	dd_set_d(D->matrix[9][0],0); dd_set_d(D->matrix[9][1], 0);  dd_set_d(D->matrix[9][2], 1);  dd_set_d(D->matrix[9][3], -1);
	/*  x3   >= 0  
		- x3 - alpha   >= 0  
		+ x9   >= 0
		- x9   >= 0
		branch condition constraint
	*/
	D->representation=dd_Inequality;
	poly4=dd_DDMatrix2Poly(D, &err);  /* compute the second (generator) representation */
	handle_cddlib_error(err);
	printf("\nInput is H-representation:\n");
	G_poly4=dd_CopyGenerators(poly4);
	dd_WriteMatrix(stdout,D);  printf("\n");
	dd_WriteMatrix(stdout,G_poly4); printf("\n");

	// Compute the combination of four branches
	m = G_poly1->rowsize + G_poly2->rowsize + G_poly3->rowsize + G_poly4->rowsize;
	E = dd_CreateMatrix(m,d);
	long i,j,ix;
	double ax; 
	for (i=0; i < G_poly1->rowsize; i++){
		for (j=0; j < G_poly1->colsize; j++) {
			if(revert_to_Real(G_poly1->matrix[i][j], &ax, &ix)){
				dd_set_d(E->matrix[i][j], ix);
			}
			else{
				dd_set_d(E->matrix[i][j], ax);
			}
		}
	}
	count = G_poly1->rowsize;
	for (i=0; i < G_poly2->rowsize; i++) {
		for(j=0; j < G_poly2->colsize; j++){
			if(revert_to_Real(G_poly2->matrix[i][j], &ax, &ix)){
				dd_set_d(E->matrix[i+count][j], ix);
			}
			else{
				dd_set_d(E->matrix[i+count][j], ax);
			}
		}
	}
	count = count + G_poly2->rowsize;
	for (i=0; i < G_poly3->rowsize; i++) {
		for(j=0; j < G_poly3->colsize; j++){
			if(revert_to_Real(G_poly3->matrix[i][j], &ax, &ix)){
				dd_set_d(E->matrix[i+count][j], ix);
			}
			else{
				dd_set_d(E->matrix[i+count][j], ax);
			}
		}
	}
	count = count + G_poly3->rowsize;
	for (i=0; i < G_poly4->rowsize; i++) {
		for(j=0; j < G_poly4->colsize; j++){
			if(revert_to_Real(G_poly4->matrix[i][j], &ax, &ix)){
				dd_set_d(E->matrix[i+count][j], ix);
			}
			else{
				dd_set_d(E->matrix[i+count][j], ax);
			}
		}
	}
	E->representation=dd_Generator;
	polyu=dd_DDMatrix2Poly(E, &err);  /* compute the second (generator) representation */
	handle_cddlib_error(err);
	printf("\nInput is V-representation of two polys:\n");
	G_polyu=dd_CopyInequalities(polyu);
	dd_WriteMatrix(stdout,E);  printf("\n");
	dd_WriteMatrix(stdout,G_polyu); printf("\n");

	dd_FreeMatrix(A); dd_FreeMatrix(B); dd_FreeMatrix(C); dd_FreeMatrix(D); dd_FreeMatrix(E);
	dd_FreeMatrix(G_poly1); dd_FreeMatrix(G_poly2); dd_FreeMatrix(G_poly3); dd_FreeMatrix(G_poly4); dd_FreeMatrix(G_polyu);
	dd_FreePolyhedra(poly1); dd_FreePolyhedra(poly2); dd_FreePolyhedra(poly3); dd_FreePolyhedra(poly4); dd_FreePolyhedra(polyu);
}

fppoly_t* fppoly_of_abstract0(elina_abstract0_t* a)
{
  return (fppoly_t*)a->value;
}

elina_abstract0_t* abstract0_of_fppoly(elina_manager_t* man, fppoly_t* fp)
{
  elina_abstract0_t* r = malloc(sizeof(elina_abstract0_t));
  assert(r);
  r->value = fp;
  r->man = elina_manager_copy(man);
  return r;
}

static inline void fppoly_internal_free(fppoly_internal_t* pr)
{
    if (pr) {
	pr->funid = ELINA_FUNID_UNKNOWN;
	free(pr);
	pr = NULL;
    }
}

static inline fppoly_internal_t* fppoly_internal_alloc(void)
{
    fppoly_internal_t* pr = (fppoly_internal_t*)malloc(sizeof(fppoly_internal_t));
    pr->funid = ELINA_FUNID_UNKNOWN;
    pr->man = NULL;
    pr->funopt = NULL; 
    pr->min_denormal = ldexpl(1.0,-1074);
	// minimum positive subnormal double
    pr->ulp = ldexpl(1.0,-52);
    return pr;
}

/* back pointer to our internal structure from the manager */
fppoly_internal_t* fppoly_init_from_manager(elina_manager_t* man, elina_funid_t funid)
{
	
    fppoly_internal_t* pr = (fppoly_internal_t*)man->internal;
    pr->funid = funid;
	
    if (!(pr->man)) pr->man = man;
	
    return pr;
}

elina_manager_t * fppoly_manager_alloc(void){
	void** funptr;
	fesetround(FE_UPWARD);
	fppoly_internal_t *pr = fppoly_internal_alloc();
	elina_manager_t *man = elina_manager_alloc("fppoly",/* Library name */
			"1.0", /* version */
			pr, /* internal structure */
			(void (*)(void*))fppoly_internal_free /* free function for internal */
			);
	funptr = man->funptr;
	funptr[ELINA_FUNID_FREE] = &fppoly_free;
	/* 3.Printing */
	funptr[ELINA_FUNID_FPRINT] = &fppoly_fprint;
	return man;
}

neuron_t *neuron_alloc(void){
	neuron_t *res =  (neuron_t *)malloc(sizeof(neuron_t));
	res->lb = INFINITY;
	res->ub = INFINITY;
	res->lexpr = NULL;
	res->uexpr = NULL;
	res->summary_lexpr = NULL;
	res->summary_uexpr = NULL;
	res->backsubstituted_lexpr = NULL;
	res->backsubstituted_uexpr = NULL;
	return res;
}

layer_t * create_layer(size_t size, bool is_activation){
	layer_t *layer = (layer_t*)malloc(sizeof(layer_t));
	layer->dims = size;
	layer->is_activation = is_activation;
	layer->neurons = (neuron_t**)malloc(size*sizeof(neuron_t*));
	size_t i;
	for(i=0; i < size; i++){
		layer->neurons[i] = neuron_alloc();
	}
	layer->h_t_inf = NULL;
	layer->h_t_sup = NULL;
	layer->c_t_inf = NULL;
	layer->c_t_sup = NULL;
	layer->is_concat = false;
	// The default for the end-of-block flag is false
	layer->is_end_layer_of_blk = false;
	layer->is_start_layer_of_blk = false;
	layer->start_idx_in_same_blk = -1;
	layer->C = NULL;
	layer->num_channels = 0;
	return layer;
}

void fppoly_from_network_input_box(fppoly_t *res, size_t intdim, size_t realdim, double *inf_array, double *sup_array){
	res->layers = NULL;
	res->numlayers = 0;
	res->lstm_index = 0;
	size_t num_pixels = intdim + realdim;
	res->input_inf = (double *)malloc(num_pixels*sizeof(double));
	res->input_sup = (double *)malloc(num_pixels*sizeof(double));
	res->original_input_inf = (double *)malloc(num_pixels*sizeof(double));
	res->original_input_sup = (double *)malloc(num_pixels*sizeof(double));
	res->input_lexpr = NULL;
	res->input_uexpr = NULL;
	size_t i;
	for(i=0; i < num_pixels; i++){
		res->input_inf[i] = -inf_array[i];
		res->input_sup[i] = sup_array[i];
		res->original_input_inf[i] = -inf_array[i];
		res->original_input_sup[i] = sup_array[i];
	}
	res->num_pixels = num_pixels;
    res->spatial_indices = NULL;
    res->spatial_neighbors = NULL;
}

elina_abstract0_t * fppoly_from_network_input(elina_manager_t *man, size_t intdim, size_t realdim, double *inf_array, double *sup_array){
	fppoly_t * res = (fppoly_t *)malloc(sizeof(fppoly_t));
	fppoly_from_network_input_box(res, intdim, realdim, inf_array, sup_array);
	return abstract0_of_fppoly(man,res);
}

void fppoly_set_network_input_box(elina_manager_t *man, elina_abstract0_t* element, size_t intdim, size_t realdim, double *inf_array, double * sup_array){
    fppoly_t * res = fppoly_of_abstract0(element);
    size_t num_pixels = intdim + realdim;
    res->numlayers = 0;
    size_t i;
    for(i=0; i < num_pixels; i++){
        res->input_inf[i] = -inf_array[i];
        res->input_sup[i] = sup_array[i];
    }
}

elina_abstract0_t* fppoly_from_network_input_poly(elina_manager_t *man, size_t intdim, size_t realdim, double *inf_array, double *sup_array, double * lexpr_weights, double * lexpr_cst, size_t * lexpr_dim, double * uexpr_weights, double * uexpr_cst, size_t * uexpr_dim, size_t expr_size, size_t * spatial_indices, size_t * spatial_neighbors, size_t spatial_size, double spatial_gamma) {
    fppoly_t * res = (fppoly_t *)malloc(sizeof(fppoly_t));
	
	fppoly_from_network_input_box(res, intdim, realdim, inf_array, sup_array);
	size_t num_pixels = intdim + realdim;
	res->input_lexpr = (expr_t **)malloc(num_pixels*sizeof(expr_t *));
	res->input_uexpr = (expr_t **)malloc(num_pixels*sizeof(expr_t *));
	res->original_input_inf = NULL;
	res->original_input_sup = NULL;
	size_t i;
	double * tmp_weights = (double*)malloc(expr_size*sizeof(double));
	size_t * tmp_dim = (size_t*)malloc(expr_size*sizeof(size_t));
	
	for(i = 0; i < num_pixels; i++){
		
		size_t j;
		for(j=0; j < expr_size; j++){
			tmp_weights[j] = lexpr_weights[i*expr_size+j];
			tmp_dim[j] = lexpr_dim[i*expr_size+j];
		}
		res->input_lexpr[i] = create_sparse_expr(tmp_weights, lexpr_cst[i], tmp_dim, expr_size);
		sort_sparse_expr(res->input_lexpr[i]);
	//printf("w: %p %g %g %g cst: %g dim: %p %zu %zu %zu\n",lexpr_weights[i],lexpr_weights[i][0],lexpr_weights[i][1], lexpr_weights[i][2],lexpr_cst[i],lexpr_dim[i],lexpr_dim[i][0],lexpr_dim[i][1], lexpr_dim[i][2]);
		//expr_print(res->input_lexpr[i]);
		//fflush(stdout);
		for(j=0; j < expr_size; j++){
			tmp_weights[j] = uexpr_weights[i*expr_size+j];
			tmp_dim[j] = uexpr_dim[i*expr_size+j];
		}
		res->input_uexpr[i] = create_sparse_expr(tmp_weights, uexpr_cst[i], tmp_dim, expr_size);
		sort_sparse_expr(res->input_uexpr[i]);
	//	expr_print(res->input_uexpr[i]);
	//	fflush(stdout);
	}
	free(tmp_weights);
	free(tmp_dim);
	
    res->spatial_size = spatial_size;
    res->spatial_gamma = spatial_gamma;
    res->spatial_indices = malloc(spatial_size * sizeof(size_t));
    res->spatial_neighbors = malloc(spatial_size * sizeof(size_t));
    memcpy(res->spatial_indices, spatial_indices, spatial_size * sizeof(size_t));
    memcpy(res->spatial_neighbors, spatial_neighbors, spatial_size * sizeof(size_t));

    return abstract0_of_fppoly(man,res);	
}

void fppoly_add_new_layer(fppoly_t *fp, size_t size, size_t *predecessors, size_t num_predecessors, bool is_activation){
	size_t numlayers = fp->numlayers;
	if(fp->numlayers==0){
		fp->layers = (layer_t **)malloc(2000*sizeof(layer_t *));
	}
	fp->layers[numlayers] = create_layer(size, is_activation);
	fp->layers[numlayers]->predecessors = predecessors;
	fp->layers[numlayers]->num_predecessors = num_predecessors;
	fp->numlayers++;
	return;
}

void handle_fully_connected_layer_with_backsubstitute(elina_manager_t* man, elina_abstract0_t* element, double **weights, double * cst, size_t num_out_neurons, size_t num_in_neurons, size_t * predecessors, size_t num_predecessors, bool alloc, fnn_op OP, bool layer_by_layer, bool is_residual, bool is_blk_segmentation, int blk_size, bool is_early_terminate, int early_termi_thre, bool is_sum_def_over_input, bool is_refinement){
    assert(num_predecessors==1);
    fppoly_t *fp = fppoly_of_abstract0(element);
    size_t numlayers = fp->numlayers;
    if(alloc){
        fppoly_add_new_layer(fp,num_out_neurons, predecessors, num_predecessors, false);
    }
	// printf("layer with %zu neurons\n",num_out_neurons);
	// printf("predecessor is %zu\n", predecessors[0]);
    neuron_t **out_neurons = fp->layers[numlayers]->neurons;
    size_t i;
    for(i=0; i < num_out_neurons; i++){
	    double cst_i = cst[i];
		if(OP==MUL){
			out_neurons[i]->lexpr = create_sparse_expr(&cst_i, 0, &i, 1);
	        }
		else if(OP==SUB1){
			double coeff = -1;
			out_neurons[i]->lexpr = create_sparse_expr(&coeff, cst_i, &i, 1);
		}
		else if(OP==SUB2){
			double coeff = 1;
			out_neurons[i]->lexpr = create_sparse_expr(&coeff, -cst_i, &i, 1);
		}
	    else{
			double * weight_i = weights[i];
	        	out_neurons[i]->lexpr = create_dense_expr(weight_i,cst_i,num_in_neurons);
		}
		out_neurons[i]->uexpr = out_neurons[i]->lexpr;
		if(layer_by_layer){
			out_neurons[i]->backsubstituted_lexpr = copy_expr(out_neurons[i]->lexpr);
			out_neurons[i]->backsubstituted_uexpr = copy_expr(out_neurons[i]->uexpr);
		}
    }
    if(layer_by_layer){
		update_state_layer_by_layer_parallel(man,fp,numlayers,layer_by_layer, is_residual, is_blk_segmentation, blk_size, is_early_terminate, early_termi_thre, is_sum_def_over_input, is_refinement);
	}
	else{
		// printf("enter layers analysis\n");
		update_state_using_previous_layers_parallel(man,fp,numlayers,layer_by_layer, is_residual, is_blk_segmentation, blk_size, is_early_terminate, early_termi_thre, is_sum_def_over_input, is_refinement);
	}
    return;
}

void handle_sub_layer(elina_manager_t* man, elina_abstract0_t * abs,  double *cst, bool is_minuend, size_t size, size_t *predecessors, size_t num_predecessors, bool layer_by_layer, bool is_residual, bool is_blk_segmentation, int blk_size, bool is_early_terminate, int early_termi_thre, bool is_sum_def_over_input, bool is_refinement){
	if(is_minuend==true){
		handle_fully_connected_layer_with_backsubstitute(man, abs, NULL, cst, size, size, predecessors, num_predecessors, true, SUB1, layer_by_layer,is_residual, is_blk_segmentation, blk_size, is_early_terminate, early_termi_thre, is_sum_def_over_input, is_refinement);
	}
	else{
        	handle_fully_connected_layer_with_backsubstitute(man, abs, NULL, cst, size, size, predecessors, num_predecessors, true, SUB2, layer_by_layer, is_residual, is_blk_segmentation, blk_size, is_early_terminate, early_termi_thre, is_sum_def_over_input, is_refinement);
	}
}

void handle_mul_layer(elina_manager_t* man, elina_abstract0_t * abs, double *bias,  size_t size, size_t *predecessors, size_t num_predecessors, bool layer_by_layer, bool is_residual, bool is_blk_segmentation, int blk_size, bool is_early_terminate, int early_termi_thre, bool is_sum_def_over_input, bool is_refinement){
        handle_fully_connected_layer_with_backsubstitute(man, abs, NULL, bias, size, size, predecessors, num_predecessors, true, MUL, layer_by_layer, is_residual, is_blk_segmentation, blk_size, is_early_terminate, early_termi_thre, is_sum_def_over_input, is_refinement);
}

void handle_fully_connected_layer_no_alloc(elina_manager_t* man, elina_abstract0_t * abs, double **weights, double *bias,   size_t size, size_t num_pixels, size_t *predecessors, size_t num_predecessors, bool layer_by_layer, bool is_residual, bool is_blk_segmentation, int blk_size, bool is_early_terminate, int early_termi_thre, bool is_sum_def_over_input, bool is_refinement){
    handle_fully_connected_layer_with_backsubstitute(man, abs, weights, bias, size, num_pixels, predecessors, num_predecessors, false, MATMULT, layer_by_layer, is_residual, is_blk_segmentation, blk_size, is_early_terminate, early_termi_thre, is_sum_def_over_input, is_refinement);
}

void handle_fully_connected_layer(elina_manager_t* man, elina_abstract0_t * abs, double **weights, double *bias,   size_t size, size_t num_pixels, size_t *predecessors, size_t num_predecessors, bool layer_by_layer, bool is_residual, bool is_blk_segmentation, int blk_size, bool is_early_terminate, int early_termi_thre, bool is_sum_def_over_input, bool is_refinement){
     handle_fully_connected_layer_with_backsubstitute(man, abs, weights, bias, size, num_pixels, predecessors, num_predecessors, true, MATMULT, layer_by_layer, is_residual, is_blk_segmentation, blk_size, is_early_terminate, early_termi_thre, is_sum_def_over_input, is_refinement);
}

void neuron_fprint(FILE * stream, neuron_t *neuron, char ** name_of_dim){
	//expr_fprint(stream,neuron->expr);
	fprintf(stream,"[%g, %g]\n",-neuron->lb,neuron->ub);
}

void layer_fprint(FILE * stream, layer_t * layer, char** name_of_dim){
	size_t dims = layer->dims;
	size_t i;
	for(i = 0; i < dims; i++){
		fprintf(stream,"neuron: %zu ", i);
		neuron_fprint(stream, layer->neurons[i], name_of_dim);
	}
}

void coeff_to_interval(elina_coeff_t *coeff, double *inf, double *sup){
	double d;
	if(coeff->discr==ELINA_COEFF_SCALAR){
		elina_scalar_t * scalar = coeff->val.scalar;
		d = scalar->val.dbl;
		*inf = -d;
		*sup = d;
	}
	else{
		elina_interval_t *interval = coeff->val.interval;
		d = interval->inf->val.dbl;
		*inf = -d;
		d = interval->sup->val.dbl;
		*sup = d;	
	}
		
}

expr_t * elina_linexpr0_to_expr(elina_linexpr0_t *linexpr0){
	size_t size = linexpr0->size;
	size_t i;
	expr_t *res = (expr_t*)malloc(sizeof(expr_t));
	res->inf_coeff = (double*)malloc(size*sizeof(double));
	res->sup_coeff = (double*)malloc(size*sizeof(double));
	res->size = size;
	if(linexpr0->discr==ELINA_LINEXPR_SPARSE){
		res->type = SPARSE;
		res->dim = (size_t *)malloc(size*sizeof(size_t));
	}
	else{
		res->type = DENSE;
		res->dim = NULL;
	}
	size_t k;
	for(i=0; i< size; i++){
		elina_coeff_t *coeff;
		if(res->type==SPARSE){
			k = linexpr0->p.linterm[i].dim;
			res->dim[i] = k;
			coeff = &linexpr0->p.linterm[i].coeff;
			coeff_to_interval(coeff,&res->inf_coeff[i],&res->sup_coeff[i]);
		}
		else{
		 	k = i;
			coeff = &linexpr0->p.coeff[k];	
			coeff_to_interval(coeff,&res->inf_coeff[k],&res->sup_coeff[k]);
		}
		
	}
	elina_coeff_t *cst = &linexpr0->cst;
	coeff_to_interval(cst,&res->inf_cst,&res->sup_cst);
	return res;
}

void *get_upper_bound_for_linexpr0_parallel(void *args){
	nn_thread_t * data = (nn_thread_t *)args;
	elina_manager_t *man = data->man;
	fppoly_t *fp = data->fp;
    fppoly_internal_t *pr = fppoly_init_from_manager(man, ELINA_FUNID_ASSIGN_LINEXPR_ARRAY);
        size_t layerno = data->layerno;
	size_t idx_start = data->start;
	size_t idx_end = data->end;
	elina_linexpr0_t ** linexpr0 = data->linexpr0;
	double * res = data->res;
	size_t i;
	for(i=idx_start; i < idx_end; i++){
		expr_t * tmp = elina_linexpr0_to_expr(linexpr0[i]);
		double ub = compute_ub_from_expr(pr,tmp,fp,layerno);
        	if(linexpr0[i]->size==1){
			res[i] = ub;
			continue;
		}
		expr_t * uexpr = NULL;
		if(fp->layers[layerno]->num_predecessors==2){
			uexpr = copy_expr(tmp);
		}
		else{
			uexpr = uexpr_replace_bounds(pr, tmp,fp->layers[layerno]->neurons, false);
		}
	
		ub = fmin(ub,get_ub_using_previous_layers(man,fp,&uexpr,layerno,false,false,false,0,false, false, false, false));
	
		free_expr(uexpr);
    		free_expr(tmp);
		res[i] = ub;
	}
	return NULL;
}
     
double *get_upper_bound_for_linexpr0(elina_manager_t *man, elina_abstract0_t *element, elina_linexpr0_t **linexpr0, size_t size, size_t layerno){
	fppoly_t * fp = fppoly_of_abstract0(element);
	size_t NUM_THREADS = sysconf(_SC_NPROCESSORS_ONLN);
	nn_thread_t args[NUM_THREADS];
	pthread_t threads[NUM_THREADS];
	size_t i;
	double * res = (double *)malloc(size*sizeof(double));
	if(size < NUM_THREADS){
		for (i = 0; i < size; i++){
			args[i].start = i;
			args[i].end = i+1;
			args[i].man = man;
			args[i].fp = fp;
			args[i].layerno = layerno;
			args[i].linexpr0 = linexpr0;
			args[i].res = res;
			pthread_create(&threads[i], NULL,get_upper_bound_for_linexpr0_parallel, (void*)&args[i]);

	  	}
		for (i = 0; i < size; i = i + 1){
			pthread_join(threads[i], NULL);
		}
	}
	else{
		size_t idx_start = 0;
		size_t idx_n = size / NUM_THREADS;
		size_t idx_end = idx_start + idx_n;
	  	for (i = 0; i < NUM_THREADS; i++){
			args[i].start = idx_start;
			args[i].end = idx_end;
			args[i].man = man;
			args[i].fp = fp;
			args[i].layerno = layerno;
			args[i].linexpr0 = linexpr0;
			args[i].res = res;
			pthread_create(&threads[i], NULL, get_upper_bound_for_linexpr0_parallel, (void*)&args[i]);
			idx_start = idx_end;
			idx_end = idx_start + idx_n;
	    		if(idx_end> size){
				idx_end = size;
			}
			if((i==NUM_THREADS-2)){
				idx_end = size;

			}
	  	}
		for (i = 0; i < NUM_THREADS; i = i + 1){
			pthread_join(threads[i], NULL);
		}
	}
	return res;
}

bool is_greater(elina_manager_t* man, elina_abstract0_t* element, elina_dim_t y, elina_dim_t x, bool layer_by_layer, bool is_residual, bool is_blk_segmentation, int blk_size, bool is_early_terminate, int early_termi_thre, bool is_sum_def_over_input, bool is_refinement){
	fppoly_t *fp = fppoly_of_abstract0(element);
	fppoly_internal_t * pr = fppoly_init_from_manager(man,ELINA_FUNID_ASSIGN_LINEXPR_ARRAY);
	expr_t * sub = (expr_t *)malloc(sizeof(expr_t));
	//sub->size = size;
	sub->inf_cst = 0;
	sub->sup_cst = 0;
	sub->inf_coeff = (double*)malloc(2*sizeof(double));
	sub->sup_coeff = (double*)malloc(2*sizeof(double));
	sub->dim =(size_t *)malloc(2*sizeof(size_t));
	sub->size = 2;
	sub->type = SPARSE;
	sub->inf_coeff[0] = -1;
	sub->sup_coeff[0] = 1;
	sub->dim[0] = y;
	sub->inf_coeff[1] = 1;
	sub->sup_coeff[1] = -1;
	sub->dim[1] = x;
	double lb = INFINITY;
	int k;
	expr_t * backsubstituted_lexpr = copy_expr(sub);
	if(is_blk_segmentation && is_refinement){
		// only apply modular for refinement process, not for original deeppoly execution
		is_blk_segmentation = false;
	}
	// printf("The auxilinary neuron is %zu - %zu\n", y, x);
	if(layer_by_layer){
		k = fp->numlayers - 1;
		while (k >= -1)
		{
			double cur_lb = get_lb_using_prev_layer(man, fp, &backsubstituted_lexpr, k, is_blk_segmentation);
			lb = fmin(lb, cur_lb);
			if (k < 0 || lb < 0)
				break;
			if(is_blk_segmentation && fp->layers[k]->is_end_layer_of_blk){
				k = fp->layers[k]->start_idx_in_same_blk;
			}
			else{
				k = fp->layers[k]->predecessors[0] - 1;
			}
		}
	}
	else{
		lb = get_lb_using_previous_layers(man, fp, &sub, fp->numlayers, layer_by_layer, is_residual, is_blk_segmentation, blk_size, is_early_terminate, early_termi_thre, is_sum_def_over_input, is_refinement);
	}
	if(sub){
		free_expr(sub);
		sub = NULL;
	}
	if(backsubstituted_lexpr){
		free_expr(backsubstituted_lexpr);
		backsubstituted_lexpr = NULL;
	}
	if(lb<0){
		return true;
	}
	else{
		return false;
	}
}

void handle_gurobi_error(int error, GRBenv *env) {
    if (error) {
        printf("Gurobi error: %s\n", GRBgeterrormsg(env));
        exit(1);
    }
}

void* run_deeppoly(elina_manager_t* man, elina_abstract0_t* element){
	fppoly_t *fp = fppoly_of_abstract0(element);
	size_t numlayers = fp->numlayers;
	size_t i, j;
	for (j=0; j < numlayers; j++){
		if(!fp->layers[j]->is_activation){
			// deeppoly run the affine layer
			neuron_t **neurons = fp->layers[j]->neurons;
			for(i=0; i < fp->layers[j]->dims; i++){
				// free before new assignment
				if(neurons[i]->backsubstituted_lexpr){
					free_expr(neurons[i]->backsubstituted_lexpr);
				}
				neurons[i]->backsubstituted_lexpr = copy_expr(neurons[i]->lexpr);
				// expr_print(neurons[i]->backsubstituted_lexpr);
				if(neurons[i]->backsubstituted_uexpr){
					free_expr(neurons[i]->backsubstituted_uexpr);
				}
				neurons[i]->backsubstituted_uexpr = copy_expr(neurons[i]->uexpr);
				// expr_print(neurons[i]->backsubstituted_uexpr);
			}	
			// printf("before update_state_layer_by_layer_parallel\n");
			update_state_layer_by_layer_parallel(man,fp, j, true, false, false, 0, false, 0, false, false);
			// printf("affine 1 's interval is [%.3f, %.3f], affine 2 's interval is [%.3f, %.3f] ", -neurons[0]->lb, neurons[0]->ub, -neurons[1]->lb, neurons[1]->ub);
			// printf("after update_state_layer_by_layer_parallel\n");
		}
		else{
			// deeppoly run the relu layer
			int k = fp->layers[j]->predecessors[0]-1;
			layer_t *predecessor_layer = fp->layers[k];
			neuron_t **in_neurons = fp->layers[k]->neurons;
			neuron_t **out_neurons = fp->layers[j]->neurons;
			for(i=0; i < fp->layers[j]->dims; i++){
				out_neurons[i]->lb = -fmax(0.0, -in_neurons[i]->lb);
				out_neurons[i]->ub = fmax(0,in_neurons[i]->ub);
				if(out_neurons[i]->lexpr){
					free_expr(out_neurons[i]->lexpr);
				}
				out_neurons[i]->lexpr = create_relu_expr(out_neurons[i], in_neurons[i], i, true, true, false);
				if(out_neurons[i]->uexpr){
					free_expr(out_neurons[i]->uexpr);
				}
				out_neurons[i]->uexpr = create_relu_expr(out_neurons[i], in_neurons[i], i, true, false, false);
				// printf("relu %zu 's upper expression is ", i);
				// expr_print(out_neurons[i]->uexpr);
			}
		}
	}
	return NULL;
}

void* run_deeppoly_in_block(elina_manager_t* man, elina_abstract0_t* element, int block_start_layer, int block_end_layer){
	// only execute in modular analysis in the refinement procedure
	fppoly_t *fp = fppoly_of_abstract0(element);
	size_t numlayers = fp->numlayers;
	size_t i, j;
	for (j = block_start_layer+1; j <= block_end_layer; j++){
		if(!fp->layers[j]->is_activation){
			neuron_t **neurons = fp->layers[j]->neurons;
			for(i=0; i < fp->layers[j]->dims; i++){
				// free previous analysis footprint
				if(neurons[i]->backsubstituted_lexpr){
					free_expr(neurons[i]->backsubstituted_lexpr);
				}
				neurons[i]->backsubstituted_lexpr = copy_expr(neurons[i]->lexpr);
				if(neurons[i]->backsubstituted_uexpr){
					free_expr(neurons[i]->backsubstituted_uexpr);
				}
				neurons[i]->backsubstituted_uexpr = copy_expr(neurons[i]->uexpr);
				if(neurons[i]->summary_lexpr){
					free_expr(neurons[i]->summary_lexpr);
					neurons[i]->summary_lexpr = NULL;
				}
				if(neurons[i]->summary_uexpr){
					free_expr(neurons[i]->summary_uexpr);
					neurons[i]->summary_uexpr = NULL;
				}
			}	
			update_state_layer_by_layer_parallel_until_certain_layer(man,fp, j, false, block_start_layer);
		}
		else{
			int k = fp->layers[j]->predecessors[0]-1;
			layer_t *predecessor_layer = fp->layers[k];
			neuron_t **in_neurons = fp->layers[k]->neurons;
			neuron_t **out_neurons = fp->layers[j]->neurons;
			for(i=0; i < fp->layers[j]->dims; i++){
				out_neurons[i]->lb = -fmax(0.0, -in_neurons[i]->lb);
				out_neurons[i]->ub = fmax(0,in_neurons[i]->ub);
				if(out_neurons[i]->lexpr){
					free_expr(out_neurons[i]->lexpr);
				}
				out_neurons[i]->lexpr = create_relu_expr(out_neurons[i], in_neurons[i], i, true, true, false);
				if(out_neurons[i]->uexpr){
					free_expr(out_neurons[i]->uexpr);
				}
				out_neurons[i]->uexpr = create_relu_expr(out_neurons[i], in_neurons[i], i, true, false, false);
			}
		}
	}
	return NULL;
}

void* run_bbpoly_in_block(elina_manager_t* man, elina_abstract0_t* element, int block_start_layer, int block_end_layer){
	// only execute in modular analysis in the refinement procedure
	fppoly_t *fp = fppoly_of_abstract0(element);
	size_t numlayers = fp->numlayers;
	size_t i, j;
	for (j = block_start_layer+1; j <= block_end_layer; j++){
		if(!fp->layers[j]->is_activation){
			neuron_t **neurons = fp->layers[j]->neurons;
			for(i=0; i < fp->layers[j]->dims; i++){
				// free previous analysis footprint
				if(neurons[i]->backsubstituted_lexpr){
					free_expr(neurons[i]->backsubstituted_lexpr);
				}
				neurons[i]->backsubstituted_lexpr = copy_expr(neurons[i]->lexpr);
				if(neurons[i]->backsubstituted_uexpr){
					free_expr(neurons[i]->backsubstituted_uexpr);
				}
				neurons[i]->backsubstituted_uexpr = copy_expr(neurons[i]->uexpr);
				if(neurons[i]->summary_lexpr){
					free_expr(neurons[i]->summary_lexpr);
					neurons[i]->summary_lexpr = NULL;
				}
				if(neurons[i]->summary_uexpr){
					free_expr(neurons[i]->summary_uexpr);
					neurons[i]->summary_uexpr = NULL;
				}
			}	
			update_state_layer_by_layer_parallel_until_certain_layer(man,fp, j, true, block_start_layer);
		}
		else{
			int k = fp->layers[j]->predecessors[0]-1;
			layer_t *predecessor_layer = fp->layers[k];
			neuron_t **in_neurons = fp->layers[k]->neurons;
			neuron_t **out_neurons = fp->layers[j]->neurons;
			for(i=0; i < fp->layers[j]->dims; i++){
				out_neurons[i]->lb = -fmax(0.0, -in_neurons[i]->lb);
				out_neurons[i]->ub = fmax(0,in_neurons[i]->ub);
				if(out_neurons[i]->lexpr){
					free_expr(out_neurons[i]->lexpr);
				}
				out_neurons[i]->lexpr = create_relu_expr(out_neurons[i], in_neurons[i], i, true, true, false);
				if(out_neurons[i]->uexpr){
					free_expr(out_neurons[i]->uexpr);
				}
				out_neurons[i]->uexpr = create_relu_expr(out_neurons[i], in_neurons[i], i, true, false, false);
			}
		}
	}
	return NULL;
}

void* clear_neurons_status(elina_manager_t* man, elina_abstract0_t* element){
	fppoly_t *fp = fppoly_of_abstract0(element);
	size_t i, j;
	for(i = 0; i < fp->numlayers; i++){
		layer_t *layer = fp->layers[i];
		neuron_t ** neurons = layer->neurons;
		for(j = 0; j < layer->dims; j++){
			neurons[j]->lb = INFINITY;
			neurons[j]->ub = INFINITY;
		}
	}
	return NULL;
}

void* clear_block_summary(elina_manager_t* man, elina_abstract0_t* element){
	fppoly_t *fp = fppoly_of_abstract0(element);
	size_t i, j;
	for(i = 0; i < fp->numlayers; i++){
		layer_t *layer = fp->layers[i];
		neuron_t ** neurons = layer->neurons;
		for(j = 0; j < layer->dims; j++){
			if(neurons[j]->summary_lexpr){
				free_expr(neurons[j]->summary_lexpr);
				neurons[j]->summary_lexpr = NULL;
			}
			if(neurons[j]->summary_uexpr){
				free_expr(neurons[j]->summary_uexpr);
				neurons[j]->summary_uexpr = NULL;
			}
		}
	}
	return NULL;
}

bool is_spurious_blk_summary(elina_manager_t* man, elina_abstract0_t* element, elina_dim_t ground_truth_label, elina_dim_t poten_cex, int * spurious_list, int spurious_count, int MAX_ITER){
	// Leverage block summary to do the LP solving
	int count, k;
	size_t i, j, n;
	fppoly_t *fp = fppoly_of_abstract0(element);
    size_t numlayers = fp->numlayers;
	double ulp = ldexpl(1.0,-52);
	int optimstatus;
	// set the status of neurons back to original status
	for(i=0; i < fp->num_pixels; i++){
		fp->input_inf[i] = fp->original_input_inf[i];
		fp->input_sup[i] = fp->original_input_sup[i];
	}
	// for(i=0; i < fp->numlayers; i++){
	// 	printf("The start layer of this layer %zu is %d, is activate %d\n", i, fp->layers[i]->start_idx_in_same_blk, fp->layers[i]->is_activation);
	// }
	clear_neurons_status(man, element);
	clear_block_summary(man, element);
	run_bbpoly_in_block(man, element, -1, numlayers - 1);
	// for(i=0; i < fp->layers[numlayers-1]->dims; i++){
	// 	printf("lb and ub are %.4f, %.4f respectively\n", -fp->layers[numlayers-1]->neurons[i]->lb, fp->layers[numlayers-1]->neurons[i]->ub);
	// }
	// return false;
	k = numlayers - 1;
	while(k>=0){
		if(k == numlayers - 1){
			printf("Handle last block!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n");
			// handle the last block of the network 
			int start_layer_index = fp->layers[numlayers - 1]->start_idx_in_same_blk;
			if(start_layer_index == numlayers -1){
				// if the last block just contain one layer, merge it with the previous block
				start_layer_index = (numlayers >= 2) ? fp->layers[numlayers - 2]->start_idx_in_same_blk : -1;
			}
			// printf("the start_layer_index of last block is %d\n", start_layer_index);
			for(count = 0; count < MAX_ITER; count++){
				// refinement within this block for MAX_ITER times
				printf("Refinement iteration %d\n", count);
				if(count!=0)
					run_bbpoly_in_block(man, element, start_layer_index, numlayers - 1);
				// only run deeppoly within this block
				GRBenv *env   = NULL;
				GRBmodel *model = NULL;
				int error = 0;	
				error = GRBemptyenv(&env);
				handle_gurobi_error(error, env);
				// error = GRBsetstrparam(env, "LogFile", NULL);
				error = GRBsetintparam(env, "OutputFlag", 0);
				handle_gurobi_error(error, env);
				error = GRBstartenv(env);
				handle_gurobi_error(error, env);
				error = GRBnewmodel(env, &model, "refinement_solver", 0, NULL, NULL, NULL, NULL, NULL);
				handle_gurobi_error(error, env);
				int layer_var_start_idx[numlayers];
				if(start_layer_index >=0){
					layer_var_start_idx[start_layer_index] = 0;
					layer_t * start_layer = fp->layers[start_layer_index];
					layer_var_start_idx[start_layer_index + 1] = start_layer->dims;
					for(i=0; i < start_layer->dims; i++){
						error = GRBaddvar(model, 0, NULL, NULL, 0.0, -start_layer->neurons[i]->lb, start_layer->neurons[i]->ub, GRB_CONTINUOUS, NULL);
						handle_gurobi_error(error, env);
					}
				}
				else{
					// The last block is the whole network, then start layer is the input layer
					for(i=0; i < fp->num_pixels; i++){
						error = GRBaddvar(model, 0, NULL, NULL, 0.0, -fp->input_inf[i], fp->input_sup[i], GRB_CONTINUOUS, NULL);
						handle_gurobi_error(error, env);
					}
					layer_var_start_idx[start_layer_index + 1] = fp->num_pixels;
				}
				// Encode other layer constraints
				for(k = start_layer_index + 1; k < numlayers; k++){
					// Record the index starter for variables at different layers
					layer_t * cur_layer = fp->layers[k];
					neuron_t ** cur_neurons = cur_layer->neurons;
					size_t num_cur_neurons = cur_layer->dims;
					if(k+1 < numlayers){
						layer_var_start_idx[k+1] = layer_var_start_idx[k] + num_cur_neurons;
					}
					int defined_var_start_idx;
					if(k==0){
						defined_var_start_idx = 0;
					}
					else{
						defined_var_start_idx = layer_var_start_idx[k-1];
					}
					if(cur_layer->is_activation){
						for(j=0; j < num_cur_neurons; j++){
							// add constraints for each ReLU node
							neuron_t * relu_node = cur_neurons[j];
							if(relu_node->ub == 0.0){
								// stable unactivated relu nodes
								expr_t * relu_expr = relu_node->lexpr;
								assert(relu_expr->type==SPARSE);
								error = GRBaddvar(model, 0, NULL, NULL, 0.0, -relu_node->lb, relu_node->ub, GRB_CONTINUOUS, NULL);
								handle_gurobi_error(error, env);
							}
							else if(relu_node->lb<0.0){
								// stable activated relu nodes
								expr_t * relu_expr = relu_node->lexpr;
								size_t num_pre_neurons = relu_expr->size;
								assert(relu_expr->type==SPARSE);
								assert(num_pre_neurons==1);
								error = GRBaddvar(model, 0, NULL, NULL, 0.0, -relu_node->lb, relu_node->ub, GRB_CONTINUOUS, NULL);
								handle_gurobi_error(error, env);
								int ind[2] = {layer_var_start_idx[k] + j, defined_var_start_idx + j};
								double val[2] = {-1.0 , relu_expr->sup_coeff[0]};
								error = GRBaddconstr(model, 2, ind, val, GRB_EQUAL, relu_expr->inf_cst, NULL);
								handle_gurobi_error(error, env);
							}
							else{
								// unstable relu nodes, add two lower constarints, and also handle FP error for upper constraint
								expr_t * relu_expr = relu_node->uexpr;
								size_t num_pre_neurons = relu_expr->size;
								assert(relu_expr->type==SPARSE);
								assert(num_pre_neurons==1);
								// The lower bound setting already indicate that relu >=0
								error = GRBaddvar(model, 0, NULL, NULL, 0.0, -relu_node->lb, relu_node->ub, GRB_CONTINUOUS, NULL);
								handle_gurobi_error(error, env);
								int ind[2] = {layer_var_start_idx[k] + j, defined_var_start_idx + j};
								double val[2] = {-1.0 , 1.0};
								// add lower bound, y >= x, -y+x <= 0
								error = GRBaddconstr(model, 2, ind, val, GRB_LESS_EQUAL, 0.0, NULL);
								handle_gurobi_error(error, env);
								int ind2[2] = {layer_var_start_idx[k] + j, defined_var_start_idx + j};
								double over_slope = relu_expr->sup_coeff[0]+ ulp;
								double val2[2] = {-1.0, over_slope};
								int pre = cur_layer->predecessors[0]-1;
								double in_lb = fp->layers[pre]->neurons[j]->lb;
								assert(in_lb>=0);
								double over_b = (fabs(in_lb)+ulp)*over_slope + ulp;
								// add upper bound, y <= ax+b, -y+ax >= -b
								error = GRBaddconstr(model, 2, ind2, val2, GRB_GREATER_EQUAL, -over_b, NULL);
								handle_gurobi_error(error, env);
							}
							// update model
							error = GRBupdatemodel(model);
							handle_gurobi_error(error, env);
						}
					}
					else{
						// current layer is affine layer
						for(j=0; j < num_cur_neurons; j++){
							neuron_t * affine_node = cur_neurons[j];
							expr_t * affine_expr = affine_node->lexpr;
							size_t num_pre_neurons = affine_expr->size;
							assert(affine_expr->type==DENSE);
							error = GRBaddvar(model, 0, NULL, NULL, 0.0, -affine_node->lb, affine_node->ub, GRB_CONTINUOUS, NULL);
							handle_gurobi_error(error, env);
							int ind[num_pre_neurons+1];
							double val[num_pre_neurons+1];
							for(n=0; n < num_pre_neurons; n++){
								ind[n] = defined_var_start_idx + n;
								val[n] = affine_expr->sup_coeff[n];
							}
							ind[num_pre_neurons] = layer_var_start_idx[k] + j;
							val[num_pre_neurons] = -1.0;
							error = GRBaddconstr(model, num_pre_neurons+1, ind, val, GRB_EQUAL, affine_expr->inf_cst, NULL);
							handle_gurobi_error(error, env);
							// update model
							error = GRBupdatemodel(model);
							handle_gurobi_error(error, env);
						}
					}
				}
				// Add constraints regarding the current potential counter-label
				int var_start_idx = layer_var_start_idx[numlayers - 1];
				int ind[2] = {var_start_idx+ground_truth_label,var_start_idx+poten_cex};
				double val[2] = {1.0, -1.0};
				error = GRBaddconstr(model, 2, ind, val, GRB_LESS_EQUAL, 0.0, NULL);
				error = GRBupdatemodel(model);
				handle_gurobi_error(error, env);

				// Add constraints regarding previous spurious labels
				for(j=0; j < spurious_count; j++){
					int spu_label = spurious_list[j];
					// we have out[ground_truth_label] - out[spu_label] > 0, for practical concern, we expand to >=
					int var_start_idx = layer_var_start_idx[numlayers - 1];
					int ind[2] = {var_start_idx+ground_truth_label,var_start_idx+spu_label};
					double val[2] = {1.0, -1.0};
					error = GRBaddconstr(model, 2, ind, val, GRB_GREATER_EQUAL, 0.0, NULL);
					handle_gurobi_error(error, env);
				}

				// solving and updatin neuron status
				error = GRBoptimize(model);
				handle_gurobi_error(error, env);
				error = GRBgetintattr(model, GRB_INT_ATTR_STATUS, &optimstatus);
				handle_gurobi_error(error, env);
				if(optimstatus == GRB_INFEASIBLE){
					GRBfreemodel(model);
					GRBfreeenv(env);
					printf("Refine succesfully at last block at %d-th iteration\n", count+1);
					return true;
				}
				// solve for interval of start layer neurons
				size_t start_layer_num_neurons = (start_layer_index >= 0) ? fp->layers[start_layer_index]->dims : fp->num_pixels;
				for(i=0; i < start_layer_num_neurons; i++){
					double solved_lb, solved_ub;
					error = GRBsetdblattrelement(model, "Obj", i, 1.0);
					handle_gurobi_error(error, env);
					// ModelSense, default value 1 indicates minimization, and -1 meaning maximization
					// lower bound solving
					error = GRBsetintattr(model, "ModelSense", 1);
					handle_gurobi_error(error, env);
					error = GRBupdatemodel(model);
					handle_gurobi_error(error, env);
					error = GRBoptimize(model);
					handle_gurobi_error(error, env);
					error = GRBgetdblattr(model, GRB_DBL_ATTR_OBJVAL, &solved_lb);
					handle_gurobi_error(error, env);
					// upper bound solving
					error = GRBsetintattr(model, "ModelSense", -1);
					handle_gurobi_error(error, env);
					error = GRBupdatemodel(model);
					handle_gurobi_error(error, env);
					error = GRBoptimize(model);
					handle_gurobi_error(error, env);
					error = GRBgetdblattr(model, GRB_DBL_ATTR_OBJVAL, &solved_ub);
					handle_gurobi_error(error, env);
					// Update the corresponding input lower and upper bound for next deeppoly execution
					double lp_solving_error = pow(10.0, -6.0) + ulp;
					if(start_layer_index >= 0){
						// printf("The start layer neuron %zu originally has interval [%.4f, %.4f]\n", i, -fp->layers[start_layer_index]->neurons[i]->lb, fp->layers[start_layer_index]->neurons[i]->ub);
						fp->layers[start_layer_index]->neurons[i]->lb = fmin(-(solved_lb - lp_solving_error),fp->layers[start_layer_index]->neurons[i]->lb);
						fp->layers[start_layer_index]->neurons[i]->ub = fmin(solved_ub + lp_solving_error,fp->layers[start_layer_index]->neurons[i]->ub);
						// printf("The start layer neuron %zu was updates to [%.4f, %.4f]\n", i, -fp->layers[start_layer_index]->neurons[i]->lb, fp->layers[start_layer_index]->neurons[i]->ub);
					}
					else{
						fp->input_inf[i] = -(solved_lb - lp_solving_error);
						fp->input_sup[i] = solved_ub + lp_solving_error;
						// printf("The resolved input neuron %zu has interval [%.4f, %.4f]\n", i, -fp->input_inf[i], fp->input_sup[i]);
					}
					// revert obj coeff back to 0
					error = GRBsetdblattrelement(model, "Obj", i, 0.0);
					handle_gurobi_error(error, env);
				}

				// solve for relu interval of unstable relu nodes
				int relu_refine_count = 0;
				for(i = start_layer_index + 1; i < numlayers; i++){
					layer_t * cur_layer = fp->layers[i];
					if(!cur_layer->is_activation && (i < numlayers-1) && fp->layers[i+1]->is_activation){
						layer_t * next_layer = fp->layers[i+1];
						assert(cur_layer->dims == next_layer->dims);
						neuron_t ** relu_neurons = next_layer->neurons;
						for(j=0; j < cur_layer->dims; j++){
							if(relu_neurons[j]->ub!=0.0 && relu_neurons[j]->lb>=0){
								double solved_lb, solved_ub;
								error = GRBsetdblattrelement(model, "Obj", layer_var_start_idx[i]+j, 1.0);
								handle_gurobi_error(error, env);
								// ModelSense, default value 1 indicates minimization, and -1 meaning maximization
								// lower bound solving
								error = GRBsetintattr(model, "ModelSense", 1);
								handle_gurobi_error(error, env);
								error = GRBupdatemodel(model);
								handle_gurobi_error(error, env);
								error = GRBoptimize(model);
								handle_gurobi_error(error, env);
								error = GRBgetdblattr(model, GRB_DBL_ATTR_OBJVAL, &solved_lb);
								handle_gurobi_error(error, env);
								// update relu node lower bound
								double lp_solving_error = pow(10.0, -6.0) + ulp;
								cur_layer->neurons[j]->lb = fmin(-(solved_lb - lp_solving_error), cur_layer->neurons[j]->lb);
								if(cur_layer->neurons[j]->lb<0){
									relu_refine_count ++;
								}
								// upper bound solving
								error = GRBsetintattr(model, "ModelSense", -1);
								handle_gurobi_error(error, env);
								error = GRBupdatemodel(model);
								handle_gurobi_error(error, env);
								error = GRBoptimize(model);
								handle_gurobi_error(error, env);
								error = GRBgetdblattr(model, GRB_DBL_ATTR_OBJVAL, &solved_ub);
								handle_gurobi_error(error, env);
								// update relu node upper bound
								cur_layer->neurons[j]->ub = fmin(solved_ub + lp_solving_error, cur_layer->neurons[j]->ub);
								if(cur_layer->neurons[j]->ub<=0){
									relu_refine_count ++;
								}
								error = GRBsetdblattrelement(model, "Obj", layer_var_start_idx[i]+j, 0.0);
								handle_gurobi_error(error, env);
							}
						}
					}
				}
				// printf("Refreshed ReLU nodes: %d\n",relu_refine_count);
				/* Free model */
				GRBfreemodel(model);
				/* Free environment */
				GRBfreeenv(env);
			}
			k = (start_layer_index >= 0) ? fp->layers[start_layer_index]->predecessors[0]-1 : -2 ;
		}
		else{
			// handle refinement through other blocks, where the encoding comes from block summaries
			int start_layer_index = fp->layers[k]->start_idx_in_same_blk;
			// printf("the start_layer_index of rest block is %d, the end layer is %d\n", start_layer_index, k);
			int ind_next_blk_connection = k + 1;
			printf("Handle block [%d-%d]!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n", start_layer_index, k);
			for(count = 0; count < MAX_ITER; count++){
				printf("Refinement iteration %d\n", count);
				// printf("Current refine iteration is %d\n", count);
				if(count!=0)
					run_bbpoly_in_block(man, element, start_layer_index, k);
				GRBenv *env   = NULL;
				GRBmodel *model = NULL;
				int error = 0;	
				error = GRBemptyenv(&env);
				handle_gurobi_error(error, env);
				error = GRBsetintparam(env, "OutputFlag", 0);
				handle_gurobi_error(error, env);
				error = GRBstartenv(env);
				handle_gurobi_error(error, env);
				error = GRBnewmodel(env, &model, "refinement_solver", 0, NULL, NULL, NULL, NULL, NULL);
				handle_gurobi_error(error, env);
				// add the vars of start layer neurons
				int end_var_start_ind;
				if(start_layer_index >=0){
					layer_t * start_layer = fp->layers[start_layer_index];
					end_var_start_ind = start_layer->dims;
					for(i=0; i < start_layer->dims; i++){
						error = GRBaddvar(model, 0, NULL, NULL, 0.0, -start_layer->neurons[i]->lb, start_layer->neurons[i]->ub, GRB_CONTINUOUS, NULL);
						handle_gurobi_error(error, env);
					}
				}
				else{
					// printf("should enter here, the start layer is the input layer\n");
					end_var_start_ind = fp->num_pixels;
					for(i=0; i < fp->num_pixels; i++){ // the start layer is the input layer
						error = GRBaddvar(model, 0, NULL, NULL, 0.0, -fp->input_inf[i], fp->input_sup[i], GRB_CONTINUOUS, NULL);
						handle_gurobi_error(error, env);
					}
				}
				// add the vars of end layer neurons, where end layer is affine layer according to our segmentation mechanism
				assert(!fp->layers[k]->is_activation); 
				layer_t * end_layer = fp->layers[k];
				neuron_t ** end_neurons = end_layer->neurons;
				assert(end_layer->dims == fp->layers[ind_next_blk_connection]->dims);
				// for(j = 0; j < end_layer->dims; j++){
				// 	printf("The block summary for neuron %zu is:\n", j);
				// 	expr_print(end_neurons[j]->summary_lexpr);
				// 	expr_print(end_neurons[j]->summary_uexpr);
				// }
				for(j = 0; j < end_layer->dims; j++){
					neuron_t * end_node = end_neurons[j];
					expr_t * summary_lexpr = end_node->summary_lexpr;
					expr_t * summary_uexpr = end_node->summary_uexpr;
					assert(summary_lexpr->type==DENSE);
					size_t num_pre_neurons = summary_lexpr->size;
					error = GRBaddvar(model, 0, NULL, NULL, 0.0, -end_node->lb, end_node->ub, GRB_CONTINUOUS, NULL);
					handle_gurobi_error(error, env);
					// For upper summary, need to encode the c+ coefficient
					int ind[num_pre_neurons+1];
  					double val[num_pre_neurons+1];
					for(n=0; n < num_pre_neurons; n++){
						ind[n] = n;
						val[n] = summary_uexpr->sup_coeff[n];
					}
					ind[num_pre_neurons] = end_var_start_ind + j;
					val[num_pre_neurons] = -1.0;
					// y <= ax+b  -> ax - y >= -b
					error = GRBaddconstr(model, num_pre_neurons+1, ind, val, GRB_GREATER_EQUAL, -summary_uexpr->sup_cst , NULL);
					handle_gurobi_error(error, env);
					// For lower summary, need to encode the c- coefficient
  					double val2[num_pre_neurons+1];
					for(n=0; n < num_pre_neurons; n++){
						val2[n] = -summary_lexpr->inf_coeff[n];
					}
					val2[num_pre_neurons] = -1.0;
					// y >= ax+b  -> ax - y <= -b
					error = GRBaddconstr(model, num_pre_neurons+1, ind, val2, GRB_LESS_EQUAL, summary_lexpr->inf_cst, NULL);
					handle_gurobi_error(error, env);
					// update model
					error = GRBupdatemodel(model);
					handle_gurobi_error(error, env);
				}

				// add the vars of next start layer neurons, which is Relu connected to the end layer
				int next_start_var_ind = end_var_start_ind + end_layer->dims;
				layer_t * start_layer_of_next_blk = fp->layers[ind_next_blk_connection];
				neuron_t ** relu_neurons = start_layer_of_next_blk->neurons;
				for(j=0; j < start_layer_of_next_blk->dims; j++){
					neuron_t * relu_node = relu_neurons[j];
					neuron_t * input_node = end_neurons[j];
					// relu interval constaint corresponds to output neuron constraints
					error = GRBaddvar(model, 0, NULL, NULL, 0.0, -relu_node->lb, relu_node->ub, GRB_CONTINUOUS, NULL);
					handle_gurobi_error(error, env);
					// the connection between, depends on the interval of end_layer
					if(input_node->ub<=0){
						// stable unactivated relu nodes
						int ind = next_start_var_ind + j;
						double val = 1.0;
						error = GRBaddconstr(model, 1, &ind, &val, GRB_EQUAL, 0.0, NULL);
						handle_gurobi_error(error, env);
					}
					else if(input_node->lb<0){
						// stable activated relu nodes, y = x, -y +x = 0
						int ind[2] = {next_start_var_ind + j, end_var_start_ind + j};
						double val[2] = {-1.0 , 1.0};
						error = GRBaddconstr(model, 2, ind, val, GRB_EQUAL, 0.0, NULL);
						handle_gurobi_error(error, env);
					}
					else{
						// unstable relu nodes, add two lower constarints, and also handle FP error for upper constraint
						// add lower constriant that relu >=0
						int ind = next_start_var_ind + j;
						double val = 1.0;
						error = GRBaddconstr(model, 1, &ind, &val, GRB_GREATER_EQUAL, 0.0, NULL);
						handle_gurobi_error(error, env);
						// add lower constraint, y >= x, -y+x <= 0
						int ind3[2] = {next_start_var_ind + j, end_var_start_ind + j};
						double val3[2] = {-1.0 , 1.0};
						error = GRBaddconstr(model, 2, ind3, val3, GRB_LESS_EQUAL, 0.0, NULL);
						handle_gurobi_error(error, env);
						// add upper bound, y <= ax+b, -y+ax >= -b
						int ind2[2] = {next_start_var_ind + j, end_var_start_ind + j};
						double in_lb = input_node->lb;
						assert(in_lb>=0);
						double over_slope = input_node->ub/(input_node->ub+in_lb) + ulp;
						double val2[2] = {-1.0, over_slope};
						double over_b = (fabs(in_lb)+ulp)*over_slope + ulp;
						error = GRBaddconstr(model, 2, ind2, val2, GRB_GREATER_EQUAL, -over_b, NULL);
						handle_gurobi_error(error, env);
					}
					// update model
					error = GRBupdatemodel(model);
					handle_gurobi_error(error, env);
				}

				// solving and updatin neuron status
				error = GRBoptimize(model);
				handle_gurobi_error(error, env);
				error = GRBgetintattr(model, GRB_INT_ATTR_STATUS, &optimstatus);
				handle_gurobi_error(error, env);
				if(optimstatus == GRB_INFEASIBLE){
					GRBfreemodel(model);
					GRBfreeenv(env);
					return true;
				}
				// solve for interval of start layer neurons
				size_t start_layer_num_neurons = (start_layer_index >= 0) ? fp->layers[start_layer_index]->dims : fp->num_pixels;
				for(i=0; i < start_layer_num_neurons; i++){
					double solved_lb, solved_ub;
					error = GRBsetdblattrelement(model, "Obj", i, 1.0);
					handle_gurobi_error(error, env);
					// ModelSense, default value 1 indicates minimization, and -1 meaning maximization
					// lower bound solving
					error = GRBsetintattr(model, "ModelSense", 1);
					handle_gurobi_error(error, env);
					error = GRBupdatemodel(model);
					handle_gurobi_error(error, env);
					error = GRBoptimize(model);
					handle_gurobi_error(error, env);
					error = GRBgetdblattr(model, GRB_DBL_ATTR_OBJVAL, &solved_lb);
					handle_gurobi_error(error, env);
					// upper bound solving
					error = GRBsetintattr(model, "ModelSense", -1);
					handle_gurobi_error(error, env);
					error = GRBupdatemodel(model);
					handle_gurobi_error(error, env);
					error = GRBoptimize(model);
					handle_gurobi_error(error, env);
					error = GRBgetdblattr(model, GRB_DBL_ATTR_OBJVAL, &solved_ub);
					handle_gurobi_error(error, env);
					// Update the corresponding input lower and upper bound for next deeppoly execution
					double lp_solving_error = pow(10.0, -6.0) + ulp;
					if(start_layer_index >= 0){
						// printf("The start layer neuron %zu originally has interval [%.4f, %.4f]\n", i, -fp->layers[start_layer_index]->neurons[i]->lb, fp->layers[start_layer_index]->neurons[i]->ub);
						fp->layers[start_layer_index]->neurons[i]->lb = fmin(-(solved_lb - lp_solving_error),fp->layers[start_layer_index]->neurons[i]->lb);
						fp->layers[start_layer_index]->neurons[i]->ub = fmin(solved_ub + lp_solving_error,fp->layers[start_layer_index]->neurons[i]->ub);
						// printf("The start layer neuron %zu was updates to [%.4f, %.4f]\n", i, -fp->layers[start_layer_index]->neurons[i]->lb, fp->layers[start_layer_index]->neurons[i]->ub);
					}
					else{
						fp->input_inf[i] = -(solved_lb - lp_solving_error);
						fp->input_sup[i] = solved_ub + lp_solving_error;
						// printf("The resolved input neuron %zu has interval [%.4f, %.4f]\n", i, -fp->input_inf[i], fp->input_sup[i]);
					}
					// revert obj coeff back to 0
					error = GRBsetdblattrelement(model, "Obj", i, 0.0);
					handle_gurobi_error(error, env);
				}

				// The end layer of the block is the input to relu layer, use LP solving to stablize it
				int relu_refine_count = 0;
				if(!end_layer->is_activation && start_layer_of_next_blk->is_activation){
					neuron_t ** relu_neurons = start_layer_of_next_blk->neurons;
					for(j=0; j < start_layer_of_next_blk->dims; j++){
						if(relu_neurons[j]->ub!=0.0 && relu_neurons[j]->lb>=0){
							// printf("testing for stablization\n");
							double solved_lb, solved_ub;
							error = GRBsetdblattrelement(model, "Obj", end_var_start_ind + j, 1.0);
							handle_gurobi_error(error, env);
							// ModelSense, default value 1 indicates minimization, and -1 meaning maximization
							// lower bound solving
							error = GRBsetintattr(model, "ModelSense", 1);
							handle_gurobi_error(error, env);
							error = GRBupdatemodel(model);
							handle_gurobi_error(error, env);
							error = GRBoptimize(model);
							handle_gurobi_error(error, env);
							error = GRBgetdblattr(model, GRB_DBL_ATTR_OBJVAL, &solved_lb);
							handle_gurobi_error(error, env);
							// update relu node lower bound
							double lp_solving_error = pow(10.0, -6.0) + ulp;
							end_layer->neurons[j]->lb = fmin(-(solved_lb - lp_solving_error), end_layer->neurons[j]->lb);
							if(end_layer->neurons[j]->lb<0){
								relu_refine_count ++;
							}
							// upper bound solving
							error = GRBsetintattr(model, "ModelSense", -1);
							handle_gurobi_error(error, env);
							error = GRBupdatemodel(model);
							handle_gurobi_error(error, env);
							error = GRBoptimize(model);
							handle_gurobi_error(error, env);
							error = GRBgetdblattr(model, GRB_DBL_ATTR_OBJVAL, &solved_ub);
							handle_gurobi_error(error, env);
							// update relu node upper bound
							end_layer->neurons[j]->ub = fmin(solved_ub + lp_solving_error, end_layer->neurons[j]->ub);
							if(end_layer->neurons[j]->ub<=0){
								relu_refine_count ++;
							}
							error = GRBsetdblattrelement(model, "Obj", end_var_start_ind + j, 0.0);
							handle_gurobi_error(error, env);
						}
					}
				}
				// printf("Refreshed ReLU nodes: %d\n",relu_refine_count);
				/* Free model */
				GRBfreemodel(model);
				/* Free environment */
				GRBfreeenv(env);
			}
			k = (start_layer_index >= 0) ? fp->layers[start_layer_index]->predecessors[0]-1 : -2 ;
		}
	}
	return false;
}

bool is_spurious_modular(elina_manager_t* man, elina_abstract0_t* element, elina_dim_t ground_truth_label, elina_dim_t poten_cex, int * spurious_list, int spurious_count, int MAX_ITER){
	// only analysis in a modular way, but do not use block summary, still encode all the constraints within a block
	int count, k;
	int total_ite = 0;
	size_t i, j, n;
	fppoly_t *fp = fppoly_of_abstract0(element);
    size_t numlayers = fp->numlayers;
	double ulp = ldexpl(1.0,-52);
	int optimstatus;
	clock_t func_begin = clock();
	// set the status of neurons back to original status
	for(i=0; i < fp->num_pixels; i++){
		fp->input_inf[i] = fp->original_input_inf[i];
		fp->input_sup[i] = fp->original_input_sup[i];
	}
	// for(i=0; i < fp->numlayers; i++){
	// 	printf("The start layer of this layer %zu is %d, is activate %d\n", i, fp->layers[i]->start_idx_in_same_blk, fp->layers[i]->is_activation);
	// }
	// clock_t dp_start = clock();
	clear_neurons_status(man, element);
	clear_block_summary(man, element);
	run_deeppoly_in_block(man, element, -1, numlayers - 1);
	// clock_t dp_end = clock();
	// double dp_spent = (double)(dp_end - dp_start) / CLOCKS_PER_SEC;
	// printf("total time to run dp is %f\n", dp_spent);
	// for(i=0; i < fp->layers[numlayers-1]->dims; i++){
	// 	printf("lb and ub are %.4f, %.4f respectively\n", -fp->layers[numlayers-1]->neurons[i]->lb, fp->layers[numlayers-1]->neurons[i]->ub);
	// }
	// return false;
	k = numlayers - 1;
	while(k>=0){
		if(k == numlayers - 1){
			// printf("Handle last block!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n");
			clock_t blk3_begin = clock();
			// handle the last block of the network 
			int start_layer_index = fp->layers[numlayers - 1]->start_idx_in_same_blk;
			if(start_layer_index == numlayers -1){
				// if the last block just contain one layer, merge it with the previous block
				start_layer_index = (numlayers >= 2) ? fp->layers[numlayers - 2]->start_idx_in_same_blk : -1;
			}
			// printf("the start_layer_index of last block is %d\n", start_layer_index);
			for(count = 0; count < MAX_ITER; count++){
				clock_t ite_begin = clock();
				total_ite++;
				// refinement within this block for MAX_ITER times
				if(count!=0)
					run_deeppoly_in_block(man, element, start_layer_index, numlayers - 1);
				// only run deeppoly within this block
				clock_t buildmodel_start = clock();
				GRBenv *env   = NULL;
				GRBmodel *model = NULL;
				int error = 0;	
				error = GRBemptyenv(&env);
				handle_gurobi_error(error, env);
				// error = GRBsetstrparam(env, "LogFile", NULL);
				error = GRBsetintparam(env, "OutputFlag", 0);
				handle_gurobi_error(error, env);
				error = GRBstartenv(env);
				handle_gurobi_error(error, env);
				error = GRBnewmodel(env, &model, "refinement_solver", 0, NULL, NULL, NULL, NULL, NULL);
				handle_gurobi_error(error, env);
				int layer_var_start_idx[numlayers];
				if(start_layer_index >=0){
					layer_var_start_idx[start_layer_index] = 0;
					layer_t * start_layer = fp->layers[start_layer_index];
					layer_var_start_idx[start_layer_index + 1] = start_layer->dims;
					for(i=0; i < start_layer->dims; i++){
						error = GRBaddvar(model, 0, NULL, NULL, 0.0, -start_layer->neurons[i]->lb, start_layer->neurons[i]->ub, GRB_CONTINUOUS, NULL);
						handle_gurobi_error(error, env);
					}
				}
				else{
					// The last block is the whole network, then start layer is the input layer
					for(i=0; i < fp->num_pixels; i++){
						error = GRBaddvar(model, 0, NULL, NULL, 0.0, -fp->input_inf[i], fp->input_sup[i], GRB_CONTINUOUS, NULL);
						handle_gurobi_error(error, env);
					}
					layer_var_start_idx[start_layer_index + 1] = fp->num_pixels;
				}
				// Encode other layer constraints
				for(k = start_layer_index + 1; k < numlayers; k++){
					// Record the index starter for variables at different layers
					layer_t * cur_layer = fp->layers[k];
					neuron_t ** cur_neurons = cur_layer->neurons;
					size_t num_cur_neurons = cur_layer->dims;
					if(k+1 < numlayers){
						layer_var_start_idx[k+1] = layer_var_start_idx[k] + num_cur_neurons;
					}
					int defined_var_start_idx;
					if(k==0){
						defined_var_start_idx = 0;
					}
					else{
						defined_var_start_idx = layer_var_start_idx[k-1];
					}
					if(cur_layer->is_activation){
						for(j=0; j < num_cur_neurons; j++){
							// add constraints for each ReLU node
							neuron_t * relu_node = cur_neurons[j];
							if(relu_node->ub == 0.0){
								// stable unactivated relu nodes
								expr_t * relu_expr = relu_node->lexpr;
								assert(relu_expr->type==SPARSE);
								error = GRBaddvar(model, 0, NULL, NULL, 0.0, -relu_node->lb, relu_node->ub, GRB_CONTINUOUS, NULL);
								handle_gurobi_error(error, env);
							}
							else if(relu_node->lb<0.0){
								// stable activated relu nodes
								expr_t * relu_expr = relu_node->lexpr;
								size_t num_pre_neurons = relu_expr->size;
								assert(relu_expr->type==SPARSE);
								assert(num_pre_neurons==1);
								error = GRBaddvar(model, 0, NULL, NULL, 0.0, -relu_node->lb, relu_node->ub, GRB_CONTINUOUS, NULL);
								handle_gurobi_error(error, env);
								int ind[2] = {layer_var_start_idx[k] + j, defined_var_start_idx + j};
								double val[2] = {-1.0 , relu_expr->sup_coeff[0]};
								error = GRBaddconstr(model, 2, ind, val, GRB_EQUAL, relu_expr->inf_cst, NULL);
								handle_gurobi_error(error, env);
							}
							else{
								// unstable relu nodes, add two lower constarints, and also handle FP error for upper constraint
								expr_t * relu_expr = relu_node->uexpr;
								size_t num_pre_neurons = relu_expr->size;
								assert(relu_expr->type==SPARSE);
								assert(num_pre_neurons==1);
								// The lower bound setting already indicate that relu >=0
								error = GRBaddvar(model, 0, NULL, NULL, 0.0, -relu_node->lb, relu_node->ub, GRB_CONTINUOUS, NULL);
								handle_gurobi_error(error, env);
								int ind[2] = {layer_var_start_idx[k] + j, defined_var_start_idx + j};
								double val[2] = {-1.0 , 1.0};
								// add lower bound, y >= x, -y+x <= 0
								error = GRBaddconstr(model, 2, ind, val, GRB_LESS_EQUAL, 0.0, NULL);
								handle_gurobi_error(error, env);
								int ind2[2] = {layer_var_start_idx[k] + j, defined_var_start_idx + j};
								double over_slope = relu_expr->sup_coeff[0]+ ulp;
								double val2[2] = {-1.0, over_slope};
								int pre = cur_layer->predecessors[0]-1;
								double in_lb = fp->layers[pre]->neurons[j]->lb;
								assert(in_lb>=0);
								double over_b = (fabs(in_lb)+ulp)*over_slope + ulp;
								// add upper bound, y <= ax+b, -y+ax >= -b
								error = GRBaddconstr(model, 2, ind2, val2, GRB_GREATER_EQUAL, -over_b, NULL);
								handle_gurobi_error(error, env);
							}
							// update model
							error = GRBupdatemodel(model);
							handle_gurobi_error(error, env);
						}
					}
					else{
						// current layer is affine layer
						for(j=0; j < num_cur_neurons; j++){
							neuron_t * affine_node = cur_neurons[j];
							expr_t * affine_expr = affine_node->lexpr;
							size_t num_pre_neurons = affine_expr->size;
							assert(affine_expr->type==DENSE);
							error = GRBaddvar(model, 0, NULL, NULL, 0.0, -affine_node->lb, affine_node->ub, GRB_CONTINUOUS, NULL);
							handle_gurobi_error(error, env);
							int ind[num_pre_neurons+1];
							double val[num_pre_neurons+1];
							for(n=0; n < num_pre_neurons; n++){
								ind[n] = defined_var_start_idx + n;
								val[n] = affine_expr->sup_coeff[n];
							}
							ind[num_pre_neurons] = layer_var_start_idx[k] + j;
							val[num_pre_neurons] = -1.0;
							error = GRBaddconstr(model, num_pre_neurons+1, ind, val, GRB_EQUAL, affine_expr->inf_cst, NULL);
							handle_gurobi_error(error, env);
							// update model
							error = GRBupdatemodel(model);
							handle_gurobi_error(error, env);
						}
					}
				}
				// Add constraints regarding the current potential counter-label
				int var_start_idx = layer_var_start_idx[numlayers - 1];
				int ind[2] = {var_start_idx+ground_truth_label,var_start_idx+poten_cex};
				double val[2] = {1.0, -1.0};
				error = GRBaddconstr(model, 2, ind, val, GRB_EQUAL, 0.0, NULL);
				error = GRBupdatemodel(model);
				handle_gurobi_error(error, env);

				// Add constraints regarding previous spurious labels
				for(j=0; j < spurious_count; j++){
					int spu_label = spurious_list[j];
					// we have out[ground_truth_label] - out[spu_label] > 0, for practical concern, we expand to >=
					int var_start_idx = layer_var_start_idx[numlayers - 1];
					int ind[2] = {var_start_idx+ground_truth_label,var_start_idx+spu_label};
					double val[2] = {1.0, -1.0};
					error = GRBaddconstr(model, 2, ind, val, GRB_GREATER_EQUAL, 0.0, NULL);
					handle_gurobi_error(error, env);
				}

				// solving and updatin neuron status
				// clock_t feasiblity_start = clock();
				// double buildmodel_spent = (double)(feasiblity_start - buildmodel_start) / CLOCKS_PER_SEC;
				// printf("total time to to build up the constraint is %f\n", buildmodel_spent);
				error = GRBoptimize(model);
				handle_gurobi_error(error, env);
				error = GRBgetintattr(model, GRB_INT_ATTR_STATUS, &optimstatus);
				handle_gurobi_error(error, env);
				// clock_t feasiblity_end = clock();
				// double feasiblity_spent = (double)(feasiblity_end - feasiblity_start) / CLOCKS_PER_SEC;
				// printf("total time to check feasiblity is %f\n", feasiblity_spent);
				if(optimstatus == GRB_INFEASIBLE){
					GRBfreemodel(model);
					GRBfreeenv(env);
					clock_t func_end = clock();
					double func_spent = (double)(func_end - func_begin) / CLOCKS_PER_SEC;
					printf("Refine succesfully at last block at %d-th iteration, # total iteration is %d, total time is %f\n", count+1, total_ite, func_spent);
					return true;
				}
				// solve for interval of start layer neurons
				size_t start_layer_num_neurons = (start_layer_index >= 0) ? fp->layers[start_layer_index]->dims : fp->num_pixels;
				clock_t LP_begin = clock();
				for(i=0; i < start_layer_num_neurons; i++){
					double solved_lb, solved_ub;
					error = GRBsetdblattrelement(model, "Obj", i, 1.0);
					handle_gurobi_error(error, env);
					// ModelSense, default value 1 indicates minimization, and -1 meaning maximization
					// lower bound solving
					error = GRBsetintattr(model, "ModelSense", 1);
					handle_gurobi_error(error, env);
					error = GRBupdatemodel(model);
					handle_gurobi_error(error, env);
					error = GRBoptimize(model);
					handle_gurobi_error(error, env);
					error = GRBgetdblattr(model, GRB_DBL_ATTR_OBJVAL, &solved_lb);
					handle_gurobi_error(error, env);
					// upper bound solving
					error = GRBsetintattr(model, "ModelSense", -1);
					handle_gurobi_error(error, env);
					error = GRBupdatemodel(model);
					handle_gurobi_error(error, env);
					error = GRBoptimize(model);
					handle_gurobi_error(error, env);
					error = GRBgetdblattr(model, GRB_DBL_ATTR_OBJVAL, &solved_ub);
					handle_gurobi_error(error, env);
					// Update the corresponding input lower and upper bound for next deeppoly execution
					double lp_solving_error = pow(10.0, -6.0) + ulp;
					if(start_layer_index >= 0){
						// printf("The start layer neuron %zu originally has interval [%.4f, %.4f]\n", i, -fp->layers[start_layer_index]->neurons[i]->lb, fp->layers[start_layer_index]->neurons[i]->ub);
						fp->layers[start_layer_index]->neurons[i]->lb = fmin(-(solved_lb - lp_solving_error),fp->layers[start_layer_index]->neurons[i]->lb);
						fp->layers[start_layer_index]->neurons[i]->ub = fmin(solved_ub + lp_solving_error,fp->layers[start_layer_index]->neurons[i]->ub);
						// printf("The start layer neuron %zu was updates to [%.4f, %.4f]\n", i, -fp->layers[start_layer_index]->neurons[i]->lb, fp->layers[start_layer_index]->neurons[i]->ub);
					}
					else{
						fp->input_inf[i] = -(solved_lb - lp_solving_error);
						fp->input_sup[i] = solved_ub + lp_solving_error;
						// printf("The resolved input neuron %zu has interval [%.4f, %.4f]\n", i, -fp->input_inf[i], fp->input_sup[i]);
					}
					// revert obj coeff back to 0
					error = GRBsetdblattrelement(model, "Obj", i, 0.0);
					handle_gurobi_error(error, env);
				}
				// clock_t LP_end = clock();
				// double LP_time_spent = (double)(LP_end - LP_begin) / CLOCKS_PER_SEC;
				// printf("Average LP solving time for each neuron is %f seconds\n", LP_time_spent/start_layer_num_neurons);	
				// solve for relu interval of unstable relu nodes
				int relu_refine_count = 0;
				for(i = start_layer_index + 1; i < numlayers; i++){
					layer_t * cur_layer = fp->layers[i];
					if(!cur_layer->is_activation && (i < numlayers-1) && fp->layers[i+1]->is_activation){
						layer_t * next_layer = fp->layers[i+1];
						assert(cur_layer->dims == next_layer->dims);
						neuron_t ** relu_neurons = next_layer->neurons;
						for(j=0; j < cur_layer->dims; j++){
							if(relu_neurons[j]->ub!=0.0 && relu_neurons[j]->lb>=0){
								double solved_lb, solved_ub;
								error = GRBsetdblattrelement(model, "Obj", layer_var_start_idx[i]+j, 1.0);
								handle_gurobi_error(error, env);
								// ModelSense, default value 1 indicates minimization, and -1 meaning maximization
								// lower bound solving
								error = GRBsetintattr(model, "ModelSense", 1);
								handle_gurobi_error(error, env);
								error = GRBupdatemodel(model);
								handle_gurobi_error(error, env);
								error = GRBoptimize(model);
								handle_gurobi_error(error, env);
								error = GRBgetdblattr(model, GRB_DBL_ATTR_OBJVAL, &solved_lb);
								handle_gurobi_error(error, env);
								// update relu node lower bound
								double lp_solving_error = pow(10.0, -6.0) + ulp;
								cur_layer->neurons[j]->lb = fmin(-(solved_lb - lp_solving_error), cur_layer->neurons[j]->lb);
								if(cur_layer->neurons[j]->lb<0){
									relu_refine_count ++;
								}
								// upper bound solving
								error = GRBsetintattr(model, "ModelSense", -1);
								handle_gurobi_error(error, env);
								error = GRBupdatemodel(model);
								handle_gurobi_error(error, env);
								error = GRBoptimize(model);
								handle_gurobi_error(error, env);
								error = GRBgetdblattr(model, GRB_DBL_ATTR_OBJVAL, &solved_ub);
								handle_gurobi_error(error, env);
								// update relu node upper bound
								cur_layer->neurons[j]->ub = fmin(solved_ub + lp_solving_error, cur_layer->neurons[j]->ub);
								if(cur_layer->neurons[j]->ub<=0){
									relu_refine_count ++;
								}
								error = GRBsetdblattrelement(model, "Obj", layer_var_start_idx[i]+j, 0.0);
								handle_gurobi_error(error, env);
							}
						}
					}
				}
				// printf("Refreshed ReLU nodes: %d\n",relu_refine_count);
				/* Free model */
				GRBfreemodel(model);
				/* Free environment */
				GRBfreeenv(env);
				// clock_t ite_end = clock();
				// double ite_time_spent = (double)(ite_end - ite_begin) / CLOCKS_PER_SEC;
				// printf("This iteration took %f seconds to execute\n", ite_time_spent);
			}
			k = (start_layer_index >= 0) ? fp->layers[start_layer_index]->predecessors[0]-1 : -2;
			// clock_t blk3_end = clock();
			// double blk3_time_spent = (double)(blk3_end - blk3_begin) / CLOCKS_PER_SEC;
			// printf("Last block took %f seconds to execute \n", blk3_time_spent);
		}
		else{
			// handle refinement through other blocks, where the encoding comes from block summaries
			clock_t blk_begin = clock();
			int end_layer_index = k;
			int start_layer_index = fp->layers[end_layer_index]->start_idx_in_same_blk;
			// printf("Handle block [%d-%d]!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n", start_layer_index, end_layer_index);
			// printf("the start_layer_index of rest block is %d, the end layer is %d\n", start_layer_index, end_layer_index);
			int ind_next_blk_connection = end_layer_index + 1;
			for(count = 0; count < MAX_ITER; count++){
				clock_t ite_begin = clock();
				total_ite++;
				// printf("Current refine iteration is %d\n", count);
				if(count!=0)
					run_deeppoly_in_block(man, element, start_layer_index, end_layer_index);
				GRBenv *env   = NULL;
				GRBmodel *model = NULL;
				int error = 0;	
				error = GRBemptyenv(&env);
				handle_gurobi_error(error, env);
				error = GRBsetintparam(env, "OutputFlag", 0);
				handle_gurobi_error(error, env);
				error = GRBstartenv(env);
				handle_gurobi_error(error, env);
				error = GRBnewmodel(env, &model, "refinement_solver", 0, NULL, NULL, NULL, NULL, NULL);
				handle_gurobi_error(error, env);
				// add the vars of start layer neurons
				int layer_var_start_idx[numlayers];
				if(start_layer_index >=0){
					layer_t * start_layer = fp->layers[start_layer_index];
					layer_var_start_idx[start_layer_index] = 0;
					layer_var_start_idx[start_layer_index + 1] = start_layer->dims;
					for(i=0; i < start_layer->dims; i++){
						error = GRBaddvar(model, 0, NULL, NULL, 0.0, -start_layer->neurons[i]->lb, start_layer->neurons[i]->ub, GRB_CONTINUOUS, NULL);
						handle_gurobi_error(error, env);
					}
				}
				else{
					// printf("should enter here, the start layer is the input layer\n");
					layer_var_start_idx[start_layer_index + 1] = fp->num_pixels;
					for(i=0; i < fp->num_pixels; i++){ // the start layer is the input layer
						error = GRBaddvar(model, 0, NULL, NULL, 0.0, -fp->input_inf[i], fp->input_sup[i], GRB_CONTINUOUS, NULL);
						handle_gurobi_error(error, env);
					}
				}
				// add the vars of start+1 until end layer, where end layer is affine layer according to our segmentation mechanism
				for(k = start_layer_index + 1; k <= end_layer_index; k++){
					// Record the index starter for variables at different layers
					layer_t * cur_layer = fp->layers[k];
					neuron_t ** cur_neurons = cur_layer->neurons;
					size_t num_cur_neurons = cur_layer->dims;
					if(k+1 < numlayers){
						layer_var_start_idx[k+1] = layer_var_start_idx[k] + num_cur_neurons;
					}
					int defined_var_start_idx;
					if(k==0){
						defined_var_start_idx = 0;
					}
					else{
						defined_var_start_idx = layer_var_start_idx[k-1];
					}
					if(cur_layer->is_activation){
						for(j=0; j < num_cur_neurons; j++){
							// add constraints for each ReLU node
							neuron_t * relu_node = cur_neurons[j];
							if(relu_node->ub == 0.0){
								// stable unactivated relu nodes
								expr_t * relu_expr = relu_node->lexpr;
								assert(relu_expr->type==SPARSE);
								error = GRBaddvar(model, 0, NULL, NULL, 0.0, -relu_node->lb, relu_node->ub, GRB_CONTINUOUS, NULL);
								handle_gurobi_error(error, env);
							}
							else if(relu_node->lb<0.0){
								// stable activated relu nodes
								expr_t * relu_expr = relu_node->lexpr;
								size_t num_pre_neurons = relu_expr->size;
								assert(relu_expr->type==SPARSE);
								assert(num_pre_neurons==1);
								error = GRBaddvar(model, 0, NULL, NULL, 0.0, -relu_node->lb, relu_node->ub, GRB_CONTINUOUS, NULL);
								handle_gurobi_error(error, env);
								int ind[2] = {layer_var_start_idx[k] + j, defined_var_start_idx + j};
								double val[2] = {-1.0 , relu_expr->sup_coeff[0]};
								error = GRBaddconstr(model, 2, ind, val, GRB_EQUAL, relu_expr->inf_cst, NULL);
								handle_gurobi_error(error, env);
							}
							else{
								// unstable relu nodes, add two lower constarints, and also handle FP error for upper constraint
								expr_t * relu_expr = relu_node->uexpr;
								size_t num_pre_neurons = relu_expr->size;
								assert(relu_expr->type==SPARSE);
								assert(num_pre_neurons==1);
								// The lower bound setting already indicate that relu >=0
								error = GRBaddvar(model, 0, NULL, NULL, 0.0, -relu_node->lb, relu_node->ub, GRB_CONTINUOUS, NULL);
								handle_gurobi_error(error, env);
								int ind[2] = {layer_var_start_idx[k] + j, defined_var_start_idx + j};
								double val[2] = {-1.0 , 1.0};
								// add lower bound, y >= x, -y+x <= 0
								error = GRBaddconstr(model, 2, ind, val, GRB_LESS_EQUAL, 0.0, NULL);
								handle_gurobi_error(error, env);
								int ind2[2] = {layer_var_start_idx[k] + j, defined_var_start_idx + j};
								double over_slope = relu_expr->sup_coeff[0]+ ulp;
								double val2[2] = {-1.0, over_slope};
								int pre = cur_layer->predecessors[0]-1;
								double in_lb = fp->layers[pre]->neurons[j]->lb;
								assert(in_lb>=0);
								double over_b = (fabs(in_lb)+ulp)*over_slope + ulp;
								// add upper bound, y <= ax+b, -y+ax >= -b
								error = GRBaddconstr(model, 2, ind2, val2, GRB_GREATER_EQUAL, -over_b, NULL);
								handle_gurobi_error(error, env);
							}
							// update model
							error = GRBupdatemodel(model);
							handle_gurobi_error(error, env);
						}
					}
					else{
						// current layer is affine layer
						for(j=0; j < num_cur_neurons; j++){
							neuron_t * affine_node = cur_neurons[j];
							expr_t * affine_expr = affine_node->lexpr;
							size_t num_pre_neurons = affine_expr->size;
							assert(affine_expr->type==DENSE);
							error = GRBaddvar(model, 0, NULL, NULL, 0.0, -affine_node->lb, affine_node->ub, GRB_CONTINUOUS, NULL);
							handle_gurobi_error(error, env);
							int ind[num_pre_neurons+1];
							double val[num_pre_neurons+1];
							for(n=0; n < num_pre_neurons; n++){
								ind[n] = defined_var_start_idx + n;
								val[n] = affine_expr->sup_coeff[n];
							}
							ind[num_pre_neurons] = layer_var_start_idx[k] + j;
							val[num_pre_neurons] = -1.0;
							error = GRBaddconstr(model, num_pre_neurons+1, ind, val, GRB_EQUAL, affine_expr->inf_cst, NULL);
							handle_gurobi_error(error, env);
							// update model
							error = GRBupdatemodel(model);
							handle_gurobi_error(error, env);
						}
					}
				}
				// add the vars of next start layer neurons, which is Relu connected to the end layer
				neuron_t ** end_neurons = fp->layers[end_layer_index]->neurons;
				layer_t * start_layer_of_next_blk = fp->layers[ind_next_blk_connection];
				neuron_t ** relu_neurons = start_layer_of_next_blk->neurons;
				for(j=0; j < start_layer_of_next_blk->dims; j++){
					neuron_t * relu_node = relu_neurons[j];
					neuron_t * input_node = end_neurons[j];
					// relu interval constaint corresponds to output neuron constraints
					error = GRBaddvar(model, 0, NULL, NULL, 0.0, -relu_node->lb, relu_node->ub, GRB_CONTINUOUS, NULL);
					handle_gurobi_error(error, env);
					// the connection between, depends on the interval of end_layer
					if(input_node->ub<=0){
						// stable unactivated relu nodes
						int ind = layer_var_start_idx[ind_next_blk_connection] + j;
						double val = 1.0;
						error = GRBaddconstr(model, 1, &ind, &val, GRB_EQUAL, 0.0, NULL);
						handle_gurobi_error(error, env);
					}
					else if(input_node->lb<0){
						// stable activated relu nodes, y = x, -y +x = 0
						int ind[2] = {layer_var_start_idx[ind_next_blk_connection] + j, layer_var_start_idx[end_layer_index] + j};
						double val[2] = {-1.0 , 1.0};
						error = GRBaddconstr(model, 2, ind, val, GRB_EQUAL, 0.0, NULL);
						handle_gurobi_error(error, env);
					}
					else{
						// unstable relu nodes, add two lower constarints, and also handle FP error for upper constraint
						// add lower constriant that relu >=0
						int ind = layer_var_start_idx[ind_next_blk_connection] + j;
						double val = 1.0;
						error = GRBaddconstr(model, 1, &ind, &val, GRB_GREATER_EQUAL, 0.0, NULL);
						handle_gurobi_error(error, env);
						// add lower constraint, y >= x, -y+x <= 0
						int ind3[2] = {layer_var_start_idx[ind_next_blk_connection] + j, layer_var_start_idx[end_layer_index] + j};
						double val3[2] = {-1.0 , 1.0};
						error = GRBaddconstr(model, 2, ind3, val3, GRB_LESS_EQUAL, 0.0, NULL);
						handle_gurobi_error(error, env);
						// add upper bound, y <= ax+b, -y+ax >= -b
						int ind2[2] = {layer_var_start_idx[ind_next_blk_connection] + j, layer_var_start_idx[end_layer_index] + j};
						double in_lb = input_node->lb;
						assert(in_lb>=0);
						double over_slope = input_node->ub/(input_node->ub+in_lb) + ulp;
						double val2[2] = {-1.0, over_slope};
						double over_b = (fabs(in_lb)+ulp)*over_slope + ulp;
						error = GRBaddconstr(model, 2, ind2, val2, GRB_GREATER_EQUAL, -over_b, NULL);
						handle_gurobi_error(error, env);
					}
					// update model
					error = GRBupdatemodel(model);
					handle_gurobi_error(error, env);
				}

				// solving and updatin neuron status
				error = GRBoptimize(model);
				handle_gurobi_error(error, env);
				error = GRBgetintattr(model, GRB_INT_ATTR_STATUS, &optimstatus);
				handle_gurobi_error(error, env);
				if(optimstatus == GRB_INFEASIBLE){
					GRBfreemodel(model);
					GRBfreeenv(env);
					printf("Refine succesfully at block [%d-%d] at %d-th iteration, # total iteration is %d\n",start_layer_index, end_layer_index, count+1, total_ite);
					return true;
				}
				// solve for interval of start layer neurons
				size_t start_layer_num_neurons = (start_layer_index >= 0) ? fp->layers[start_layer_index]->dims : fp->num_pixels;
				clock_t LP_begin = clock();
				for(i=0; i < start_layer_num_neurons; i++){
					double solved_lb, solved_ub;
					error = GRBsetdblattrelement(model, "Obj", i, 1.0);
					handle_gurobi_error(error, env);
					// ModelSense, default value 1 indicates minimization, and -1 meaning maximization
					// lower bound solving
					error = GRBsetintattr(model, "ModelSense", 1);
					handle_gurobi_error(error, env);
					error = GRBupdatemodel(model);
					handle_gurobi_error(error, env);
					error = GRBoptimize(model);
					handle_gurobi_error(error, env);
					error = GRBgetdblattr(model, GRB_DBL_ATTR_OBJVAL, &solved_lb);
					handle_gurobi_error(error, env);
					// upper bound solving
					error = GRBsetintattr(model, "ModelSense", -1);
					handle_gurobi_error(error, env);
					error = GRBupdatemodel(model);
					handle_gurobi_error(error, env);
					error = GRBoptimize(model);
					handle_gurobi_error(error, env);
					error = GRBgetdblattr(model, GRB_DBL_ATTR_OBJVAL, &solved_ub);
					handle_gurobi_error(error, env);
					// Update the corresponding input lower and upper bound for next deeppoly execution
					double lp_solving_error = pow(10.0, -6.0) + ulp;
					if(start_layer_index >= 0){
						// printf("The start layer neuron %zu originally has interval [%.4f, %.4f]\n", i, -fp->layers[start_layer_index]->neurons[i]->lb, fp->layers[start_layer_index]->neurons[i]->ub);
						fp->layers[start_layer_index]->neurons[i]->lb = fmin(-(solved_lb - lp_solving_error),fp->layers[start_layer_index]->neurons[i]->lb);
						fp->layers[start_layer_index]->neurons[i]->ub = fmin(solved_ub + lp_solving_error,fp->layers[start_layer_index]->neurons[i]->ub);
						// printf("The start layer neuron %zu was updates to [%.4f, %.4f]\n", i, -fp->layers[start_layer_index]->neurons[i]->lb, fp->layers[start_layer_index]->neurons[i]->ub);
					}
					else{
						fp->input_inf[i] = -(solved_lb - lp_solving_error);
						fp->input_sup[i] = solved_ub + lp_solving_error;
						// printf("The resolved input neuron %zu has interval [%.4f, %.4f]\n", i, -fp->input_inf[i], fp->input_sup[i]);
					}
					// revert obj coeff back to 0
					error = GRBsetdblattrelement(model, "Obj", i, 0.0);
					handle_gurobi_error(error, env);
				}
				// clock_t LP_end = clock();
				// double LP_time_spent = (double)(LP_end - LP_begin) / CLOCKS_PER_SEC;
				// printf("Average LP solving time for each neuron is %f seconds\n", LP_time_spent/start_layer_num_neurons);	
				// solve for relu interval of unstable relu nodes
				int relu_refine_count = 0;
				for(i = start_layer_index + 1; i <= end_layer_index; i++){
					layer_t * cur_layer = fp->layers[i];
					if(!cur_layer->is_activation && (i < numlayers-1) && fp->layers[i+1]->is_activation){
						layer_t * next_layer = fp->layers[i+1];
						assert(cur_layer->dims == next_layer->dims);
						neuron_t ** relu_neurons = next_layer->neurons;
						for(j=0; j < cur_layer->dims; j++){
							if(relu_neurons[j]->ub!=0.0 && relu_neurons[j]->lb>=0){
								double solved_lb, solved_ub;
								error = GRBsetdblattrelement(model, "Obj", layer_var_start_idx[i]+j, 1.0);
								handle_gurobi_error(error, env);
								// ModelSense, default value 1 indicates minimization, and -1 meaning maximization
								// lower bound solving
								error = GRBsetintattr(model, "ModelSense", 1);
								handle_gurobi_error(error, env);
								error = GRBupdatemodel(model);
								handle_gurobi_error(error, env);
								error = GRBoptimize(model);
								handle_gurobi_error(error, env);
								error = GRBgetdblattr(model, GRB_DBL_ATTR_OBJVAL, &solved_lb);
								handle_gurobi_error(error, env);
								// update relu node lower bound
								double lp_solving_error = pow(10.0, -6.0) + ulp;
								cur_layer->neurons[j]->lb = fmin(-(solved_lb - lp_solving_error), cur_layer->neurons[j]->lb);
								if(cur_layer->neurons[j]->lb<0){
									relu_refine_count ++;
								}
								// upper bound solving
								error = GRBsetintattr(model, "ModelSense", -1);
								handle_gurobi_error(error, env);
								error = GRBupdatemodel(model);
								handle_gurobi_error(error, env);
								error = GRBoptimize(model);
								handle_gurobi_error(error, env);
								error = GRBgetdblattr(model, GRB_DBL_ATTR_OBJVAL, &solved_ub);
								handle_gurobi_error(error, env);
								// update relu node upper bound
								cur_layer->neurons[j]->ub = fmin(solved_ub + lp_solving_error, cur_layer->neurons[j]->ub);
								if(cur_layer->neurons[j]->ub<=0){
									relu_refine_count ++;
								}
								error = GRBsetdblattrelement(model, "Obj", layer_var_start_idx[i]+j, 0.0);
								handle_gurobi_error(error, env);
							}
						}
					}
				}
				// printf("Refreshed ReLU nodes: %d\n",relu_refine_count);
				/* Free model */
				GRBfreemodel(model);
				/* Free environment */
				GRBfreeenv(env);
				// clock_t ite_end = clock();
				// double ite_time_spent = (double)(ite_end - ite_begin) / CLOCKS_PER_SEC;
				// printf("This iteration took %f seconds to execute\n", ite_time_spent);
			}
			k = (start_layer_index >= 0) ? fp->layers[start_layer_index]->predecessors[0]-1 : -2 ;
			// clock_t blk_end = clock();
			// double blk_time_spent = (double)(blk_end - blk_begin) / CLOCKS_PER_SEC;
			// printf("Block [%d-%d] took %f seconds to execute \n", start_layer_index, end_layer_index, blk_time_spent);
		}
	}
	clock_t func_end = clock();
	double func_spent = (double)(func_end - func_begin) / CLOCKS_PER_SEC;
	printf("fail refinement, # total iteration is %d,total time is %f\n", total_ite, func_spent);
	return false;
}

bool is_spurious_modular_oneblock(elina_manager_t* man, elina_abstract0_t* element, elina_dim_t ground_truth_label, elina_dim_t poten_cex, int * spurious_list, int spurious_count, int MAX_ITER){
	// only analysis in a modular way, but do not use block summary, still encode all the constraints within a block
	int count, k;
	int total_ite = 0;
	size_t i, j, n;
	fppoly_t *fp = fppoly_of_abstract0(element);
    size_t numlayers = fp->numlayers;
	double ulp = ldexpl(1.0,-52);
	int optimstatus;
	clock_t func_begin = clock();
	// set the status of neurons back to original status
	for(i=0; i < fp->num_pixels; i++){
		fp->input_inf[i] = fp->original_input_inf[i];
		fp->input_sup[i] = fp->original_input_sup[i];
	}
	clear_neurons_status(man, element);
	clear_block_summary(man, element);
	run_deeppoly_in_block(man, element, -1, numlayers - 1);
	k = numlayers - 1;

	// handle the last block of the network 
	int start_layer_index = fp->layers[numlayers - 1]->start_idx_in_same_blk;
	if(start_layer_index == numlayers -1){
		// if the last block just contain one layer, merge it with the previous block
		start_layer_index = (numlayers >= 2) ? fp->layers[numlayers - 2]->start_idx_in_same_blk : -1;
	}
	// printf("the start_layer_index of last block is %d\n", start_layer_index);
	start_layer_index = 0;
	for(count = 0; count < MAX_ITER; count++){
		printf("Refinement iteration %d\n", count+1);
		clock_t ite_begin = clock();
		total_ite++;
		// refinement within this block for MAX_ITER times
		if(count!=0)
			run_deeppoly_in_block(man, element, start_layer_index, numlayers - 1);
		// only run deeppoly within this block
		clock_t buildmodel_start = clock();
		GRBenv *env   = NULL;
		GRBmodel *model = NULL;
		int error = 0;	
		error = GRBemptyenv(&env);
		handle_gurobi_error(error, env);
		// error = GRBsetstrparam(env, "LogFile", NULL);
		error = GRBsetintparam(env, "OutputFlag", 0);
		handle_gurobi_error(error, env);
		error = GRBstartenv(env);
		handle_gurobi_error(error, env);
		error = GRBnewmodel(env, &model, "refinement_solver", 0, NULL, NULL, NULL, NULL, NULL);
		handle_gurobi_error(error, env);
		int layer_var_start_idx[numlayers];
		if(start_layer_index >=0){
			layer_var_start_idx[start_layer_index] = 0;
			layer_t * start_layer = fp->layers[start_layer_index];
			layer_var_start_idx[start_layer_index + 1] = start_layer->dims;
			for(i=0; i < start_layer->dims; i++){
				error = GRBaddvar(model, 0, NULL, NULL, 0.0, -start_layer->neurons[i]->lb, start_layer->neurons[i]->ub, GRB_CONTINUOUS, NULL);
				handle_gurobi_error(error, env);
			}
		}
		else{
			// The last block is the whole network, then start layer is the input layer
			for(i=0; i < fp->num_pixels; i++){
				error = GRBaddvar(model, 0, NULL, NULL, 0.0, -fp->input_inf[i], fp->input_sup[i], GRB_CONTINUOUS, NULL);
				handle_gurobi_error(error, env);
			}
			layer_var_start_idx[start_layer_index + 1] = fp->num_pixels;
		}
		// Encode other layer constraints
		for(k = start_layer_index + 1; k < numlayers; k++){
			// Record the index starter for variables at different layers
			layer_t * cur_layer = fp->layers[k];
			neuron_t ** cur_neurons = cur_layer->neurons;
			size_t num_cur_neurons = cur_layer->dims;
			if(k+1 < numlayers){
				layer_var_start_idx[k+1] = layer_var_start_idx[k] + num_cur_neurons;
			}
			int defined_var_start_idx;
			if(k==0){
				defined_var_start_idx = 0;
			}
			else{
				defined_var_start_idx = layer_var_start_idx[k-1];
			}
			if(cur_layer->is_activation){
				for(j=0; j < num_cur_neurons; j++){
					// add constraints for each ReLU node
					neuron_t * relu_node = cur_neurons[j];
					if(relu_node->ub == 0.0){
						// stable unactivated relu nodes
						expr_t * relu_expr = relu_node->lexpr;
						assert(relu_expr->type==SPARSE);
						error = GRBaddvar(model, 0, NULL, NULL, 0.0, -relu_node->lb, relu_node->ub, GRB_CONTINUOUS, NULL);
						handle_gurobi_error(error, env);
					}
					else if(relu_node->lb<0.0){
						// stable activated relu nodes
						expr_t * relu_expr = relu_node->lexpr;
						size_t num_pre_neurons = relu_expr->size;
						assert(relu_expr->type==SPARSE);
						assert(num_pre_neurons==1);
						error = GRBaddvar(model, 0, NULL, NULL, 0.0, -relu_node->lb, relu_node->ub, GRB_CONTINUOUS, NULL);
						handle_gurobi_error(error, env);
						int ind[2] = {layer_var_start_idx[k] + j, defined_var_start_idx + j};
						double val[2] = {-1.0 , relu_expr->sup_coeff[0]};
						error = GRBaddconstr(model, 2, ind, val, GRB_EQUAL, relu_expr->inf_cst, NULL);
						handle_gurobi_error(error, env);
					}
					else{
						// unstable relu nodes, add two lower constarints, and also handle FP error for upper constraint
						expr_t * relu_expr = relu_node->uexpr;
						size_t num_pre_neurons = relu_expr->size;
						assert(relu_expr->type==SPARSE);
						assert(num_pre_neurons==1);
						// The lower bound setting already indicate that relu >=0
						error = GRBaddvar(model, 0, NULL, NULL, 0.0, -relu_node->lb, relu_node->ub, GRB_CONTINUOUS, NULL);
						handle_gurobi_error(error, env);
						int ind[2] = {layer_var_start_idx[k] + j, defined_var_start_idx + j};
						double val[2] = {-1.0 , 1.0};
						// add lower bound, y >= x, -y+x <= 0
						error = GRBaddconstr(model, 2, ind, val, GRB_LESS_EQUAL, 0.0, NULL);
						handle_gurobi_error(error, env);
						int ind2[2] = {layer_var_start_idx[k] + j, defined_var_start_idx + j};
						double over_slope = relu_expr->sup_coeff[0]+ ulp;
						double val2[2] = {-1.0, over_slope};
						int pre = cur_layer->predecessors[0]-1;
						double in_lb = fp->layers[pre]->neurons[j]->lb;
						assert(in_lb>=0);
						double over_b = (fabs(in_lb)+ulp)*over_slope + ulp;
						// add upper bound, y <= ax+b, -y+ax >= -b
						error = GRBaddconstr(model, 2, ind2, val2, GRB_GREATER_EQUAL, -over_b, NULL);
						handle_gurobi_error(error, env);
					}
					// update model
					error = GRBupdatemodel(model);
					handle_gurobi_error(error, env);
				}
			}
			else{
				// current layer is affine layer
				for(j=0; j < num_cur_neurons; j++){
					neuron_t * affine_node = cur_neurons[j];
					expr_t * affine_expr = affine_node->lexpr;
					size_t num_pre_neurons = affine_expr->size;
					assert(affine_expr->type==DENSE);
					error = GRBaddvar(model, 0, NULL, NULL, 0.0, -affine_node->lb, affine_node->ub, GRB_CONTINUOUS, NULL);
					handle_gurobi_error(error, env);
					int ind[num_pre_neurons+1];
					double val[num_pre_neurons+1];
					for(n=0; n < num_pre_neurons; n++){
						ind[n] = defined_var_start_idx + n;
						val[n] = affine_expr->sup_coeff[n];
					}
					ind[num_pre_neurons] = layer_var_start_idx[k] + j;
					val[num_pre_neurons] = -1.0;
					error = GRBaddconstr(model, num_pre_neurons+1, ind, val, GRB_EQUAL, affine_expr->inf_cst, NULL);
					handle_gurobi_error(error, env);
					// update model
					error = GRBupdatemodel(model);
					handle_gurobi_error(error, env);
				}
			}
		}
		// Add constraints regarding the current potential counter-label
		int var_start_idx = layer_var_start_idx[numlayers - 1];
		int ind[2] = {var_start_idx+ground_truth_label,var_start_idx+poten_cex};
		double val[2] = {1.0, -1.0};
		error = GRBaddconstr(model, 2, ind, val, GRB_EQUAL, 0.0, NULL);
		error = GRBupdatemodel(model);
		handle_gurobi_error(error, env);

		// Add constraints regarding previous spurious labels
		for(j=0; j < spurious_count; j++){
			int spu_label = spurious_list[j];
			// we have out[ground_truth_label] - out[spu_label] > 0, for practical concern, we expand to >=
			int var_start_idx = layer_var_start_idx[numlayers - 1];
			int ind[2] = {var_start_idx+ground_truth_label,var_start_idx+spu_label};
			double val[2] = {1.0, -1.0};
			error = GRBaddconstr(model, 2, ind, val, GRB_GREATER_EQUAL, 0.0, NULL);
			handle_gurobi_error(error, env);
		}

		error = GRBoptimize(model);
		handle_gurobi_error(error, env);
		error = GRBgetintattr(model, GRB_INT_ATTR_STATUS, &optimstatus);
		handle_gurobi_error(error, env);
		if(optimstatus == GRB_INFEASIBLE){
			GRBfreemodel(model);
			GRBfreeenv(env);
			clock_t func_end = clock();
			double func_spent = (double)(func_end - func_begin) / CLOCKS_PER_SEC;
			printf("Refine succesfully at last block at %d-th iteration, # total iteration is %d, total time is %f\n", count+1, total_ite, func_spent);
			return true;
		}
		// solve for interval of start layer neurons
		size_t start_layer_num_neurons = (start_layer_index >= 0) ? fp->layers[start_layer_index]->dims : fp->num_pixels;
		clock_t LP_begin = clock();
		for(i=0; i < start_layer_num_neurons; i++){
			double solved_lb, solved_ub;
			error = GRBsetdblattrelement(model, "Obj", i, 1.0);
			handle_gurobi_error(error, env);
			// ModelSense, default value 1 indicates minimization, and -1 meaning maximization
			// lower bound solving
			error = GRBsetintattr(model, "ModelSense", 1);
			handle_gurobi_error(error, env);
			error = GRBupdatemodel(model);
			handle_gurobi_error(error, env);
			error = GRBoptimize(model);
			handle_gurobi_error(error, env);
			error = GRBgetdblattr(model, GRB_DBL_ATTR_OBJVAL, &solved_lb);
			handle_gurobi_error(error, env);
			// upper bound solving
			error = GRBsetintattr(model, "ModelSense", -1);
			handle_gurobi_error(error, env);
			error = GRBupdatemodel(model);
			handle_gurobi_error(error, env);
			error = GRBoptimize(model);
			handle_gurobi_error(error, env);
			error = GRBgetdblattr(model, GRB_DBL_ATTR_OBJVAL, &solved_ub);
			handle_gurobi_error(error, env);
			// Update the corresponding input lower and upper bound for next deeppoly execution
			double lp_solving_error = pow(10.0, -6.0) + ulp;
			if(start_layer_index >= 0){
				if(((solved_lb - lp_solving_error) > -fp->layers[start_layer_index]->neurons[i]->lb) || ((solved_ub + lp_solving_error) < fp->layers[start_layer_index]->neurons[i]->ub)){
					printf("The start layer neuron %zu originally has interval [%.4f, %.4f]\n", i, -fp->layers[start_layer_index]->neurons[i]->lb, fp->layers[start_layer_index]->neurons[i]->ub);
					printf("The start layer neuron %zu was updates to [%.4f, %.4f]\n", i, (solved_lb - lp_solving_error), solved_ub + lp_solving_error);
				}
				fp->layers[start_layer_index]->neurons[i]->lb = fmin(-(solved_lb - lp_solving_error),fp->layers[start_layer_index]->neurons[i]->lb);
				fp->layers[start_layer_index]->neurons[i]->ub = fmin(solved_ub + lp_solving_error,fp->layers[start_layer_index]->neurons[i]->ub);
			}
			else{
				fp->input_inf[i] = -(solved_lb - lp_solving_error);
				fp->input_sup[i] = solved_ub + lp_solving_error;
				// printf("The resolved input neuron %zu has interval [%.4f, %.4f]\n", i, -fp->input_inf[i], fp->input_sup[i]);
			}
			// revert obj coeff back to 0
			error = GRBsetdblattrelement(model, "Obj", i, 0.0);
			handle_gurobi_error(error, env);
		}
		// clock_t LP_end = clock();
		// double LP_time_spent = (double)(LP_end - LP_begin) / CLOCKS_PER_SEC;
		// printf("Average LP solving time for each neuron is %f seconds\n", LP_time_spent/start_layer_num_neurons);	
		// solve for relu interval of unstable relu nodes
		int relu_refine_count = 0;
		for(i = start_layer_index + 1; i < numlayers; i++){
			layer_t * cur_layer = fp->layers[i];
			if(!cur_layer->is_activation && (i < numlayers-1) && fp->layers[i+1]->is_activation){
				layer_t * next_layer = fp->layers[i+1];
				assert(cur_layer->dims == next_layer->dims);
				neuron_t ** relu_neurons = next_layer->neurons;
				for(j=0; j < cur_layer->dims; j++){
					if(relu_neurons[j]->ub!=0.0 && relu_neurons[j]->lb>=0){
						double solved_lb, solved_ub;
						error = GRBsetdblattrelement(model, "Obj", layer_var_start_idx[i]+j, 1.0);
						handle_gurobi_error(error, env);
						// ModelSense, default value 1 indicates minimization, and -1 meaning maximization
						// lower bound solving
						error = GRBsetintattr(model, "ModelSense", 1);
						handle_gurobi_error(error, env);
						error = GRBupdatemodel(model);
						handle_gurobi_error(error, env);
						error = GRBoptimize(model);
						handle_gurobi_error(error, env);
						error = GRBgetdblattr(model, GRB_DBL_ATTR_OBJVAL, &solved_lb);
						handle_gurobi_error(error, env);
						// update relu node lower bound
						double lp_solving_error = pow(10.0, -6.0) + ulp;
						cur_layer->neurons[j]->lb = fmin(-(solved_lb - lp_solving_error), cur_layer->neurons[j]->lb);
						if(cur_layer->neurons[j]->lb<0){
							relu_refine_count ++;
						}
						// upper bound solving
						error = GRBsetintattr(model, "ModelSense", -1);
						handle_gurobi_error(error, env);
						error = GRBupdatemodel(model);
						handle_gurobi_error(error, env);
						error = GRBoptimize(model);
						handle_gurobi_error(error, env);
						error = GRBgetdblattr(model, GRB_DBL_ATTR_OBJVAL, &solved_ub);
						handle_gurobi_error(error, env);
						// update relu node upper bound
						cur_layer->neurons[j]->ub = fmin(solved_ub + lp_solving_error, cur_layer->neurons[j]->ub);
						if(cur_layer->neurons[j]->ub<=0){
							relu_refine_count ++;
						}
						error = GRBsetdblattrelement(model, "Obj", layer_var_start_idx[i]+j, 0.0);
						handle_gurobi_error(error, env);
					}
				}
			}
		}
		printf("Refreshed ReLU nodes: %d\n",relu_refine_count);
		/* Free model */
		GRBfreemodel(model);
		/* Free environment */
		GRBfreeenv(env);
		// clock_t ite_end = clock();
		// double ite_time_spent = (double)(ite_end - ite_begin) / CLOCKS_PER_SEC;
		// printf("This iteration took %f seconds to execute\n", ite_time_spent);
	}
	clock_t func_end = clock();
	double func_spent = (double)(func_end - func_begin) / CLOCKS_PER_SEC;
	printf("fail refinement, # total iteration is %d,total time is %f\n", total_ite, func_spent);
	return false;
}

bool is_spurious_whole_net_with_blksum(elina_manager_t* man, elina_abstract0_t* element, elina_dim_t ground_truth_label, elina_dim_t poten_cex, int * spurious_list, int spurious_count, int MAX_ITER){
	// firstly consider the default case, where like in SMU paper, to encode all the constraints within the network
	int count, k;
	size_t i, j, n;
	fppoly_t *fp = fppoly_of_abstract0(element);
    size_t numlayers = fp->numlayers;
	double ulp = ldexpl(1.0,-52);
	int optimstatus;
	for(i=0; i < fp->num_pixels; i++){
		// set the input neurons back to the original input space
		fp->input_inf[i] = fp->original_input_inf[i];
		fp->input_sup[i] = fp->original_input_sup[i];
	}
	// For new CEX pruning, clear the previous analysis bounds and block summaries
	clear_neurons_status(man, element);
	clear_block_summary(man, element);

	// Refine for MAX_ITER times
	for(count = 0; count < MAX_ITER; count++){
		// run bbpoly for the whole network
		run_bbpoly_in_block(man, element, -1, numlayers - 1);
		// printf("Refinement iteration %d\n", count);
		/* Create environment */
  		GRBenv *env   = NULL;
  		GRBmodel *model = NULL;
		int error = 0;	
		error = GRBemptyenv(&env);
		handle_gurobi_error(error, env);
		error = GRBsetintparam(env, "OutputFlag", 0);
		handle_gurobi_error(error, env);
		error = GRBstartenv(env);
		handle_gurobi_error(error, env);
		/* Create an empty model */
		error = GRBnewmodel(env, &model, "refinement_solver", 0, NULL, NULL, NULL, NULL, NULL);
		handle_gurobi_error(error, env);

		// The index starter for variables at different layers
		// For intermediate layers within a block that we don't count, set the idx to be -1, meaning infeasible
		int layer_var_start_idx[numlayers];
		for(i=0; i < numlayers; i++){
			// initialize the value, -1 meaning this layer will not be counted during encoding
			layer_var_start_idx[i] = -1;
		}
		k = numlayers-1;
		while(k>=0){
			// Setting up flag to indicate how to add constraint for this layer
			if(k == numlayers - 1){
				int start_layer_index = fp->layers[k]->start_idx_in_same_blk;
				// printf("start_layer_index is %d\n", start_layer_index);
				// printf("k is %d\n", k);
				if(start_layer_index == numlayers -1){
					// if the last block just contain one layer, merge it with the previous block
					start_layer_index = (numlayers >= 2) ? fp->layers[numlayers - 2]->start_idx_in_same_blk : -1;
				}
				for(k = start_layer_index; k < numlayers; k++){
					// indicating that for this layer, full constraints will be considered instead of block summary
					if(k>=0)
						layer_var_start_idx[k] = 2;
				}
				k = (start_layer_index >= 0) ? fp->layers[start_layer_index]->predecessors[0]-1 : -2 ;
			}
			else{
				layer_var_start_idx[k] = 1; //encode with block summary
				int start_layer_index = fp->layers[k]->start_idx_in_same_blk;
				if(start_layer_index >=0){
					layer_var_start_idx[start_layer_index] = 1;
				}
				k = (start_layer_index >= 0) ? fp->layers[start_layer_index]->predecessors[0]-1 : -2 ;
			}
		}
		// add the input layer constraints
		for(i=0; i < fp->num_pixels; i++){
			error = GRBaddvar(model, 0, NULL, NULL, 0.0, -fp->input_inf[i], fp->input_sup[i], GRB_CONTINUOUS, NULL);
        	handle_gurobi_error(error, env);
		}
		// add block summaried for each block, and for the last block, still encode all the constraints
		for(i = 0 ; i < numlayers; i++){
			layer_t * cur_layer = fp->layers[i];
			neuron_t ** cur_neurons = cur_layer->neurons;
			size_t num_cur_neurons = cur_layer->dims;
			if((layer_var_start_idx[i]==1) && (cur_layer->is_end_layer_of_blk)){
				int start_layer_index = fp->layers[i]->start_idx_in_same_blk;
				if(start_layer_index<0){
					layer_var_start_idx[i] = fp->num_pixels;
				}
				else{
					layer_var_start_idx[i] = layer_var_start_idx[start_layer_index] + fp->layers[start_layer_index]->dims;
				}
				// Encode the constaints of block summary
				assert(!cur_layer->is_activation); 
				for(j = 0; j < num_cur_neurons; j++){
					neuron_t * end_node = cur_neurons[j];
					expr_t * summary_lexpr = end_node->summary_lexpr;
					expr_t * summary_uexpr = end_node->summary_uexpr;
					assert(summary_lexpr->type==DENSE);
					size_t num_pre_neurons = summary_lexpr->size;
					error = GRBaddvar(model, 0, NULL, NULL, 0.0, -end_node->lb, end_node->ub, GRB_CONTINUOUS, NULL);
					handle_gurobi_error(error, env);
					// For upper summary, need to encode the c+ coefficient
					int ind[num_pre_neurons+1];
  					double val[num_pre_neurons+1];
					int defined_over_idx = (start_layer_index>=0) ? layer_var_start_idx[start_layer_index] : 0;
					for(n=0; n < num_pre_neurons; n++){
						ind[n] = defined_over_idx + n;
						val[n] = summary_uexpr->sup_coeff[n];
					}
					ind[num_pre_neurons] = layer_var_start_idx[i] + j;
					val[num_pre_neurons] = -1.0;
					// y <= ax+b  -> ax - y >= -b
					error = GRBaddconstr(model, num_pre_neurons+1, ind, val, GRB_GREATER_EQUAL, -summary_uexpr->sup_cst , NULL);
					handle_gurobi_error(error, env);
					// For lower summary, need to encode the c- coefficient
  					double val2[num_pre_neurons+1];
					for(n=0; n < num_pre_neurons; n++){
						val2[n] = -summary_lexpr->inf_coeff[n];
					}
					val2[num_pre_neurons] = -1.0;
					// y >= ax+b  -> ax - y <= -b
					error = GRBaddconstr(model, num_pre_neurons+1, ind, val2, GRB_LESS_EQUAL, summary_lexpr->inf_cst, NULL);
					handle_gurobi_error(error, env);
					// update model
					error = GRBupdatemodel(model);
					handle_gurobi_error(error, env);
				}
			}
			else if((layer_var_start_idx[i]==1) && (cur_layer->is_start_layer_of_blk)){
				assert(cur_layer->is_activation);
				layer_var_start_idx[i] = layer_var_start_idx[i-1] + fp->layers[i-1]->dims;
				//current layer is ReLU layer, we add the constraints according to RELU behavior
				for(j=0; j < num_cur_neurons; j++){
					// add constraints for each ReLU node
					// need to handle non-stable (two lower constraints will be added) and stable constraint
					neuron_t * relu_node = cur_neurons[j];
					if(relu_node->ub == 0.0){
						// stable unactivated relu nodes
						expr_t * relu_expr = relu_node->lexpr;
						assert(relu_expr->type==SPARSE);
						error = GRBaddvar(model, 0, NULL, NULL, 0.0, -relu_node->lb, relu_node->ub, GRB_CONTINUOUS, NULL);
						handle_gurobi_error(error, env);
					}
					else if(relu_node->lb<0.0){
						// stable activated relu nodes
						expr_t * relu_expr = relu_node->lexpr;
						size_t num_pre_neurons = relu_expr->size;
						assert(relu_expr->type==SPARSE);
						assert(num_pre_neurons==1);
						error = GRBaddvar(model, 0, NULL, NULL, 0.0, -relu_node->lb, relu_node->ub, GRB_CONTINUOUS, NULL);
						handle_gurobi_error(error, env);
						int ind[2] = {layer_var_start_idx[i] + j, layer_var_start_idx[i-1] + j};
						double val[2] = {-1.0 , relu_expr->sup_coeff[0]};
						error = GRBaddconstr(model, 2, ind, val, GRB_EQUAL, relu_expr->inf_cst, NULL);
						handle_gurobi_error(error, env);
					}
					else{
						// unstable relu nodes, add two lower constarints, and also handle FP error for upper constraint
						expr_t * relu_expr = relu_node->uexpr;
						size_t num_pre_neurons = relu_expr->size;
						assert(relu_expr->type==SPARSE);
						assert(num_pre_neurons==1);
						// The lower bound setting already indicate that relu >=0
						error = GRBaddvar(model, 0, NULL, NULL, 0.0, -relu_node->lb, relu_node->ub, GRB_CONTINUOUS, NULL);
						handle_gurobi_error(error, env);
						int ind[2] = {layer_var_start_idx[i] + j, layer_var_start_idx[i-1] + j};
						double val[2] = {-1.0 , 1.0};
						// add lower bound, y >= x, -y+x <= 0
						error = GRBaddconstr(model, 2, ind, val, GRB_LESS_EQUAL, 0.0, NULL);
						handle_gurobi_error(error, env);
						int ind2[2] = {layer_var_start_idx[i] + j, layer_var_start_idx[i-1] + j};
						double over_slope = relu_expr->sup_coeff[0]+ ulp;
						double val2[2] = {-1.0, over_slope};
						int pre = cur_layer->predecessors[0]-1;
						double in_lb = fp->layers[pre]->neurons[j]->lb;
						assert(in_lb>=0);
						double over_b = (fabs(in_lb)+ulp)*over_slope + ulp;
						// add upper bound, y <= ax+b, -y+ax >= -b
						error = GRBaddconstr(model, 2, ind2, val2, GRB_GREATER_EQUAL, -over_b, NULL);
						handle_gurobi_error(error, env);
					}
					// update model
					error = GRBupdatemodel(model);
					handle_gurobi_error(error, env);
				}
			}
			else if(layer_var_start_idx[i]==2){
				// for layer like this, we directly encode the symbolic constraints
				layer_var_start_idx[i] = layer_var_start_idx[i-1] + fp->layers[i-1]->dims;
				if(cur_layer->is_activation){
					for(j=0; j < num_cur_neurons; j++){
						// add constraints for each ReLU node
						neuron_t * relu_node = cur_neurons[j];
						if(relu_node->ub == 0.0){
							// stable unactivated relu nodes
							expr_t * relu_expr = relu_node->lexpr;
							assert(relu_expr->type==SPARSE);
							error = GRBaddvar(model, 0, NULL, NULL, 0.0, -relu_node->lb, relu_node->ub, GRB_CONTINUOUS, NULL);
							handle_gurobi_error(error, env);
						}
						else if(relu_node->lb<0.0){
							// stable activated relu nodes
							expr_t * relu_expr = relu_node->lexpr;
							size_t num_pre_neurons = relu_expr->size;
							assert(relu_expr->type==SPARSE);
							assert(num_pre_neurons==1);
							error = GRBaddvar(model, 0, NULL, NULL, 0.0, -relu_node->lb, relu_node->ub, GRB_CONTINUOUS, NULL);
							handle_gurobi_error(error, env);
							int ind[2] = {layer_var_start_idx[i] + j, layer_var_start_idx[i-1] + j};
							double val[2] = {-1.0 , relu_expr->sup_coeff[0]};
							error = GRBaddconstr(model, 2, ind, val, GRB_EQUAL, relu_expr->inf_cst, NULL);
							handle_gurobi_error(error, env);
						}
						else{
							// unstable relu nodes, add two lower constarints, and also handle FP error for upper constraint
							expr_t * relu_expr = relu_node->uexpr;
							size_t num_pre_neurons = relu_expr->size;
							assert(relu_expr->type==SPARSE);
							assert(num_pre_neurons==1);
							// The lower bound setting already indicate that relu >=0
							error = GRBaddvar(model, 0, NULL, NULL, 0.0, -relu_node->lb, relu_node->ub, GRB_CONTINUOUS, NULL);
							handle_gurobi_error(error, env);
							int ind[2] = {layer_var_start_idx[i] + j, layer_var_start_idx[i-1] + j};
							double val[2] = {-1.0 , 1.0};
							// add lower bound, y >= x, -y+x <= 0
							error = GRBaddconstr(model, 2, ind, val, GRB_LESS_EQUAL, 0.0, NULL);
							handle_gurobi_error(error, env);
							int ind2[2] = {layer_var_start_idx[i] + j, layer_var_start_idx[i-1] + j};
							double over_slope = relu_expr->sup_coeff[0]+ ulp;
							double val2[2] = {-1.0, over_slope};
							int pre = cur_layer->predecessors[0]-1;
							double in_lb = fp->layers[pre]->neurons[j]->lb;
							assert(in_lb>=0);
							double over_b = (fabs(in_lb)+ulp)*over_slope + ulp;
							// add upper bound, y <= ax+b, -y+ax >= -b
							error = GRBaddconstr(model, 2, ind2, val2, GRB_GREATER_EQUAL, -over_b, NULL);
							handle_gurobi_error(error, env);
						}
						// update model
						error = GRBupdatemodel(model);
						handle_gurobi_error(error, env);
					}
				}
				else{
					// current layer is affine layer
					for(j=0; j < num_cur_neurons; j++){
						neuron_t * affine_node = cur_neurons[j];
						expr_t * affine_expr = affine_node->lexpr;
						size_t num_pre_neurons = affine_expr->size;
						assert(affine_expr->type==DENSE);
						error = GRBaddvar(model, 0, NULL, NULL, 0.0, -affine_node->lb, affine_node->ub, GRB_CONTINUOUS, NULL);
						handle_gurobi_error(error, env);
						int ind[num_pre_neurons+1];
						double val[num_pre_neurons+1];
						for(n=0; n < num_pre_neurons; n++){
							ind[n] = layer_var_start_idx[i-1] + n;
							val[n] = affine_expr->sup_coeff[n];
						}
						ind[num_pre_neurons] = layer_var_start_idx[i] + j;
						val[num_pre_neurons] = -1.0;
						error = GRBaddconstr(model, num_pre_neurons+1, ind, val, GRB_EQUAL, affine_expr->inf_cst, NULL);
						handle_gurobi_error(error, env);
						// update model
						error = GRBupdatemodel(model);
						handle_gurobi_error(error, env);
					}
				}
			}
		}

		// add constraints for previously spurious labels
		for(k=0; k < spurious_count; k++){
			int spu_label = spurious_list[k];
			// we have out[ground_truth_label] - out[spu_label] > 0, for practical concern, we expand to >=
			int var_start_idx = layer_var_start_idx[numlayers - 1];
			int ind[2] = {var_start_idx+ground_truth_label,var_start_idx+spu_label};
			double val[2] = {1.0, -1.0};
			error = GRBaddconstr(model, 2, ind, val, GRB_GREATER_EQUAL, 0.0, NULL);
			handle_gurobi_error(error, env);
		}

		// add constraints regarding the current potential adversarial labels we try to eliminate, out[ground_truth_label] - out[spu_label] <= 0
		int var_start_idx = layer_var_start_idx[numlayers - 1];
		int ind[2] = {var_start_idx+ground_truth_label,var_start_idx+poten_cex};
		double val[2] = {1.0, -1.0};
		error = GRBaddconstr(model, 2, ind, val, GRB_LESS_EQUAL, 0.0, NULL);
		// update model
		error = GRBupdatemodel(model);
		handle_gurobi_error(error, env);

		// Simply check the feasibility, without objective function, if infeasible, then successfully prove spurious, return True
		error = GRBoptimize(model);
		handle_gurobi_error(error, env);
		/* Capture solution information */
		error = GRBgetintattr(model, GRB_INT_ATTR_STATUS, &optimstatus);
		handle_gurobi_error(error, env);
		if(optimstatus == GRB_INFEASIBLE){
			GRBfreemodel(model);
			GRBfreeenv(env);
			return true;
		}
		// Else, if detect feasibility, do the solving
		// solve for interval of input neurons
		for(i=0; i < fp->num_pixels; i++){
			double solved_lb, solved_ub;
			error = GRBsetdblattrelement(model, "Obj", i, 1.0);
        	handle_gurobi_error(error, env);
			// ModelSense, default value 1 indicates minimization, and -1 meaning maximization
			// lower bound solving
			error = GRBsetintattr(model, "ModelSense", 1);
    		handle_gurobi_error(error, env);
			error = GRBupdatemodel(model);
			handle_gurobi_error(error, env);
			error = GRBoptimize(model);
			handle_gurobi_error(error, env);
			error = GRBgetdblattr(model, GRB_DBL_ATTR_OBJVAL, &solved_lb);
			handle_gurobi_error(error, env);
			// upper bound solving
			error = GRBsetintattr(model, "ModelSense", -1);
    		handle_gurobi_error(error, env);
			error = GRBupdatemodel(model);
			handle_gurobi_error(error, env);
			error = GRBoptimize(model);
			handle_gurobi_error(error, env);
			error = GRBgetdblattr(model, GRB_DBL_ATTR_OBJVAL, &solved_ub);
			handle_gurobi_error(error, env);
			// Update the corresponding input lower and upper bound for next deeppoly execution
			double lp_solving_error = pow(10.0, -6.0) + ulp;
			fp->input_inf[i] = -(solved_lb - lp_solving_error);
			fp->input_sup[i] = solved_ub + lp_solving_error;
			// printf("The resolved input neuron %zu has interval [%.4f, %.4f]\n", i, -fp->input_inf[i], fp->input_sup[i]);
			// revert obj coeff back to 0
			error = GRBsetdblattrelement(model, "Obj", i, 0.0);
			handle_gurobi_error(error, env);
		}
		// solve for relu interval of unstable relu nodes
		int relu_refine_count = 0;
		for(i=0; i < numlayers; i++){
			layer_t * cur_layer = fp->layers[i];
			if((layer_var_start_idx[i]>=1) && !cur_layer->is_activation && (i < numlayers-1) && fp->layers[i+1]->is_activation){
				layer_t * next_layer = fp->layers[i+1];
				assert(cur_layer->dims == next_layer->dims);
				neuron_t ** relu_neurons = next_layer->neurons;
				for(j=0; j < cur_layer->dims; j++){
					if(relu_neurons[j]->ub!=0.0 && relu_neurons[j]->lb>=0){
						double solved_lb, solved_ub;
						error = GRBsetdblattrelement(model, "Obj", layer_var_start_idx[i]+j, 1.0);
						handle_gurobi_error(error, env);
						// ModelSense, default value 1 indicates minimization, and -1 meaning maximization
						// lower bound solving
						error = GRBsetintattr(model, "ModelSense", 1);
						handle_gurobi_error(error, env);
						error = GRBupdatemodel(model);
						handle_gurobi_error(error, env);
						error = GRBoptimize(model);
						handle_gurobi_error(error, env);
						error = GRBgetdblattr(model, GRB_DBL_ATTR_OBJVAL, &solved_lb);
						handle_gurobi_error(error, env);
						// update relu node lower bound
						double lp_solving_error = pow(10.0, -6.0) + ulp;
						cur_layer->neurons[j]->lb = fmin(-(solved_lb - lp_solving_error), cur_layer->neurons[j]->lb);
						if(cur_layer->neurons[j]->lb<0){
							relu_refine_count ++;
						}
						// upper bound solving
						error = GRBsetintattr(model, "ModelSense", -1);
						handle_gurobi_error(error, env);
						error = GRBupdatemodel(model);
						handle_gurobi_error(error, env);
						error = GRBoptimize(model);
						handle_gurobi_error(error, env);
						error = GRBgetdblattr(model, GRB_DBL_ATTR_OBJVAL, &solved_ub);
						handle_gurobi_error(error, env);
						// update relu node upper bound
						cur_layer->neurons[j]->ub = fmin(solved_ub + lp_solving_error, cur_layer->neurons[j]->ub);
						if(cur_layer->neurons[j]->ub<=0){
							relu_refine_count ++;
						}
						error = GRBsetdblattrelement(model, "Obj", layer_var_start_idx[i]+j, 0.0);
						handle_gurobi_error(error, env);
					}
				}
			}
		}
		// printf("Refreshed ReLU nodes: %d\n",relu_refine_count);
		/* Free model */
  		GRBfreemodel(model);
  		/* Free environment */
  		GRBfreeenv(env);
	}
	if(optimstatus == GRB_OPTIMAL){
		printf("Need to do adversarial example finding or quantitative robustness\n");
	}
	return false;
}

double return_max_among_four(double n1, double n2, double n3, double n4){
	double max1 = (n1 > n2) ? n1 : n2;
	double max2 = (n3 > n4) ? n3 : n4;
	return (max1 > max2) ? max1 : max2;
}

bool test_better_pair_neurons(elina_manager_t* man, elina_abstract0_t* element, elina_dim_t ground_truth_label, elina_dim_t poten_cex){
	int count, k;
	size_t i, j, n;
	fppoly_t *fp = fppoly_of_abstract0(element);
    size_t numlayers = fp->numlayers;
	double ulp = ldexpl(1.0,-52);
	double lp_solving_error = pow(10.0, -6.0) + ulp;
	int optimstatus;
	/* Create environment */
	GRBenv *env   = NULL;
	GRBmodel *model = NULL;
	int error = 0;	
	error = GRBemptyenv(&env);
	handle_gurobi_error(error, env);
	error = GRBsetintparam(env, "OutputFlag", 0);
	handle_gurobi_error(error, env);
	error = GRBstartenv(env);
	handle_gurobi_error(error, env);
	/* Create an empty model */
	error = GRBnewmodel(env, &model, "refinement_solver", 0, NULL, NULL, NULL, NULL, NULL);
	handle_gurobi_error(error, env);

	// The index starter for variables at different layers
	int layer_var_start_idx[numlayers];

	// fp->input_inf[i], fp->input_sup[i], add the input layer constraints
	layer_var_start_idx[0] = fp->num_pixels;
	for(i=0; i < fp->num_pixels; i++){
		error = GRBaddvar(model, 0, NULL, NULL, 0.0, -fp->input_inf[i], fp->input_sup[i], GRB_CONTINUOUS, NULL);
		handle_gurobi_error(error, env);
	}
	
	// add constaints for each hidden and output layer
	for(i=0; i < numlayers; i++){
		layer_t * cur_layer = fp->layers[i];
		neuron_t ** cur_neurons = cur_layer->neurons;
		size_t num_cur_neurons = cur_layer->dims;
		if(i+1 < numlayers){
			// Set up the variable start index 
			layer_var_start_idx[i+1] = layer_var_start_idx[i] + num_cur_neurons;
		}
		int defined_var_start_idx;
		if(i==0){
			defined_var_start_idx = 0;
		}
		else{
			defined_var_start_idx = layer_var_start_idx[i-1];
		}

		if(cur_layer->is_activation){
			//current layer is ReLU layer, we add the constraints according to RELU behavior
			for(j=0; j < num_cur_neurons; j++){
				// add constraints for each ReLU node
				// need to handle non-stable (two lower constraints will be added) and stable constraint
				neuron_t * relu_node = cur_neurons[j];
				if(relu_node->ub == 0.0){
					// stable unactivated relu nodes
					expr_t * relu_expr = relu_node->lexpr;
					assert(relu_expr->type==SPARSE);
					error = GRBaddvar(model, 0, NULL, NULL, 0.0, -relu_node->lb, relu_node->ub, GRB_CONTINUOUS, NULL);
					handle_gurobi_error(error, env);
				}
				else if(relu_node->lb<0.0){
					// stable activated relu nodes
					expr_t * relu_expr = relu_node->lexpr;
					size_t num_pre_neurons = relu_expr->size;
					assert(relu_expr->type==SPARSE);
					assert(num_pre_neurons==1);
					error = GRBaddvar(model, 0, NULL, NULL, 0.0, -relu_node->lb, relu_node->ub, GRB_CONTINUOUS, NULL);
					handle_gurobi_error(error, env);
					int ind[2] = {layer_var_start_idx[i] + j, defined_var_start_idx + j};
					double val[2] = {-1.0 , relu_expr->sup_coeff[0]};
					error = GRBaddconstr(model, 2, ind, val, GRB_EQUAL, relu_expr->inf_cst, NULL);
					handle_gurobi_error(error, env);
				}
				else{
					// unstable relu nodes, add two lower constarints, and also handle FP error for upper constraint
					expr_t * relu_expr = relu_node->uexpr;
					size_t num_pre_neurons = relu_expr->size;
					assert(relu_expr->type==SPARSE);
					assert(num_pre_neurons==1);
					// The lower bound setting already indicate that relu >=0
					error = GRBaddvar(model, 0, NULL, NULL, 0.0, -relu_node->lb, relu_node->ub, GRB_CONTINUOUS, NULL);
					handle_gurobi_error(error, env);
					int ind[2] = {layer_var_start_idx[i] + j, defined_var_start_idx + j};
					double val[2] = {-1.0 , 1.0};
					// add lower bound, y >= x, -y+x <= 0
					error = GRBaddconstr(model, 2, ind, val, GRB_LESS_EQUAL, 0.0, NULL);
					handle_gurobi_error(error, env);
					int ind2[2] = {layer_var_start_idx[i] + j, defined_var_start_idx + j};
					double over_slope = relu_expr->sup_coeff[0]+ ulp;
					double val2[2] = {-1.0, over_slope};
					int pre = cur_layer->predecessors[0]-1;
					double in_lb = fp->layers[pre]->neurons[j]->lb;
					assert(in_lb>=0);
					double over_b = (fabs(in_lb)+ulp)*over_slope + ulp;
					// add upper bound, y <= ax+b, -y+ax >= -b
					error = GRBaddconstr(model, 2, ind2, val2, GRB_GREATER_EQUAL, -over_b, NULL);
					handle_gurobi_error(error, env);
				}
				// update model
				error = GRBupdatemodel(model);
				handle_gurobi_error(error, env);
			}
		}
		else{
			// current layer is affine layer
			for(j=0; j < num_cur_neurons; j++){
				neuron_t * affine_node = cur_neurons[j];
				expr_t * affine_expr = affine_node->lexpr;
				size_t num_pre_neurons = affine_expr->size;
				assert(affine_expr->type==DENSE);
				error = GRBaddvar(model, 0, NULL, NULL, 0.0, -affine_node->lb, affine_node->ub, GRB_CONTINUOUS, NULL);
				handle_gurobi_error(error, env);
				int ind[num_pre_neurons+1];
				double val[num_pre_neurons+1];
				for(n=0; n < num_pre_neurons; n++){
					ind[n] = defined_var_start_idx + n;
					val[n] = affine_expr->sup_coeff[n];
				}
				ind[num_pre_neurons] = layer_var_start_idx[i] + j;
				val[num_pre_neurons] = -1.0;
				error = GRBaddconstr(model, num_pre_neurons+1, ind, val, GRB_EQUAL, affine_expr->inf_cst, NULL);
				handle_gurobi_error(error, env);
				// update model
				error = GRBupdatemodel(model);
				handle_gurobi_error(error, env);
			}
		}
	}


	int var_start_idx = layer_var_start_idx[numlayers - 1];
	error = GRBsetdblattrelement(model, "Obj", var_start_idx+ground_truth_label, 1.0);
	handle_gurobi_error(error, env);
	error = GRBsetdblattrelement(model, "Obj", var_start_idx+poten_cex, -1.0);
	handle_gurobi_error(error, env);
	error = GRBsetintattr(model, "ModelSense", -1);
	handle_gurobi_error(error, env);
	error = GRBupdatemodel(model);
	handle_gurobi_error(error, env);
	error = GRBoptimize(model);
	handle_gurobi_error(error, env);
	double lb_deviation;
	error = GRBgetdblattr(model, GRB_DBL_ATTR_OBJVAL, &lb_deviation);
	handle_gurobi_error(error, env);
	printf("Before adding new constraints, the deviation is %.6f\n", lb_deviation);
	error = GRBsetdblattrelement(model, "Obj", var_start_idx+ground_truth_label, 0.0);
	handle_gurobi_error(error, env);
	error = GRBsetdblattrelement(model, "Obj", var_start_idx+poten_cex, 0.0);
	handle_gurobi_error(error, env);

	// Adding new constraints regarding 2 neurons
	for(i=0; i < numlayers; i++){
		int unstable_relu_count = 0;
		layer_t * cur_layer = fp->layers[i];
		if(!cur_layer->is_activation && (i < numlayers-1) && fp->layers[i+1]->is_activation){
			layer_t * next_layer = fp->layers[i+1];
			assert(cur_layer->dims == next_layer->dims);
			neuron_t ** relu_neurons = next_layer->neurons;
			neuron_t ** input_neurons = cur_layer->neurons;
			// Pair information
			int pair_node[cur_layer->dims];
			int relu_unstable[next_layer->dims];
			for(j=0; j < cur_layer->dims; j++){
				pair_node[j] = -1;
				relu_unstable[j] = 0;
				if(relu_neurons[j]->ub!=0.0 && relu_neurons[j]->lb>=0){
					relu_unstable[j] = 1;
				}
			}
			// Find the best pair (non-duplicate) for each unstable neuron
			for(j=0; j < cur_layer->dims; j++){
				if(relu_unstable[j] == 1){
					int max_count = -1;
					for(k=0; k < cur_layer->dims; k++){
						if((k != j) && relu_unstable[k] && (j != pair_node[k])){
							assert(input_neurons[k]->lexpr->size == input_neurons[j]->lexpr->size);
							int count = 0;
							for(n=0; n < input_neurons[k]->lexpr->size; n++){
								if(input_neurons[k]->lexpr->sup_coeff[n] * input_neurons[j]->lexpr->sup_coeff[n] < 0){
									count ++;
								}
							}
							if(count > max_count){
								max_count = count;
								pair_node[j] = k;
							}
						}
					}
					double solved_ub;
					printf("The combination pair is %zu, %d\n", j, pair_node[j]);
					error = GRBsetdblattrelement(model, "Obj", layer_var_start_idx[i]+pair_node[j], 1.0);
					handle_gurobi_error(error, env);
					error = GRBsetdblattrelement(model, "Obj", layer_var_start_idx[i]+j, 1.0);
					handle_gurobi_error(error, env);
					error = GRBsetintattr(model, "ModelSense", -1);
					handle_gurobi_error(error, env);
					error = GRBupdatemodel(model);
					handle_gurobi_error(error, env);
					error = GRBoptimize(model);
					handle_gurobi_error(error, env);
					error = GRBgetdblattr(model, GRB_DBL_ATTR_OBJVAL, &solved_ub);
					handle_gurobi_error(error, env);
					double new_ub = return_max_among_four(0.0, solved_ub + lp_solving_error, input_neurons[pair_node[j]]->ub, input_neurons[j]->ub);
					error = GRBsetdblattrelement(model, "Obj", layer_var_start_idx[i]+pair_node[j], 0.0);
					handle_gurobi_error(error, env);
					error = GRBsetdblattrelement(model, "Obj", layer_var_start_idx[i]+j, 0.0);
					handle_gurobi_error(error, env);

					// The upper bound of directly solving
					error = GRBsetdblattrelement(model, "Obj", layer_var_start_idx[i+1]+pair_node[j], 1.0);
					handle_gurobi_error(error, env);
					error = GRBsetdblattrelement(model, "Obj", layer_var_start_idx[i+1]+j, 1.0);
					handle_gurobi_error(error, env);
					error = GRBsetintattr(model, "ModelSense", -1);
					handle_gurobi_error(error, env);
					error = GRBupdatemodel(model);
					handle_gurobi_error(error, env);
					error = GRBoptimize(model);
					handle_gurobi_error(error, env);
					error = GRBgetdblattr(model, GRB_DBL_ATTR_OBJVAL, &solved_ub);
					double original_bound = solved_ub + lp_solving_error;
					error = GRBsetdblattrelement(model, "Obj", layer_var_start_idx[i+1]+pair_node[j], 0.0);
					handle_gurobi_error(error, env);
					error = GRBsetdblattrelement(model, "Obj", layer_var_start_idx[i+1]+j, 0.0);
					handle_gurobi_error(error, env);
					// Print and compare between the two
					printf("Directly solving returns value %.6f, using 4 cases returns value %.6f\n", original_bound, new_ub);
					if(new_ub < original_bound){
						// Add this new constraint into the model
						int ind[2] = {layer_var_start_idx[i+1]+pair_node[j], layer_var_start_idx[i+1]+j};
						double val[2] = {1.0, 1.0};
						error = GRBaddconstr(model, 2, ind, val, GRB_LESS_EQUAL, new_ub, NULL);
						handle_gurobi_error(error, env);
						error = GRBupdatemodel(model);
						handle_gurobi_error(error, env);
					}
				
				}
			}
		}
	}
	
	error = GRBsetdblattrelement(model, "Obj", var_start_idx+ground_truth_label, 1.0);
	handle_gurobi_error(error, env);
	error = GRBsetdblattrelement(model, "Obj", var_start_idx+poten_cex, -1.0);
	handle_gurobi_error(error, env);
	error = GRBsetintattr(model, "ModelSense", -1);
	handle_gurobi_error(error, env);
	error = GRBupdatemodel(model);
	handle_gurobi_error(error, env);
	error = GRBoptimize(model);
	handle_gurobi_error(error, env);
	error = GRBgetdblattr(model, GRB_DBL_ATTR_OBJVAL, &lb_deviation);
	handle_gurobi_error(error, env);
	printf("After adding new constraints, the deviation is %.6f\n", lb_deviation);
	error = GRBsetdblattrelement(model, "Obj", var_start_idx+ground_truth_label, 0.0);
	handle_gurobi_error(error, env);
	error = GRBsetdblattrelement(model, "Obj", var_start_idx+poten_cex, 0.0);
	handle_gurobi_error(error, env);


	/* Free model */
	GRBfreemodel(model);
	/* Free environment */
	GRBfreeenv(env);
	return false;
}

bool test_two_neurons(elina_manager_t* man, elina_abstract0_t* element, elina_dim_t ground_truth_label, elina_dim_t poten_cex){
	int count, k;
	size_t i, j, n;
	fppoly_t *fp = fppoly_of_abstract0(element);
    size_t numlayers = fp->numlayers;
	double ulp = ldexpl(1.0,-52);
	double lp_solving_error = pow(10.0, -6.0) + ulp;
	int optimstatus;
	/* Create environment */
	GRBenv *env   = NULL;
	GRBmodel *model = NULL;
	int error = 0;	
	error = GRBemptyenv(&env);
	handle_gurobi_error(error, env);
	error = GRBsetintparam(env, "OutputFlag", 0);
	handle_gurobi_error(error, env);
	error = GRBstartenv(env);
	handle_gurobi_error(error, env);
	/* Create an empty model */
	error = GRBnewmodel(env, &model, "refinement_solver", 0, NULL, NULL, NULL, NULL, NULL);
	handle_gurobi_error(error, env);

	// The index starter for variables at different layers
	int layer_var_start_idx[numlayers];

	// fp->input_inf[i], fp->input_sup[i], add the input layer constraints
	layer_var_start_idx[0] = fp->num_pixels;
	for(i=0; i < fp->num_pixels; i++){
		error = GRBaddvar(model, 0, NULL, NULL, 0.0, -fp->input_inf[i], fp->input_sup[i], GRB_CONTINUOUS, NULL);
		handle_gurobi_error(error, env);
	}
	
	// add constaints for each hidden and output layer
	for(i=0; i < numlayers; i++){
		layer_t * cur_layer = fp->layers[i];
		neuron_t ** cur_neurons = cur_layer->neurons;
		size_t num_cur_neurons = cur_layer->dims;
		if(i+1 < numlayers){
			// Set up the variable start index 
			layer_var_start_idx[i+1] = layer_var_start_idx[i] + num_cur_neurons;
		}
		int defined_var_start_idx;
		if(i==0){
			defined_var_start_idx = 0;
		}
		else{
			defined_var_start_idx = layer_var_start_idx[i-1];
		}

		if(cur_layer->is_activation){
			//current layer is ReLU layer, we add the constraints according to RELU behavior
			for(j=0; j < num_cur_neurons; j++){
				// add constraints for each ReLU node
				// need to handle non-stable (two lower constraints will be added) and stable constraint
				neuron_t * relu_node = cur_neurons[j];
				if(relu_node->ub == 0.0){
					// stable unactivated relu nodes
					expr_t * relu_expr = relu_node->lexpr;
					assert(relu_expr->type==SPARSE);
					error = GRBaddvar(model, 0, NULL, NULL, 0.0, -relu_node->lb, relu_node->ub, GRB_CONTINUOUS, NULL);
					handle_gurobi_error(error, env);
				}
				else if(relu_node->lb<0.0){
					// stable activated relu nodes
					expr_t * relu_expr = relu_node->lexpr;
					size_t num_pre_neurons = relu_expr->size;
					assert(relu_expr->type==SPARSE);
					assert(num_pre_neurons==1);
					error = GRBaddvar(model, 0, NULL, NULL, 0.0, -relu_node->lb, relu_node->ub, GRB_CONTINUOUS, NULL);
					handle_gurobi_error(error, env);
					int ind[2] = {layer_var_start_idx[i] + j, defined_var_start_idx + j};
					double val[2] = {-1.0 , relu_expr->sup_coeff[0]};
					error = GRBaddconstr(model, 2, ind, val, GRB_EQUAL, relu_expr->inf_cst, NULL);
					handle_gurobi_error(error, env);
				}
				else{
					// unstable relu nodes, add two lower constarints, and also handle FP error for upper constraint
					expr_t * relu_expr = relu_node->uexpr;
					size_t num_pre_neurons = relu_expr->size;
					assert(relu_expr->type==SPARSE);
					assert(num_pre_neurons==1);
					// The lower bound setting already indicate that relu >=0
					error = GRBaddvar(model, 0, NULL, NULL, 0.0, -relu_node->lb, relu_node->ub, GRB_CONTINUOUS, NULL);
					handle_gurobi_error(error, env);
					int ind[2] = {layer_var_start_idx[i] + j, defined_var_start_idx + j};
					double val[2] = {-1.0 , 1.0};
					// add lower bound, y >= x, -y+x <= 0
					error = GRBaddconstr(model, 2, ind, val, GRB_LESS_EQUAL, 0.0, NULL);
					handle_gurobi_error(error, env);
					int ind2[2] = {layer_var_start_idx[i] + j, defined_var_start_idx + j};
					double over_slope = relu_expr->sup_coeff[0]+ ulp;
					double val2[2] = {-1.0, over_slope};
					int pre = cur_layer->predecessors[0]-1;
					double in_lb = fp->layers[pre]->neurons[j]->lb;
					assert(in_lb>=0);
					double over_b = (fabs(in_lb)+ulp)*over_slope + ulp;
					// add upper bound, y <= ax+b, -y+ax >= -b
					error = GRBaddconstr(model, 2, ind2, val2, GRB_GREATER_EQUAL, -over_b, NULL);
					handle_gurobi_error(error, env);
				}
				// update model
				error = GRBupdatemodel(model);
				handle_gurobi_error(error, env);
			}
		}
		else{
			// current layer is affine layer
			for(j=0; j < num_cur_neurons; j++){
				neuron_t * affine_node = cur_neurons[j];
				expr_t * affine_expr = affine_node->lexpr;
				size_t num_pre_neurons = affine_expr->size;
				assert(affine_expr->type==DENSE);
				error = GRBaddvar(model, 0, NULL, NULL, 0.0, -affine_node->lb, affine_node->ub, GRB_CONTINUOUS, NULL);
				handle_gurobi_error(error, env);
				int ind[num_pre_neurons+1];
				double val[num_pre_neurons+1];
				for(n=0; n < num_pre_neurons; n++){
					ind[n] = defined_var_start_idx + n;
					val[n] = affine_expr->sup_coeff[n];
				}
				ind[num_pre_neurons] = layer_var_start_idx[i] + j;
				val[num_pre_neurons] = -1.0;
				error = GRBaddconstr(model, num_pre_neurons+1, ind, val, GRB_EQUAL, affine_expr->inf_cst, NULL);
				handle_gurobi_error(error, env);
				// update model
				error = GRBupdatemodel(model);
				handle_gurobi_error(error, env);
			}
		}
	}


	// int var_start_idx = layer_var_start_idx[numlayers - 1];
	// error = GRBsetdblattrelement(model, "Obj", var_start_idx+ground_truth_label, 1.0);
	// handle_gurobi_error(error, env);
	// error = GRBsetdblattrelement(model, "Obj", var_start_idx+poten_cex, -1.0);
	// handle_gurobi_error(error, env);
	// error = GRBsetintattr(model, "ModelSense", -1);
	// handle_gurobi_error(error, env);
	// error = GRBupdatemodel(model);
	// handle_gurobi_error(error, env);
	// error = GRBoptimize(model);
	// handle_gurobi_error(error, env);
	// double lb_deviation;
	// error = GRBgetdblattr(model, GRB_DBL_ATTR_OBJVAL, &lb_deviation);
	// handle_gurobi_error(error, env);
	// printf("Before adding new constraints, the deviation is %.6f\n", lb_deviation);
	// error = GRBsetdblattrelement(model, "Obj", var_start_idx+ground_truth_label, 0.0);
	// handle_gurobi_error(error, env);
	// error = GRBsetdblattrelement(model, "Obj", var_start_idx+poten_cex, 0.0);
	// handle_gurobi_error(error, env);

	// Adding new constraints regarding 2 neurons
	int pre_index = 0;
	for(i=0; i < numlayers; i++){
		int unstable_relu_count = 0;
		layer_t * cur_layer = fp->layers[i];
		if(!cur_layer->is_activation && (i < numlayers-1) && fp->layers[i+1]->is_activation){
			layer_t * next_layer = fp->layers[i+1];
			assert(cur_layer->dims == next_layer->dims);
			neuron_t ** relu_neurons = next_layer->neurons;
			neuron_t ** input_neurons = cur_layer->neurons;
			for(j=0; j < cur_layer->dims; j++){
				if(relu_neurons[j]->ub!=0.0 && relu_neurons[j]->lb>=0){
					unstable_relu_count ++;
					if(unstable_relu_count%2 == 1){
						pre_index = j;
					}
					if(unstable_relu_count%2 == 0){
						// The upper bound by considering 4 cases
						double solved_ub;
						error = GRBsetdblattrelement(model, "Obj", layer_var_start_idx[i]+pre_index, 1.0);
						handle_gurobi_error(error, env);
						error = GRBsetdblattrelement(model, "Obj", layer_var_start_idx[i]+j, 1.0);
						handle_gurobi_error(error, env);
						error = GRBsetintattr(model, "ModelSense", -1);
						handle_gurobi_error(error, env);
						error = GRBupdatemodel(model);
						handle_gurobi_error(error, env);
						error = GRBoptimize(model);
						handle_gurobi_error(error, env);
						error = GRBgetdblattr(model, GRB_DBL_ATTR_OBJVAL, &solved_ub);
						handle_gurobi_error(error, env);
						double new_ub = return_max_among_four(0.0, solved_ub + lp_solving_error, input_neurons[pre_index]->ub, input_neurons[j]->ub);
						// expr_print(input_neurons[pre_index]->lexpr);
						count = 0;
						for(k=0; k < input_neurons[pre_index]->lexpr->size; k++){
							if(input_neurons[pre_index]->lexpr->sup_coeff[k] > 0){
								count ++;
							}
						}
						printf("The number of positive coefficients are %d, negative coefficients are %d\n", count, input_neurons[pre_index]->lexpr->size - count);
						count = 0;
						for(k=0; k < input_neurons[j]->lexpr->size; k++){
							if(input_neurons[j]->lexpr->sup_coeff[k] > 0){
								count ++;
							}
						}
						printf("The number of positive coefficients are %d, negative coefficients are %d\n", count, input_neurons[j]->lexpr->size - count);
						
						error = GRBsetdblattrelement(model, "Obj", layer_var_start_idx[i]+pre_index, 0.0);
						handle_gurobi_error(error, env);
						error = GRBsetdblattrelement(model, "Obj", layer_var_start_idx[i]+j, 0.0);
						handle_gurobi_error(error, env);

						// The upper bound of directly solving
						error = GRBsetdblattrelement(model, "Obj", layer_var_start_idx[i+1]+pre_index, 1.0);
						handle_gurobi_error(error, env);
						error = GRBsetdblattrelement(model, "Obj", layer_var_start_idx[i+1]+j, 1.0);
						handle_gurobi_error(error, env);
						error = GRBsetintattr(model, "ModelSense", -1);
						handle_gurobi_error(error, env);
						error = GRBupdatemodel(model);
						handle_gurobi_error(error, env);
						error = GRBoptimize(model);
						handle_gurobi_error(error, env);
						error = GRBgetdblattr(model, GRB_DBL_ATTR_OBJVAL, &solved_ub);
						double original_bound = solved_ub + lp_solving_error;
						error = GRBsetdblattrelement(model, "Obj", layer_var_start_idx[i+1]+pre_index, 0.0);
						handle_gurobi_error(error, env);
						error = GRBsetdblattrelement(model, "Obj", layer_var_start_idx[i+1]+j, 0.0);
						handle_gurobi_error(error, env);
						// Print and compare between the two
						// printf("Directly solving returns value %.6f, using 4 cases returns value %.6f\n", original_bound, new_ub);
						if(new_ub < original_bound){
							// Add this new constraint into the model
							int ind[2] = {layer_var_start_idx[i+1]+pre_index, layer_var_start_idx[i+1]+j};
							double val[2] = {1.0, 1.0};
							error = GRBaddconstr(model, 2, ind, val, GRB_LESS_EQUAL, new_ub, NULL);
							handle_gurobi_error(error, env);
							error = GRBupdatemodel(model);
							handle_gurobi_error(error, env);
						}
					}
				}
			}
		}
	}
	
	// error = GRBsetdblattrelement(model, "Obj", var_start_idx+ground_truth_label, 1.0);
	// handle_gurobi_error(error, env);
	// error = GRBsetdblattrelement(model, "Obj", var_start_idx+poten_cex, -1.0);
	// handle_gurobi_error(error, env);
	// error = GRBsetintattr(model, "ModelSense", -1);
	// handle_gurobi_error(error, env);
	// error = GRBupdatemodel(model);
	// handle_gurobi_error(error, env);
	// error = GRBoptimize(model);
	// handle_gurobi_error(error, env);
	// error = GRBgetdblattr(model, GRB_DBL_ATTR_OBJVAL, &lb_deviation);
	// handle_gurobi_error(error, env);
	// printf("After adding new constraints, the deviation is %.6f\n", lb_deviation);
	// error = GRBsetdblattrelement(model, "Obj", var_start_idx+ground_truth_label, 0.0);
	// handle_gurobi_error(error, env);
	// error = GRBsetdblattrelement(model, "Obj", var_start_idx+poten_cex, 0.0);
	// handle_gurobi_error(error, env);


	/* Free model */
	GRBfreemodel(model);
	/* Free environment */
	GRBfreeenv(env);
	return false;
}

bool test_multi_neurons(elina_manager_t* man, elina_abstract0_t* element, elina_dim_t ground_truth_label, elina_dim_t poten_cex){
	int count, k;
	size_t i, j, n;
	fppoly_t *fp = fppoly_of_abstract0(element);
    size_t numlayers = fp->numlayers;
	double ulp = ldexpl(1.0,-52);
	double lp_solving_error = pow(10.0, -6.0) + ulp;
	int optimstatus;
	/* Create environment */
	GRBenv *env   = NULL;
	GRBmodel *model = NULL;
	int error = 0;	
	error = GRBemptyenv(&env);
	handle_gurobi_error(error, env);
	error = GRBsetintparam(env, "OutputFlag", 0);
	handle_gurobi_error(error, env);
	error = GRBstartenv(env);
	handle_gurobi_error(error, env);
	/* Create an empty model */
	error = GRBnewmodel(env, &model, "refinement_solver", 0, NULL, NULL, NULL, NULL, NULL);
	handle_gurobi_error(error, env);

	// The index starter for variables at different layers
	int layer_var_start_idx[numlayers];

	// fp->input_inf[i], fp->input_sup[i], add the input layer constraints
	layer_var_start_idx[0] = fp->num_pixels;
	for(i=0; i < fp->num_pixels; i++){
		error = GRBaddvar(model, 0, NULL, NULL, 0.0, -fp->input_inf[i], fp->input_sup[i], GRB_CONTINUOUS, NULL);
		handle_gurobi_error(error, env);
	}
	
	// add constaints for each hidden and output layer
	for(i=0; i < numlayers; i++){
		layer_t * cur_layer = fp->layers[i];
		neuron_t ** cur_neurons = cur_layer->neurons;
		size_t num_cur_neurons = cur_layer->dims;
		if(i+1 < numlayers){
			// Set up the variable start index 
			layer_var_start_idx[i+1] = layer_var_start_idx[i] + num_cur_neurons;
		}
		int defined_var_start_idx;
		if(i==0){
			defined_var_start_idx = 0;
		}
		else{
			defined_var_start_idx = layer_var_start_idx[i-1];
		}

		if(cur_layer->is_activation){
			//current layer is ReLU layer, we add the constraints according to RELU behavior
			for(j=0; j < num_cur_neurons; j++){
				// add constraints for each ReLU node
				// need to handle non-stable (two lower constraints will be added) and stable constraint
				neuron_t * relu_node = cur_neurons[j];
				if(relu_node->ub == 0.0){
					// stable unactivated relu nodes
					expr_t * relu_expr = relu_node->lexpr;
					assert(relu_expr->type==SPARSE);
					error = GRBaddvar(model, 0, NULL, NULL, 0.0, -relu_node->lb, relu_node->ub, GRB_CONTINUOUS, NULL);
					handle_gurobi_error(error, env);
				}
				else if(relu_node->lb<0.0){
					// stable activated relu nodes
					expr_t * relu_expr = relu_node->lexpr;
					size_t num_pre_neurons = relu_expr->size;
					assert(relu_expr->type==SPARSE);
					assert(num_pre_neurons==1);
					error = GRBaddvar(model, 0, NULL, NULL, 0.0, -relu_node->lb, relu_node->ub, GRB_CONTINUOUS, NULL);
					handle_gurobi_error(error, env);
					int ind[2] = {layer_var_start_idx[i] + j, defined_var_start_idx + j};
					double val[2] = {-1.0 , relu_expr->sup_coeff[0]};
					error = GRBaddconstr(model, 2, ind, val, GRB_EQUAL, relu_expr->inf_cst, NULL);
					handle_gurobi_error(error, env);
				}
				else{
					// unstable relu nodes, add two lower constarints, and also handle FP error for upper constraint
					expr_t * relu_expr = relu_node->uexpr;
					size_t num_pre_neurons = relu_expr->size;
					assert(relu_expr->type==SPARSE);
					assert(num_pre_neurons==1);
					// The lower bound setting already indicate that relu >=0
					error = GRBaddvar(model, 0, NULL, NULL, 0.0, -relu_node->lb, relu_node->ub, GRB_CONTINUOUS, NULL);
					handle_gurobi_error(error, env);
					int ind[2] = {layer_var_start_idx[i] + j, defined_var_start_idx + j};
					double val[2] = {-1.0 , 1.0};
					// add lower bound, y >= x, -y+x <= 0
					error = GRBaddconstr(model, 2, ind, val, GRB_LESS_EQUAL, 0.0, NULL);
					handle_gurobi_error(error, env);
					int ind2[2] = {layer_var_start_idx[i] + j, defined_var_start_idx + j};
					double over_slope = relu_expr->sup_coeff[0]+ ulp;
					double val2[2] = {-1.0, over_slope};
					int pre = cur_layer->predecessors[0]-1;
					double in_lb = fp->layers[pre]->neurons[j]->lb;
					assert(in_lb>=0);
					double over_b = (fabs(in_lb)+ulp)*over_slope + ulp;
					// add upper bound, y <= ax+b, -y+ax >= -b
					error = GRBaddconstr(model, 2, ind2, val2, GRB_GREATER_EQUAL, -over_b, NULL);
					handle_gurobi_error(error, env);
				}
				// update model
				error = GRBupdatemodel(model);
				handle_gurobi_error(error, env);
			}
		}
		else{
			// current layer is affine layer
			for(j=0; j < num_cur_neurons; j++){
				neuron_t * affine_node = cur_neurons[j];
				expr_t * affine_expr = affine_node->lexpr;
				size_t num_pre_neurons = affine_expr->size;
				assert(affine_expr->type==DENSE);
				error = GRBaddvar(model, 0, NULL, NULL, 0.0, -affine_node->lb, affine_node->ub, GRB_CONTINUOUS, NULL);
				handle_gurobi_error(error, env);
				int ind[num_pre_neurons+1];
				double val[num_pre_neurons+1];
				for(n=0; n < num_pre_neurons; n++){
					ind[n] = defined_var_start_idx + n;
					val[n] = affine_expr->sup_coeff[n];
				}
				ind[num_pre_neurons] = layer_var_start_idx[i] + j;
				val[num_pre_neurons] = -1.0;
				error = GRBaddconstr(model, num_pre_neurons+1, ind, val, GRB_EQUAL, affine_expr->inf_cst, NULL);
				handle_gurobi_error(error, env);
				// update model
				error = GRBupdatemodel(model);
				handle_gurobi_error(error, env);
			}
		}
	}


	int var_start_idx = layer_var_start_idx[numlayers - 1];
	error = GRBsetdblattrelement(model, "Obj", var_start_idx+ground_truth_label, 1.0);
	handle_gurobi_error(error, env);
	error = GRBsetdblattrelement(model, "Obj", var_start_idx+poten_cex, -1.0);
	handle_gurobi_error(error, env);
	error = GRBsetintattr(model, "ModelSense", 1);
	handle_gurobi_error(error, env);
	error = GRBupdatemodel(model);
	handle_gurobi_error(error, env);
	error = GRBoptimize(model);
	handle_gurobi_error(error, env);
	double lb_deviation;
	error = GRBgetdblattr(model, GRB_DBL_ATTR_OBJVAL, &lb_deviation);
	handle_gurobi_error(error, env);
	printf("Before adding new constraints, the deviation is %.6f\n", lb_deviation);
	error = GRBsetdblattrelement(model, "Obj", var_start_idx+ground_truth_label, 0.0);
	handle_gurobi_error(error, env);
	error = GRBsetdblattrelement(model, "Obj", var_start_idx+poten_cex, 0.0);
	handle_gurobi_error(error, env);

	// Adding two constraints regarding all unstable neurons in one layer
	int pre_index = 0;
	for(i=0; i < numlayers; i++){
		layer_t * cur_layer = fp->layers[i];
		if(!cur_layer->is_activation && (i < numlayers-1) && fp->layers[i+1]->is_activation){
			layer_t * next_layer = fp->layers[i+1];
			assert(cur_layer->dims == next_layer->dims);
			neuron_t ** relu_neurons = next_layer->neurons;
			neuron_t ** input_neurons = cur_layer->neurons;
			for(j=0; j < cur_layer->dims; j++){
				if(relu_neurons[j]->ub!=0.0 && relu_neurons[j]->lb>=0){
					// error = GRBsetdblattrelement(model, "Obj", layer_var_start_idx[i]+j, relu_neurons[j]->uexpr->sup_coeff[0]);
					error = GRBsetdblattrelement(model, "Obj", layer_var_start_idx[i]+j, 1.0);
					handle_gurobi_error(error, env);
				}
			}
			double solved_ub;
			// error = GRBsetintattr(model, "ModelSense", -1);
			error = GRBsetintattr(model, "ModelSense", 1);
			handle_gurobi_error(error, env);
			error = GRBupdatemodel(model);
			handle_gurobi_error(error, env);
			error = GRBoptimize(model);
			handle_gurobi_error(error, env);
			error = GRBgetdblattr(model, GRB_DBL_ATTR_OBJVAL, &solved_ub);
			double new_ub = solved_ub + lp_solving_error;

			for(j=0; j < cur_layer->dims; j++){
				if(relu_neurons[j]->ub!=0.0 && relu_neurons[j]->lb>=0){
					// new_ub = new_ub + relu_neurons[j]->uexpr->sup_cst;
					error = GRBsetdblattrelement(model, "Obj", layer_var_start_idx[i]+j, 0.0);
					handle_gurobi_error(error, env);
					error = GRBsetdblattrelement(model, "Obj", layer_var_start_idx[i+1]+j, 1.0);
					handle_gurobi_error(error, env);
				}
			}
			error = GRBupdatemodel(model);
			handle_gurobi_error(error, env);
			error = GRBoptimize(model);
			handle_gurobi_error(error, env);
			error = GRBgetdblattr(model, GRB_DBL_ATTR_OBJVAL, &solved_ub);
			double orignal_ub = solved_ub + lp_solving_error;
			printf("The original solved lower bound is %.6f, the new lower bound is %.6f\n", orignal_ub, new_ub);
			for(j=0; j < cur_layer->dims; j++){
				if(relu_neurons[j]->ub!=0.0 && relu_neurons[j]->lb>=0){
					error = GRBsetdblattrelement(model, "Obj", layer_var_start_idx[i+1]+j, 0.0);
					handle_gurobi_error(error, env);
				}
			}
		}
	}
	
	error = GRBsetdblattrelement(model, "Obj", var_start_idx+ground_truth_label, 1.0);
	handle_gurobi_error(error, env);
	error = GRBsetdblattrelement(model, "Obj", var_start_idx+poten_cex, -1.0);
	handle_gurobi_error(error, env);
	error = GRBsetintattr(model, "ModelSense", 1);
	handle_gurobi_error(error, env);
	error = GRBupdatemodel(model);
	handle_gurobi_error(error, env);
	error = GRBoptimize(model);
	handle_gurobi_error(error, env);
	error = GRBgetdblattr(model, GRB_DBL_ATTR_OBJVAL, &lb_deviation);
	handle_gurobi_error(error, env);
	printf("After adding new constraints, the deviation is %.6f\n", lb_deviation);
	error = GRBsetdblattrelement(model, "Obj", var_start_idx+ground_truth_label, 0.0);
	handle_gurobi_error(error, env);
	error = GRBsetdblattrelement(model, "Obj", var_start_idx+poten_cex, 0.0);
	handle_gurobi_error(error, env);


	/* Free model */
	GRBfreemodel(model);
	/* Free environment */
	GRBfreeenv(env);
	return false;
}

bool is_spurious_pair_two_neurons(elina_manager_t* man, elina_abstract0_t* element, elina_dim_t ground_truth_label, elina_dim_t poten_cex, int * spurious_list, int spurious_count, int MAX_ITER){
	// SMUPoly paper with additional constraints regarding two neurons
	int count, k;
	clock_t func_begin = clock();
	size_t i, j, n;
	fppoly_t *fp = fppoly_of_abstract0(element);
    size_t numlayers = fp->numlayers;
	double ulp = ldexpl(1.0,-52);
	double lp_solving_error = pow(10.0, -6.0) + ulp;
	int optimstatus;
	for(i=0; i < fp->num_pixels; i++){
		// set the input neurons back to the original input space
		fp->input_inf[i] = fp->original_input_inf[i];
		fp->input_sup[i] = fp->original_input_sup[i];
	}
	clear_neurons_status(man, element);
	// Refine for MAX_ITER times
	for(count = 0; count < MAX_ITER; count++){
		run_deeppoly(man, element);
		printf("Refinement iteration %d\n", count+1);
		/* Create environment */
  		GRBenv *env   = NULL;
  		GRBmodel *model = NULL;
		int error = 0;	
		error = GRBemptyenv(&env);
		handle_gurobi_error(error, env);
		error = GRBsetintparam(env, "OutputFlag", 0);
		handle_gurobi_error(error, env);
		error = GRBstartenv(env);
		handle_gurobi_error(error, env);
		/* Create an empty model */
		error = GRBnewmodel(env, &model, "refinement_solver", 0, NULL, NULL, NULL, NULL, NULL);
		handle_gurobi_error(error, env);

		// The index starter for variables at different layers
		int layer_var_start_idx[numlayers];

		// fp->input_inf[i], fp->input_sup[i], add the input layer constraints
		layer_var_start_idx[0] = fp->num_pixels;
		for(i=0; i < fp->num_pixels; i++){
			error = GRBaddvar(model, 0, NULL, NULL, 0.0, -fp->input_inf[i], fp->input_sup[i], GRB_CONTINUOUS, NULL);
        	handle_gurobi_error(error, env);
		}
		
		// add constaints for each hidden and output layer
		for(i=0; i < numlayers; i++){
			layer_t * cur_layer = fp->layers[i];
			neuron_t ** cur_neurons = cur_layer->neurons;
			size_t num_cur_neurons = cur_layer->dims;
			if(i+1 < numlayers){
				// Set up the variable start index 
				layer_var_start_idx[i+1] = layer_var_start_idx[i] + num_cur_neurons;
			}
			int defined_var_start_idx;
			if(i==0){
				defined_var_start_idx = 0;
			}
			else{
				defined_var_start_idx = layer_var_start_idx[i-1];
			}

			if(cur_layer->is_activation){
				//current layer is ReLU layer, we add the constraints according to RELU behavior
				for(j=0; j < num_cur_neurons; j++){
					// add constraints for each ReLU node
					// need to handle non-stable (two lower constraints will be added) and stable constraint
					neuron_t * relu_node = cur_neurons[j];
					if(relu_node->ub == 0.0){
						// stable unactivated relu nodes
						expr_t * relu_expr = relu_node->lexpr;
						assert(relu_expr->type==SPARSE);
						error = GRBaddvar(model, 0, NULL, NULL, 0.0, -relu_node->lb, relu_node->ub, GRB_CONTINUOUS, NULL);
						handle_gurobi_error(error, env);
					}
					else if(relu_node->lb<0.0){
						// stable activated relu nodes
						expr_t * relu_expr = relu_node->lexpr;
						size_t num_pre_neurons = relu_expr->size;
						assert(relu_expr->type==SPARSE);
						assert(num_pre_neurons==1);
						error = GRBaddvar(model, 0, NULL, NULL, 0.0, -relu_node->lb, relu_node->ub, GRB_CONTINUOUS, NULL);
						handle_gurobi_error(error, env);
						int ind[2] = {layer_var_start_idx[i] + j, defined_var_start_idx + j};
						double val[2] = {-1.0 , relu_expr->sup_coeff[0]};
						error = GRBaddconstr(model, 2, ind, val, GRB_EQUAL, relu_expr->inf_cst, NULL);
						handle_gurobi_error(error, env);
					}
					else{
						// unstable relu nodes, add two lower constarints, and also handle FP error for upper constraint
						expr_t * relu_expr = relu_node->uexpr;
						size_t num_pre_neurons = relu_expr->size;
						assert(relu_expr->type==SPARSE);
						assert(num_pre_neurons==1);
						// The lower bound setting already indicate that relu >=0
						error = GRBaddvar(model, 0, NULL, NULL, 0.0, -relu_node->lb, relu_node->ub, GRB_CONTINUOUS, NULL);
						handle_gurobi_error(error, env);
						int ind[2] = {layer_var_start_idx[i] + j, defined_var_start_idx + j};
						double val[2] = {-1.0 , 1.0};
						// add lower bound, y >= x, -y+x <= 0
						error = GRBaddconstr(model, 2, ind, val, GRB_LESS_EQUAL, 0.0, NULL);
						handle_gurobi_error(error, env);
						int ind2[2] = {layer_var_start_idx[i] + j, defined_var_start_idx + j};
						double over_slope = relu_expr->sup_coeff[0]+ ulp;
						double val2[2] = {-1.0, over_slope};
						int pre = cur_layer->predecessors[0]-1;
						double in_lb = fp->layers[pre]->neurons[j]->lb;
						assert(in_lb>=0);
						double over_b = (fabs(in_lb)+ulp)*over_slope + ulp;
						// add upper bound, y <= ax+b, -y+ax >= -b
						error = GRBaddconstr(model, 2, ind2, val2, GRB_GREATER_EQUAL, -over_b, NULL);
						handle_gurobi_error(error, env);
					}
					// update model
					error = GRBupdatemodel(model);
					handle_gurobi_error(error, env);
				}
			}
			else{
				// current layer is affine layer
				for(j=0; j < num_cur_neurons; j++){
					neuron_t * affine_node = cur_neurons[j];
					expr_t * affine_expr = affine_node->lexpr;
					size_t num_pre_neurons = affine_expr->size;
					assert(affine_expr->type==DENSE);
					error = GRBaddvar(model, 0, NULL, NULL, 0.0, -affine_node->lb, affine_node->ub, GRB_CONTINUOUS, NULL);
					handle_gurobi_error(error, env);
					int ind[num_pre_neurons+1];
  					double val[num_pre_neurons+1];
					for(n=0; n < num_pre_neurons; n++){
						ind[n] = defined_var_start_idx + n;
						val[n] = affine_expr->sup_coeff[n];
					}
					ind[num_pre_neurons] = layer_var_start_idx[i] + j;
					val[num_pre_neurons] = -1.0;
					error = GRBaddconstr(model, num_pre_neurons+1, ind, val, GRB_EQUAL, affine_expr->inf_cst, NULL);
					handle_gurobi_error(error, env);
					// update model
					error = GRBupdatemodel(model);
					handle_gurobi_error(error, env);
				}
			}
		}

		// add constraints for previously spurious labels
		for(k=0; k < spurious_count; k++){
			int spu_label = spurious_list[k];
			// we have out[ground_truth_label] - out[spu_label] > 0, for practical concern, we expand to >=
			int var_start_idx = layer_var_start_idx[numlayers - 1];
			int ind[2] = {var_start_idx+ground_truth_label,var_start_idx+spu_label};
			double val[2] = {1.0, -1.0};
			error = GRBaddconstr(model, 2, ind, val, GRB_GREATER_EQUAL, 0.0, NULL);
			handle_gurobi_error(error, env);
		}

		// add constraints regarding the current potential adversarial labels we try to eliminate, out[ground_truth_label] - out[spu_label] <= 0
		int var_start_idx = layer_var_start_idx[numlayers - 1];
		int ind[2] = {var_start_idx+ground_truth_label,var_start_idx+poten_cex};
		double val[2] = {1.0, -1.0};
		error = GRBaddconstr(model, 2, ind, val, GRB_LESS_EQUAL, 0.0, NULL);
		// update model
		error = GRBupdatemodel(model);
		handle_gurobi_error(error, env);

		// Simply check the feasibility, without objective function, if infeasible, then successfully prove spurious, return True
		error = GRBoptimize(model);
		handle_gurobi_error(error, env);
		/* Capture solution information */
		error = GRBgetintattr(model, GRB_INT_ATTR_STATUS, &optimstatus);
		handle_gurobi_error(error, env);
		if(optimstatus == GRB_INFEASIBLE){
			GRBfreemodel(model);
			GRBfreeenv(env);
			clock_t func_end = clock();
			double func_spent = (double)(func_end - func_begin) / CLOCKS_PER_SEC;
			printf("Refine succesfully at %d-th iteration, total time is %f\n", count+1, func_spent);
			return true;
		}
		
		// If detect feasibility, add constraints regarding two neurons and do the solving
		// To enumerate all of them is too costly
		for(i=0; i < numlayers; i++){
			int unstable_relu_count = 0;
			layer_t * cur_layer = fp->layers[i];
			if(!cur_layer->is_activation && (i < numlayers-1) && fp->layers[i+1]->is_activation){
				layer_t * next_layer = fp->layers[i+1];
				assert(cur_layer->dims == next_layer->dims);
				neuron_t ** relu_neurons = next_layer->neurons;
				neuron_t ** input_neurons = cur_layer->neurons;
				// Pair information
				int unstable_index[next_layer->dims];
				for(j=0; j < cur_layer->dims; j++){
					if(relu_neurons[j]->ub!=0.0 && relu_neurons[j]->lb>=0){
						unstable_index[unstable_relu_count] = j;
						unstable_relu_count ++;
					}
				}
				// Find the best pair (non-duplicate) for each unstable neuron
				for(j=0; j < unstable_relu_count; j++){
					for(n=j+1; n < unstable_relu_count; n++){
						int index1 = unstable_index[j];
						int index2 = unstable_index[n];
						double solved_ub;
						error = GRBsetdblattrelement(model, "Obj", layer_var_start_idx[i]+index1, 1.0);
						handle_gurobi_error(error, env);
						error = GRBsetdblattrelement(model, "Obj", layer_var_start_idx[i]+index2, 1.0);
						handle_gurobi_error(error, env);
						error = GRBsetintattr(model, "ModelSense", -1);
						handle_gurobi_error(error, env);
						error = GRBupdatemodel(model);
						handle_gurobi_error(error, env);
						error = GRBoptimize(model);
						handle_gurobi_error(error, env);
						// With new constraints added in the last iteration, need to check feasibility again
						error = GRBgetintattr(model, GRB_INT_ATTR_STATUS, &optimstatus);
						handle_gurobi_error(error, env);
						if(optimstatus == GRB_INFEASIBLE){
							GRBfreemodel(model);
							GRBfreeenv(env);
							clock_t func_end = clock();
							double func_spent = (double)(func_end - func_begin) / CLOCKS_PER_SEC;
							printf("Refine succesfully at %d-th iteration, where we add additional constraints, total time is %f\n", count+1, func_spent);
							return true;
						}
						error = GRBgetdblattr(model, GRB_DBL_ATTR_OBJVAL, &solved_ub);
						handle_gurobi_error(error, env);
						double new_ub = return_max_among_four(0.0, solved_ub + lp_solving_error, input_neurons[index1]->ub, input_neurons[index2]->ub);
						error = GRBsetdblattrelement(model, "Obj", layer_var_start_idx[i]+index1, 0.0);
						handle_gurobi_error(error, env);
						error = GRBsetdblattrelement(model, "Obj", layer_var_start_idx[i]+index2, 0.0);
						handle_gurobi_error(error, env);
						// Add this new constraint into the model
						int ind[2] = {layer_var_start_idx[i+1]+index1, layer_var_start_idx[i+1]+index2};
						double val[2] = {1.0, 1.0};
						error = GRBaddconstr(model, 2, ind, val, GRB_LESS_EQUAL, new_ub, NULL);
						handle_gurobi_error(error, env);
						error = GRBupdatemodel(model);
						handle_gurobi_error(error, env);
					}
				}
			}
		}
	
		// solve for interval of input neurons
		for(i=0; i < fp->num_pixels; i++){
			double solved_lb, solved_ub;
			error = GRBsetdblattrelement(model, "Obj", i, 1.0);
        	handle_gurobi_error(error, env);
			// ModelSense, default value 1 indicates minimization, and -1 meaning maximization
			// lower bound solving
			error = GRBsetintattr(model, "ModelSense", 1);
    		handle_gurobi_error(error, env);
			error = GRBupdatemodel(model);
			handle_gurobi_error(error, env);
			error = GRBoptimize(model);
			handle_gurobi_error(error, env);
			error = GRBgetdblattr(model, GRB_DBL_ATTR_OBJVAL, &solved_lb);
			handle_gurobi_error(error, env);
			// upper bound solving
			error = GRBsetintattr(model, "ModelSense", -1);
    		handle_gurobi_error(error, env);
			error = GRBupdatemodel(model);
			handle_gurobi_error(error, env);
			error = GRBoptimize(model);
			handle_gurobi_error(error, env);
			error = GRBgetdblattr(model, GRB_DBL_ATTR_OBJVAL, &solved_ub);
			handle_gurobi_error(error, env);
			// Update the corresponding input lower and upper bound for next deeppoly execution
			// if(((solved_lb - lp_solving_error) > -fp->input_inf[i]) || ((solved_ub + lp_solving_error) < fp->input_sup[i])){
			// 	printf("The start layer neuron %zu originally has interval [%.4f, %.4f]\n", i, -fp->input_inf[i], fp->input_sup[i]);
			// 	printf("The start layer neuron %zu was updates to [%.4f, %.4f]\n", i, (solved_lb - lp_solving_error), solved_ub + lp_solving_error);
			// }
			fp->input_inf[i] = -(solved_lb - lp_solving_error);
			fp->input_sup[i] = solved_ub + lp_solving_error;
			// revert obj coeff back to 0
			error = GRBsetdblattrelement(model, "Obj", i, 0.0);
			handle_gurobi_error(error, env);
		}

		// solve for relu interval of unstable relu nodes
		int relu_refine_count = 0;
		for(i=0; i < numlayers; i++){
			layer_t * cur_layer = fp->layers[i];
			if(!cur_layer->is_activation && (i < numlayers-1) && fp->layers[i+1]->is_activation){
				layer_t * next_layer = fp->layers[i+1];
				assert(cur_layer->dims == next_layer->dims);
				neuron_t ** relu_neurons = next_layer->neurons;
				for(j=0; j < cur_layer->dims; j++){
					if(relu_neurons[j]->ub!=0.0 && relu_neurons[j]->lb>=0){
						double solved_lb, solved_ub;
						error = GRBsetdblattrelement(model, "Obj", layer_var_start_idx[i]+j, 1.0);
						handle_gurobi_error(error, env);
						// ModelSense, default value 1 indicates minimization, and -1 meaning maximization
						// lower bound solving
						error = GRBsetintattr(model, "ModelSense", 1);
						handle_gurobi_error(error, env);
						error = GRBupdatemodel(model);
						handle_gurobi_error(error, env);
						error = GRBoptimize(model);
						handle_gurobi_error(error, env);
						error = GRBgetdblattr(model, GRB_DBL_ATTR_OBJVAL, &solved_lb);
						handle_gurobi_error(error, env);
						// update relu node lower bound
						double lp_solving_error = pow(10.0, -6.0) + ulp;
						cur_layer->neurons[j]->lb = fmin(-(solved_lb - lp_solving_error), cur_layer->neurons[j]->lb);
						if(cur_layer->neurons[j]->lb<0){
							relu_refine_count ++;
						}
						// upper bound solving
						error = GRBsetintattr(model, "ModelSense", -1);
						handle_gurobi_error(error, env);
						error = GRBupdatemodel(model);
						handle_gurobi_error(error, env);
						error = GRBoptimize(model);
						handle_gurobi_error(error, env);
						error = GRBgetdblattr(model, GRB_DBL_ATTR_OBJVAL, &solved_ub);
						handle_gurobi_error(error, env);
						// update relu node upper bound
						cur_layer->neurons[j]->ub = fmin(solved_ub + lp_solving_error, cur_layer->neurons[j]->ub);
						if(cur_layer->neurons[j]->ub<=0){
							relu_refine_count ++;
						}
						error = GRBsetdblattrelement(model, "Obj", layer_var_start_idx[i]+j, 0.0);
						handle_gurobi_error(error, env);
					}
				}
			}
		}
		printf("Refreshed ReLU nodes: %d\n",relu_refine_count);
		/* Free model */
  		GRBfreemodel(model);
  		/* Free environment */
  		GRBfreeenv(env);
	}
	if(optimstatus == GRB_OPTIMAL){
		printf("Need to do adversarial example finding or quantitative robustness\n");
	}
	clock_t func_end = clock();
	double func_spent = (double)(func_end - func_begin) / CLOCKS_PER_SEC;
	printf("fail refinement, # total iteration is %d,total time is %f\n", count, func_spent);
	return false;
}

bool is_spurious_pair_three_neurons(elina_manager_t* man, elina_abstract0_t* element, elina_dim_t ground_truth_label, elina_dim_t poten_cex, int * spurious_list, int spurious_count, int MAX_ITER){
	// We consider three relus together, and our 4 LP solving can give us 4 additional constraints
	int count, k;
	clock_t func_begin = clock();
	size_t i, j, n;
	fppoly_t *fp = fppoly_of_abstract0(element);
    size_t numlayers = fp->numlayers;
	double ulp = ldexpl(1.0,-52);
	double lp_solving_error = pow(10.0, -6.0) + ulp;
	int optimstatus;
	for(i=0; i < fp->num_pixels; i++){
		// set the input neurons back to the original input space
		fp->input_inf[i] = fp->original_input_inf[i];
		fp->input_sup[i] = fp->original_input_sup[i];
	}
	clear_neurons_status(man, element);
	for(count = 0; count < MAX_ITER; count++){
		run_deeppoly(man, element);
		printf("Refinement iteration %d\n", count+1);
		/* Create environment */
  		GRBenv *env   = NULL;
  		GRBmodel *model = NULL;
		int error = 0;	
		error = GRBemptyenv(&env);
		handle_gurobi_error(error, env);
		error = GRBsetintparam(env, "OutputFlag", 0);
		handle_gurobi_error(error, env);
		error = GRBstartenv(env);
		handle_gurobi_error(error, env);
		/* Create an empty model */
		error = GRBnewmodel(env, &model, "refinement_solver", 0, NULL, NULL, NULL, NULL, NULL);
		handle_gurobi_error(error, env);

		// The index starter for variables at different layers
		int layer_var_start_idx[numlayers];
		// fp->input_inf[i], fp->input_sup[i], add the input layer constraints
		layer_var_start_idx[0] = fp->num_pixels;
		for(i=0; i < fp->num_pixels; i++){
			error = GRBaddvar(model, 0, NULL, NULL, 0.0, -fp->input_inf[i], fp->input_sup[i], GRB_CONTINUOUS, NULL);
        	handle_gurobi_error(error, env);
		}
		
		// add constaints for each hidden and output layer
		for(i=0; i < numlayers; i++){
			layer_t * cur_layer = fp->layers[i];
			neuron_t ** cur_neurons = cur_layer->neurons;
			size_t num_cur_neurons = cur_layer->dims;
			if(i+1 < numlayers){
				// Set up the variable start index 
				layer_var_start_idx[i+1] = layer_var_start_idx[i] + num_cur_neurons;
			}
			int defined_var_start_idx;
			if(i==0){
				defined_var_start_idx = 0;
			}
			else{
				defined_var_start_idx = layer_var_start_idx[i-1];
			}

			if(cur_layer->is_activation){
				//current layer is ReLU layer, we add the constraints according to RELU behavior
				for(j=0; j < num_cur_neurons; j++){
					// add constraints for each ReLU node
					// need to handle non-stable (two lower constraints will be added) and stable constraint
					neuron_t * relu_node = cur_neurons[j];
					if(relu_node->ub == 0.0){
						// stable unactivated relu nodes
						expr_t * relu_expr = relu_node->lexpr;
						assert(relu_expr->type==SPARSE);
						error = GRBaddvar(model, 0, NULL, NULL, 0.0, -relu_node->lb, relu_node->ub, GRB_CONTINUOUS, NULL);
						handle_gurobi_error(error, env);
					}
					else if(relu_node->lb<0.0){
						// stable activated relu nodes
						expr_t * relu_expr = relu_node->lexpr;
						size_t num_pre_neurons = relu_expr->size;
						assert(relu_expr->type==SPARSE);
						assert(num_pre_neurons==1);
						error = GRBaddvar(model, 0, NULL, NULL, 0.0, -relu_node->lb, relu_node->ub, GRB_CONTINUOUS, NULL);
						handle_gurobi_error(error, env);
						int ind[2] = {layer_var_start_idx[i] + j, defined_var_start_idx + j};
						double val[2] = {-1.0 , relu_expr->sup_coeff[0]};
						error = GRBaddconstr(model, 2, ind, val, GRB_EQUAL, relu_expr->inf_cst, NULL);
						handle_gurobi_error(error, env);
					}
					else{
						// unstable relu nodes, add two lower constarints, and also handle FP error for upper constraint
						expr_t * relu_expr = relu_node->uexpr;
						size_t num_pre_neurons = relu_expr->size;
						assert(relu_expr->type==SPARSE);
						assert(num_pre_neurons==1);
						// The lower bound setting already indicate that relu >=0
						error = GRBaddvar(model, 0, NULL, NULL, 0.0, -relu_node->lb, relu_node->ub, GRB_CONTINUOUS, NULL);
						handle_gurobi_error(error, env);
						int ind[2] = {layer_var_start_idx[i] + j, defined_var_start_idx + j};
						double val[2] = {-1.0 , 1.0};
						// add lower bound, y >= x, -y+x <= 0
						error = GRBaddconstr(model, 2, ind, val, GRB_LESS_EQUAL, 0.0, NULL);
						handle_gurobi_error(error, env);
						int ind2[2] = {layer_var_start_idx[i] + j, defined_var_start_idx + j};
						double over_slope = relu_expr->sup_coeff[0]+ ulp;
						double val2[2] = {-1.0, over_slope};
						int pre = cur_layer->predecessors[0]-1;
						double in_lb = fp->layers[pre]->neurons[j]->lb;
						assert(in_lb>=0);
						double over_b = (fabs(in_lb)+ulp)*over_slope + ulp;
						// add upper bound, y <= ax+b, -y+ax >= -b
						error = GRBaddconstr(model, 2, ind2, val2, GRB_GREATER_EQUAL, -over_b, NULL);
						handle_gurobi_error(error, env);
					}
					// update model
					error = GRBupdatemodel(model);
					handle_gurobi_error(error, env);
				}
			}
			else{
				// current layer is affine layer
				for(j=0; j < num_cur_neurons; j++){
					neuron_t * affine_node = cur_neurons[j];
					expr_t * affine_expr = affine_node->lexpr;
					size_t num_pre_neurons = affine_expr->size;
					assert(affine_expr->type==DENSE);
					error = GRBaddvar(model, 0, NULL, NULL, 0.0, -affine_node->lb, affine_node->ub, GRB_CONTINUOUS, NULL);
					handle_gurobi_error(error, env);
					int ind[num_pre_neurons+1];
  					double val[num_pre_neurons+1];
					for(n=0; n < num_pre_neurons; n++){
						ind[n] = defined_var_start_idx + n;
						val[n] = affine_expr->sup_coeff[n];
					}
					ind[num_pre_neurons] = layer_var_start_idx[i] + j;
					val[num_pre_neurons] = -1.0;
					error = GRBaddconstr(model, num_pre_neurons+1, ind, val, GRB_EQUAL, affine_expr->inf_cst, NULL);
					handle_gurobi_error(error, env);
					// update model
					error = GRBupdatemodel(model);
					handle_gurobi_error(error, env);
				}
			}
		}

		// If detect feasibility, add constraints regarding 3 neurons and do the solving
		for(i=0; i < numlayers; i++){
			int unstable_relu_count = 0;
			layer_t * cur_layer = fp->layers[i];
			if(!cur_layer->is_activation && (i < numlayers-1) && fp->layers[i+1]->is_activation){
				layer_t * next_layer = fp->layers[i+1];
				assert(cur_layer->dims == next_layer->dims);
				neuron_t ** relu_neurons = next_layer->neurons;
				neuron_t ** input_neurons = cur_layer->neurons;
				int unstable_index[next_layer->dims];
				for(j=0; j < cur_layer->dims; j++){
					if(relu_neurons[j]->ub!=0.0 && relu_neurons[j]->lb>=0){
						unstable_index[unstable_relu_count] = j;
						unstable_relu_count ++;
					}
				}
				// Update the model to be multi-scenario
				// printf("the number of unstable relu nodes is %d\n", unstable_relu_count);
				if(unstable_relu_count >= 3){
					error = GRBsetintattr(model, "NumScenarios", (unstable_relu_count - 2)*3 + 1);
					handle_gurobi_error(error, env);
					error = GRBupdatemodel(model);
					handle_gurobi_error(error, env);
				}
				for(j=0; j < unstable_relu_count - 2; j++){
					int index1 = unstable_index[j];
					int index2 = unstable_index[j+1];
					int index3 = unstable_index[j+2];
					// printf("Layer num is %zu, 3 affine nodes are %d, %d, %d\n", i, index1, index2, index3);
					// The three neurons are under indexes index1, index2, index3
					// compute z1+z2 firstly
					if(j == 0){
						error = GRBsetintparam(GRBgetenv(model), "ScenarioNumber", 0);
						handle_gurobi_error(error, GRBgetenv(model));
						error = GRBsetdblattrelement(model, "ScenNObj", layer_var_start_idx[i]+index1, 1.0);
						error = GRBupdatemodel(model);
						handle_gurobi_error(error, env);
						error = GRBsetdblattrelement(model, "ScenNObj", layer_var_start_idx[i]+index2, 1.0);
						handle_gurobi_error(error, env);
						error = GRBupdatemodel(model);
						handle_gurobi_error(error, env);
					}	
					// Compute z1 + z3
					error = GRBsetintparam(GRBgetenv(model), "ScenarioNumber", j*3+1);
					handle_gurobi_error(error, GRBgetenv(model));
					error = GRBsetdblattrelement(model, "ScenNObj", layer_var_start_idx[i]+index3, 1.0);
					error = GRBupdatemodel(model);
					handle_gurobi_error(error, env);
					error = GRBsetdblattrelement(model, "ScenNObj", layer_var_start_idx[i]+index1, 1.0);
					handle_gurobi_error(error, env);
					error = GRBupdatemodel(model);
					handle_gurobi_error(error, env);
					// Compute z2 + z3
					error = GRBsetintparam(GRBgetenv(model), "ScenarioNumber", j*3+2);
					handle_gurobi_error(error, GRBgetenv(model));
					error = GRBsetdblattrelement(model, "ScenNObj", layer_var_start_idx[i]+index3, 1.0);
					error = GRBupdatemodel(model);
					handle_gurobi_error(error, env);
					error = GRBsetdblattrelement(model, "ScenNObj", layer_var_start_idx[i]+index2, 1.0);
					handle_gurobi_error(error, env);
					error = GRBupdatemodel(model);
					handle_gurobi_error(error, env);
					// Compute z1 + z2 + z3
					error = GRBsetintparam(GRBgetenv(model), "ScenarioNumber", j*3+3);
					handle_gurobi_error(error, GRBgetenv(model));
					error = GRBsetdblattrelement(model, "ScenNObj", layer_var_start_idx[i]+index3, 1.0);
					error = GRBupdatemodel(model);
					handle_gurobi_error(error, env);
					error = GRBsetdblattrelement(model, "ScenNObj", layer_var_start_idx[i]+index2, 1.0);
					error = GRBupdatemodel(model);
					handle_gurobi_error(error, env);
					error = GRBsetdblattrelement(model, "ScenNObj", layer_var_start_idx[i]+index1, 1.0);
					handle_gurobi_error(error, env);
					error = GRBupdatemodel(model);
					handle_gurobi_error(error, env);
				}
				// Do the multi-scenario solving
				error = GRBsetintattr(model, "ModelSense", -1); //maximization
				handle_gurobi_error(error, env);
				error = GRBupdatemodel(model);
				handle_gurobi_error(error, env);
				error = GRBoptimize(model);
				handle_gurobi_error(error, env);
				error = GRBgetintattr(model, GRB_INT_ATTR_STATUS, &optimstatus);
				handle_gurobi_error(error, env);
				// printf("The solution status is %d\n", optimstatus);
				if(optimstatus == GRB_INFEASIBLE){
					GRBfreemodel(model);
					GRBfreeenv(env);
					clock_t func_end = clock();
					double func_spent = (double)(func_end - func_begin) / CLOCKS_PER_SEC;
					printf("Refine succesfully at %d-th iteration, where we add additional constraints, total time is %f\n", count+1, func_spent);
					return true;
				}
				// error = GRBwrite(model, "multi_sce.mps");
				// handle_gurobi_error(error, env);
				double solved_ub, res_last_round; 
				for(j=0; j < unstable_relu_count - 2; j++){
					int index1 = unstable_index[j];
					int index2 = unstable_index[j+1];
					int index3 = unstable_index[j+2];
					double max1; double val[2] = {1.0, 1.0};
					// compute z1 + z2
					if(j==0){
						error = GRBsetintparam(GRBgetenv(model), "ScenarioNumber", 0);
						handle_gurobi_error(error, GRBgetenv(model));
						error = GRBgetintattr(model, GRB_INT_ATTR_STATUS, &optimstatus);
						handle_gurobi_error(error, env);
						// printf("The ScenarioNumber is %d, the solution status is %d\n", 0, optimstatus);
						error = GRBgetdblattr(model, "ScenNObjVal", &solved_ub);
						handle_gurobi_error(error, env);
						max1 = return_max_among_four(0.0, solved_ub + lp_solving_error, input_neurons[index1]->ub, input_neurons[index2]->ub);
						int ind[2] = {layer_var_start_idx[i+1]+index1, layer_var_start_idx[i+1]+index2};
						error = GRBaddconstr(model, 2, ind, val, GRB_LESS_EQUAL, max1, NULL);
						handle_gurobi_error(error, env);
					}else{
						max1 = res_last_round;
					}
					// compute z1 + z3
					error = GRBsetintparam(GRBgetenv(model), "ScenarioNumber", 3*j+1);
					// printf("The ScenarioNumber is %d\n", 3*j+1);
					handle_gurobi_error(error, GRBgetenv(model));
					error = GRBgetintattr(model, GRB_INT_ATTR_STATUS, &optimstatus);
					handle_gurobi_error(error, env);
					// printf("The ScenarioNumber is %d, the solution status is %d\n", 3*j + 1, optimstatus);
					error = GRBgetdblattr(model, "ScenNObjVal", &solved_ub);
					handle_gurobi_error(error, env);
					double max2 = return_max_among_four(0.0, solved_ub + lp_solving_error, input_neurons[index1]->ub, input_neurons[index3]->ub);
					int ind2[2] = {layer_var_start_idx[i+1]+index1, layer_var_start_idx[i+1]+index3};
					error = GRBaddconstr(model, 2, ind2, val, GRB_LESS_EQUAL, max2, NULL);
					handle_gurobi_error(error, env);
					// compute z2 + z3
					error = GRBsetintparam(GRBgetenv(model), "ScenarioNumber", 3*j+2);
					handle_gurobi_error(error, GRBgetenv(model));
					// printf("The ScenarioNumber is %d\n", 3*j+2);
					error = GRBgetdblattr(model, "ScenNObjVal", &solved_ub);
					handle_gurobi_error(error, env);
					error = GRBgetintattr(model, GRB_INT_ATTR_STATUS, &optimstatus);
					handle_gurobi_error(error, env);
					// printf("The ScenarioNumber is %d, the solution status is %d\n", 3*j+2, optimstatus);
					double max3 = return_max_among_four(0.0, solved_ub + lp_solving_error, input_neurons[index2]->ub, input_neurons[index3]->ub);
					res_last_round = max3;
					int ind3[2] = {layer_var_start_idx[i+1]+index2, layer_var_start_idx[i+1]+index3};
					error = GRBaddconstr(model, 2, ind3, val, GRB_LESS_EQUAL, max3, NULL);
					handle_gurobi_error(error, env);
					// compute z1 + z2 + z3
					error = GRBsetintparam(GRBgetenv(model), "ScenarioNumber", 3*j+3);
					handle_gurobi_error(error, GRBgetenv(model));
					// printf("The ScenarioNumber is %d\n", 3*j+3);
					error = GRBgetdblattr(model, "ScenNObjVal", &solved_ub);
					handle_gurobi_error(error, env);
					error = GRBgetintattr(model, GRB_INT_ATTR_STATUS, &optimstatus);
					handle_gurobi_error(error, env);
					// printf("The ScenarioNumber is %d, the solution status is %d\n", 3*j+3, optimstatus);
					double max4 = return_max_among_four(max1, solved_ub + lp_solving_error, max2, max3);
					int ind4[3] = {layer_var_start_idx[i+1]+index1, layer_var_start_idx[i+1]+index2, layer_var_start_idx[i+1]+index3};
					double val4[3] = {1.0, 1.0, 1.0};
					error = GRBaddconstr(model, 3, ind4, val4, GRB_LESS_EQUAL, max4, NULL);
					handle_gurobi_error(error, env);	
					// error = GRBupdatemodel(model);
					// handle_gurobi_error(error, env);
				}
				// revert the model back to the single model
				error = GRBsetintattr(model, "NumScenarios", 0);
				handle_gurobi_error(error, env);
				error = GRBupdatemodel(model);
				handle_gurobi_error(error, env);
			}
		}
		
		// add constraints for previously spurious labels
		for(k=0; k < spurious_count; k++){
			int spu_label = spurious_list[k];
			// we have out[ground_truth_label] - out[spu_label] > 0, for practical concern, we expand to >=
			int var_start_idx = layer_var_start_idx[numlayers - 1];
			int ind[2] = {var_start_idx+ground_truth_label,var_start_idx+spu_label};
			double val[2] = {1.0, -1.0};
			error = GRBaddconstr(model, 2, ind, val, GRB_GREATER_EQUAL, 0.0, NULL);
			handle_gurobi_error(error, env);
		}

		// add constraints regarding the current potential adversarial labels we try to eliminate, out[ground_truth_label] - out[spu_label] <= 0
		int var_start_idx = layer_var_start_idx[numlayers - 1];
		int ind[2] = {var_start_idx+ground_truth_label,var_start_idx+poten_cex};
		double val[2] = {1.0, -1.0};
		error = GRBaddconstr(model, 2, ind, val, GRB_LESS_EQUAL, 0.0, NULL);
		// update model
		error = GRBupdatemodel(model);
		handle_gurobi_error(error, env);

		// Simply check the feasibility, without objective function, if infeasible, then successfully prove spurious, return True
		error = GRBoptimize(model);
		handle_gurobi_error(error, env);
		/* Capture solution information */
		error = GRBgetintattr(model, GRB_INT_ATTR_STATUS, &optimstatus);
		handle_gurobi_error(error, env);
		if(optimstatus == GRB_INFEASIBLE){
			GRBfreemodel(model);
			GRBfreeenv(env);
			clock_t func_end = clock();
			double func_spent = (double)(func_end - func_begin) / CLOCKS_PER_SEC;
			printf("Refine succesfully at %d-th iteration, total time is %f\n", count+1, func_spent);
			return true;
		}
		
		// If feasible, transfer this current model (all constraints and objective function to be 0) to be Multiple Scenarios
		// So that one call can handle all scenario solving
		error = GRBsetintattr(model, "NumScenarios", fp->num_pixels);
		handle_gurobi_error(error, env);
		error = GRBupdatemodel(model);
		handle_gurobi_error(error, env);
		for(i=0; i < fp->num_pixels; i++){
			error = GRBsetintparam(GRBgetenv(model), "ScenarioNumber", i);
			handle_gurobi_error(error, GRBgetenv(model));
			error = GRBsetdblattrelement(model, "ScenNObj", i, 1.0);
			handle_gurobi_error(error, env);
			error = GRBupdatemodel(model);
			handle_gurobi_error(error, env);
		}
		error = GRBsetintattr(model, "ModelSense", 1); //minimization
		handle_gurobi_error(error, env);
		error = GRBupdatemodel(model);
		handle_gurobi_error(error, env);
		error = GRBoptimize(model);
		handle_gurobi_error(error, env);
		double solved_lb, solved_ub;
		for(i=0; i < fp->num_pixels; i++){
			error = GRBsetintparam(GRBgetenv(model), "ScenarioNumber", i);
			handle_gurobi_error(error, GRBgetenv(model));
			error = GRBgetdblattr(model, "ScenNObjVal", &solved_lb);
			handle_gurobi_error(error, env);
			// printf("The start layer neuron %zu originally has lower bound %.4f, solved_lb is %.4f\n", i, -fp->input_inf[i], solved_lb);
			fp->input_inf[i] = -(solved_lb - lp_solving_error);
		}
		error = GRBsetintattr(model, "ModelSense", -1); //maximization
		handle_gurobi_error(error, env);
		error = GRBupdatemodel(model);
		handle_gurobi_error(error, env);
		error = GRBoptimize(model);
		handle_gurobi_error(error, env);
		for(i=0; i < fp->num_pixels; i++){
			error = GRBsetintparam(GRBgetenv(model), "ScenarioNumber", i);
			handle_gurobi_error(error, GRBgetenv(model));
			error = GRBgetdblattr(model, "ScenNObjVal", &solved_ub);
			handle_gurobi_error(error, env);
			fp->input_sup[i] = solved_ub + lp_solving_error;
		}
		
		// multiple scenarios of unstable relu nodes
		error = GRBsetintattr(model, "NumScenarios", 0);
		handle_gurobi_error(error, env);
		error = GRBupdatemodel(model);
		handle_gurobi_error(error, env);
		int counter = 0;
		int unstable_relu_count = 0;
		int relu_refine_count = 0; 
		// count unstable relu number to create scenarios
		for(i=0; i < numlayers; i++){
			layer_t * cur_layer = fp->layers[i];
			if(!cur_layer->is_activation && (i < numlayers-1) && fp->layers[i+1]->is_activation){
				layer_t * next_layer = fp->layers[i+1];
				neuron_t ** relu_neurons = next_layer->neurons;
				for(j=0; j < cur_layer->dims; j++){
					if(relu_neurons[j]->ub!=0.0 && relu_neurons[j]->lb>=0){
						unstable_relu_count ++;
					}
				}
			}
		}
		// printf("The number of unstable neuron is %d\n", unstable_relu_count);
		error = GRBsetintattr(model, "NumScenarios", unstable_relu_count);
		handle_gurobi_error(error, env);
		int * layer_info = (int *)malloc(unstable_relu_count*sizeof(int));
		int * index_info = (int *)malloc(unstable_relu_count*sizeof(int));
		for(i=0; i < numlayers; i++){
			layer_t * cur_layer = fp->layers[i];
			if(!cur_layer->is_activation && (i < numlayers-1) && fp->layers[i+1]->is_activation){
				layer_t * next_layer = fp->layers[i+1];
				neuron_t ** relu_neurons = next_layer->neurons;
				for(j=0; j < cur_layer->dims; j++){
					if(relu_neurons[j]->ub!=0.0 && relu_neurons[j]->lb>=0){
						error = GRBsetintparam(GRBgetenv(model), "ScenarioNumber", counter);
						handle_gurobi_error(error, GRBgetenv(model));
						error = GRBsetdblattrelement(model, "ScenNObj", layer_var_start_idx[i]+j, 1.0);
						handle_gurobi_error(error, env);
						// printf("The recored unstable relu info are %d, %zu, %zu\n", counter, i, j);
						layer_info[counter] = i;
						index_info[counter] = j;
						counter ++;
					}
				}
			}
		}
		error = GRBsetintattr(model, "ModelSense", 1); //minimization
		handle_gurobi_error(error, env);
		error = GRBupdatemodel(model);
		handle_gurobi_error(error, env);
		error = GRBoptimize(model);
		handle_gurobi_error(error, env);
		for(i=0; i < unstable_relu_count; i++){
			layer_t * cur_layer = fp->layers[layer_info[i]];
			error = GRBsetintparam(GRBgetenv(model), "ScenarioNumber", i);
			handle_gurobi_error(error, GRBgetenv(model));
			error = GRBgetdblattr(model, "ScenNObjVal", &solved_lb);
			handle_gurobi_error(error, env);
			cur_layer->neurons[index_info[i]]->lb = fmin(-(solved_lb - lp_solving_error), cur_layer->neurons[index_info[i]]->lb);
			if(cur_layer->neurons[index_info[i]]->lb<0){
				printf("The refreshed pos Relu is layer %zu, index %zu\n",layer_info[i], index_info[i]);
				relu_refine_count ++;
			}	
		}
		error = GRBsetintattr(model, "ModelSense", -1); //maximization
		handle_gurobi_error(error, env);
		error = GRBupdatemodel(model);
		handle_gurobi_error(error, env);
		error = GRBoptimize(model);
		handle_gurobi_error(error, env);
		for(i=0; i < unstable_relu_count; i++){
			layer_t * cur_layer = fp->layers[layer_info[i]];
			error = GRBsetintparam(GRBgetenv(model), "ScenarioNumber", i);
			handle_gurobi_error(error, GRBgetenv(model));
			error = GRBgetdblattr(model, "ScenNObjVal", &solved_ub);
			handle_gurobi_error(error, env);
			cur_layer->neurons[index_info[i]]->ub = fmin(solved_ub + lp_solving_error, cur_layer->neurons[index_info[i]]->ub);
			if(cur_layer->neurons[index_info[i]]->ub<=0){
				printf("The refreshed neg Relu is layer %zu, index %zu\n",layer_info[i], index_info[i]);
				relu_refine_count ++;
			}
		}
		
		printf("Refreshed ReLU nodes: %d\n", relu_refine_count);
		/* Free model */
  		GRBfreemodel(model);
  		/* Free environment */
  		GRBfreeenv(env);
	}
	if(optimstatus == GRB_OPTIMAL){
		printf("Need to do adversarial example finding or quantitative robustness\n");
	}
	clock_t func_end = clock();
	double func_spent = (double)(func_end - func_begin) / CLOCKS_PER_SEC;
	printf("fail refinement, # total iteration is %d,total time is %f\n", count, func_spent);
	return false;
}

void * descending_sort_dictionary(int * key_list, double * value_list, int num){
	int i, j, index_temp;
	double value_temp;
	for(i = 0; i < num; i++){
		for(j = i+1; j < num; j++){
			if(value_list[i] < value_list[j]){
				value_temp = value_list[i];
				value_list[i] = value_list[j];
				value_list[j] = value_temp;
				index_temp = key_list[i];
				key_list[i] = key_list[j];
				key_list[j] = index_temp;
			}
		}
	}
	// for(i = 0; i < num; i++){
	// 	printf("%.2f ", value_list[i]);
	// }
	// printf("\n");
	return NULL;
}

bool is_spurious_MILP_encoding(elina_manager_t* man, elina_abstract0_t* element, elina_dim_t ground_truth_label, elina_dim_t poten_cex, int * spurious_list, int spurious_count, int MAX_ITER){
	// SMUPoly paper with additional constraints regarding two neurons
	int count, k;
	clock_t func_begin = clock();
	size_t i, j, n;
	fppoly_t *fp = fppoly_of_abstract0(element);
    size_t numlayers = fp->numlayers;
	double ulp = ldexpl(1.0,-52);
	double lp_solving_error = pow(10.0, -6.0) + ulp;
	int optimstatus;
	printf("Adversarial label is %zu\n", poten_cex);
	for(i=0; i < fp->num_pixels; i++){
		// set the input neurons back to the original input space
		fp->input_inf[i] = fp->original_input_inf[i];
		fp->input_sup[i] = fp->original_input_sup[i];
	}
	clear_neurons_status(man, element);

	// Refine for MAX_ITER times
	for(count = 0; count < MAX_ITER; count++){
		run_deeppoly(man, element);
		printf("Refinement iteration %d\n", count+1);
		/* Create environment */
  		GRBenv *env   = NULL;
  		GRBmodel *model = NULL;
		int error = 0;	
		error = GRBemptyenv(&env);
		handle_gurobi_error(error, env);
		error = GRBsetintparam(env, "OutputFlag", 0);
		handle_gurobi_error(error, env);
		error = GRBstartenv(env);
		handle_gurobi_error(error, env);
		/* Create an empty model */
		error = GRBnewmodel(env, &model, "refinement_solver", 0, NULL, NULL, NULL, NULL, NULL);
		handle_gurobi_error(error, env);

		// The index starter for variables at different layers
		int layer_var_start_idx[numlayers];

		// fp->input_inf[i], fp->input_sup[i], add the input layer constraints
		layer_var_start_idx[0] = fp->num_pixels;
		for(i=0; i < fp->num_pixels; i++){
			error = GRBaddvar(model, 0, NULL, NULL, 0.0, -fp->input_inf[i], fp->input_sup[i], GRB_CONTINUOUS, NULL);
        	handle_gurobi_error(error, env);
		}
		
		// add constaints for each hidden and output layer
		for(i=0; i < numlayers; i++){
			layer_t * cur_layer = fp->layers[i];
			neuron_t ** cur_neurons = cur_layer->neurons;
			size_t num_cur_neurons = cur_layer->dims;
			if(i+1 < numlayers){
				// Set up the variable start index 
				layer_var_start_idx[i+1] = layer_var_start_idx[i] + num_cur_neurons;
			}
			int defined_var_start_idx;
			if(i==0){
				defined_var_start_idx = 0;
			}
			else{
				defined_var_start_idx = layer_var_start_idx[i-1];
			}

	
			if(cur_layer->is_activation){
				//current layer is ReLU layer, we add the constraints according to RELU behavior
				for(j=0; j < num_cur_neurons; j++){
					// add constraints for each ReLU node
					// need to handle non-stable (two lower constraints will be added) and stable constraint
					neuron_t * relu_node = cur_neurons[j];
					if(relu_node->ub == 0.0){
						// stable unactivated relu nodes
						expr_t * relu_expr = relu_node->lexpr;
						assert(relu_expr->type==SPARSE);
						error = GRBaddvar(model, 0, NULL, NULL, 0.0, -relu_node->lb, relu_node->ub, GRB_CONTINUOUS, NULL);
						handle_gurobi_error(error, env);
					}
					else if(relu_node->lb<0.0){
						// stable activated relu nodes
						expr_t * relu_expr = relu_node->lexpr;
						size_t num_pre_neurons = relu_expr->size;
						assert(relu_expr->type==SPARSE);
						assert(num_pre_neurons==1);
						error = GRBaddvar(model, 0, NULL, NULL, 0.0, -relu_node->lb, relu_node->ub, GRB_CONTINUOUS, NULL);
						handle_gurobi_error(error, env);
						int ind[2] = {layer_var_start_idx[i] + j, defined_var_start_idx + j};
						double val[2] = {-1.0 , relu_expr->sup_coeff[0]};
						error = GRBaddconstr(model, 2, ind, val, GRB_EQUAL, relu_expr->inf_cst, NULL);
						handle_gurobi_error(error, env);
					}
					else{
						// unstable relu nodes, add two lower constarints, and also handle FP error for upper constraint
						expr_t * relu_expr = relu_node->uexpr;
						size_t num_pre_neurons = relu_expr->size;
						assert(relu_expr->type==SPARSE);
						assert(num_pre_neurons==1);
						// The lower bound setting already indicate that relu >=0
						error = GRBaddvar(model, 0, NULL, NULL, 0.0, -relu_node->lb, relu_node->ub, GRB_CONTINUOUS, NULL);
						handle_gurobi_error(error, env);
						int ind[2] = {layer_var_start_idx[i] + j, defined_var_start_idx + j};
						double val[2] = {-1.0 , 1.0};
						// add lower bound, y >= x, -y+x <= 0
						error = GRBaddconstr(model, 2, ind, val, GRB_LESS_EQUAL, 0.0, NULL);
						handle_gurobi_error(error, env);
						int ind2[2] = {layer_var_start_idx[i] + j, defined_var_start_idx + j};
						double over_slope = relu_expr->sup_coeff[0]+ ulp;
						double val2[2] = {-1.0, over_slope};
						int pre = cur_layer->predecessors[0]-1;
						double in_lb = fp->layers[pre]->neurons[j]->lb;
						assert(in_lb>=0);
						double over_b = (fabs(in_lb)+ulp)*over_slope + ulp;
						// add upper bound, y <= ax+b, -y+ax >= -b
						error = GRBaddconstr(model, 2, ind2, val2, GRB_GREATER_EQUAL, -over_b, NULL);
						handle_gurobi_error(error, env);
					}
					// update model
					error = GRBupdatemodel(model);
					handle_gurobi_error(error, env);
				}
			}
			else{
				// current layer is affine layer
				for(j=0; j < num_cur_neurons; j++){
					neuron_t * affine_node = cur_neurons[j];
					expr_t * affine_expr = affine_node->lexpr;
					size_t num_pre_neurons = affine_expr->size;
					assert(affine_expr->type==DENSE);
					error = GRBaddvar(model, 0, NULL, NULL, 0.0, -affine_node->lb, affine_node->ub, GRB_CONTINUOUS, NULL);
					handle_gurobi_error(error, env);
					int ind[num_pre_neurons+1];
  					double val[num_pre_neurons+1];
					for(n=0; n < num_pre_neurons; n++){
						ind[n] = defined_var_start_idx + n;
						val[n] = affine_expr->sup_coeff[n];
					}
					ind[num_pre_neurons] = layer_var_start_idx[i] + j;
					val[num_pre_neurons] = -1.0;
					error = GRBaddconstr(model, num_pre_neurons+1, ind, val, GRB_EQUAL, affine_expr->inf_cst, NULL);
					handle_gurobi_error(error, env);
					// update model
					error = GRBupdatemodel(model);
					handle_gurobi_error(error, env);
				}
			}
		}
		
		// double lp_return;
		// error = GRBsetdblattrelement(model, "Obj", 9, 1.0);
		// handle_gurobi_error(error, env);
		// error = GRBsetintattr(model, "ModelSense", -1);
		// handle_gurobi_error(error, env);
		// error = GRBupdatemodel(model);
		// handle_gurobi_error(error, env);
		// error = GRBoptimize(model);
		// handle_gurobi_error(error, env);
		// error = GRBgetdblattr(model, GRB_DBL_ATTR_OBJVAL, &lp_return);
		// printf("Before adding MILP, the bound of y2 is %.6f\n", lp_return);
		// handle_gurobi_error(error, env);
		// error = GRBsetdblattrelement(model, "Obj", 9, 0.0);
		// handle_gurobi_error(error, env);
		// error = GRBupdatemodel(model);
		// handle_gurobi_error(error, env);

		// add additional MILP encoding for selected unstable relu nodes
		int binary_var_start_ind = layer_var_start_idx[numlayers - 1] + fp->layers[numlayers - 1]->dims;
		int bivar_count = 0;
		for(i=0; i < numlayers; i++){
			layer_t * cur_layer = fp->layers[i];
			if(!cur_layer->is_activation && (i < numlayers-1) && fp->layers[i+1]->is_activation){
				layer_t * next_layer = fp->layers[i+1];
				assert(cur_layer->dims == next_layer->dims);
				neuron_t ** relu_neurons = next_layer->neurons;
				neuron_t ** input_neurons = cur_layer->neurons;
				int unstable_count = 0;
				for(j=0; j < cur_layer->dims; j++){
					if(relu_neurons[j]->ub!=0.0 && relu_neurons[j]->lb>=0){
						unstable_count++;
					}
				}
				int index_record[unstable_count];
				double value_record[unstable_count];
				unstable_count = 0;
				for(j=0; j < cur_layer->dims; j++){
					if(relu_neurons[j]->ub!=0.0 && relu_neurons[j]->lb>=0){
						index_record[unstable_count] = j;
						value_record[unstable_count] = input_neurons[j]->lb * input_neurons[j]->ub;
						unstable_count++;
					}
				}
				descending_sort_dictionary(index_record, value_record, unstable_count);
				// Add MILP encode
				int MILP_neu_num = (int)unstable_count/10;
				for(k = 0; k < MILP_neu_num; k++){
					// testing for one relu only
					int neuron_index = index_record[k];
					int input_var_ind = layer_var_start_idx[i] + neuron_index;
					int output_var_ind = layer_var_start_idx[i+1] + neuron_index;
					// printf("The input index is %d, the relu index is %d\n", input_var_ind, output_var_ind);
					// Generate the new binary variable, take default variable range
					error = GRBaddvar(model, 0, NULL, NULL, 0.0, 0.0, GRB_INFINITY, GRB_BINARY, NULL);
					handle_gurobi_error(error, env);
					// add the new upper bound where y <= x + l * a -l, y - x - l * a <= -l
					int ind[3] = {output_var_ind, input_var_ind, binary_var_start_ind + bivar_count};
					double val[3] = {1.0, -1.0, -input_neurons[neuron_index]->lb};
					error = GRBaddconstr(model, 3, ind, val, GRB_LESS_EQUAL, -input_neurons[neuron_index]->lb, NULL);
					handle_gurobi_error(error, env);
					// add the new upper bound where y <= u * a, y - u * a <= 0
					int ind2[2] = {output_var_ind, binary_var_start_ind + bivar_count};
					double val2[2] = {1.0, -input_neurons[neuron_index]->ub};
					error = GRBaddconstr(model, 2, ind2, val2, GRB_LESS_EQUAL, 0.0, NULL);
					handle_gurobi_error(error, env);
					bivar_count ++;
					// update model
					error = GRBupdatemodel(model);
					handle_gurobi_error(error, env);
				}
			}
		}

		// error = GRBsetdblattrelement(model, "Obj", 9, 1.0);
		// handle_gurobi_error(error, env);
		// error = GRBsetintattr(model, "ModelSense", -1);
		// handle_gurobi_error(error, env);
		// error = GRBupdatemodel(model);
		// handle_gurobi_error(error, env);
		// error = GRBoptimize(model);
		// handle_gurobi_error(error, env);
		// error = GRBgetdblattr(model, GRB_DBL_ATTR_OBJVAL, &lp_return);
		// printf("After adding MILP, the bound of y2 is %.6f\n", lp_return);
		// handle_gurobi_error(error, env);
		// error = GRBsetdblattrelement(model, "Obj", 9, 0.0);
		// handle_gurobi_error(error, env);
		// error = GRBupdatemodel(model);
		// handle_gurobi_error(error, env);

		// add constraints for previously spurious labels
		for(k=0; k < spurious_count; k++){
			int spu_label = spurious_list[k];
			// we have out[ground_truth_label] - out[spu_label] > 0, for practical concern, we expand to >=
			int var_start_idx = layer_var_start_idx[numlayers - 1];
			int ind[2] = {var_start_idx+ground_truth_label,var_start_idx+spu_label};
			double val[2] = {1.0, -1.0};
			error = GRBaddconstr(model, 2, ind, val, GRB_GREATER_EQUAL, 0.0, NULL);
			handle_gurobi_error(error, env);
		}

		// add constraints regarding the current potential adversarial labels we try to eliminate, out[ground_truth_label] - out[spu_label] <= 0
		int var_start_idx = layer_var_start_idx[numlayers - 1];
		int ind[2] = {var_start_idx+ground_truth_label,var_start_idx+poten_cex};
		double val[2] = {1.0, -1.0};
		error = GRBaddconstr(model, 2, ind, val, GRB_LESS_EQUAL, 0.0, NULL);
		// update model
		error = GRBupdatemodel(model);
		handle_gurobi_error(error, env);

		// Simply check the feasibility, without objective function, if infeasible, then successfully prove spurious, return True
		error = GRBoptimize(model);
		handle_gurobi_error(error, env);
		/* Capture solution information */
		error = GRBgetintattr(model, GRB_INT_ATTR_STATUS, &optimstatus);
		handle_gurobi_error(error, env);
		if(optimstatus == GRB_INFEASIBLE){
			GRBfreemodel(model);
			GRBfreeenv(env);
			clock_t func_end = clock();
			double func_spent = (double)(func_end - func_begin) / CLOCKS_PER_SEC;
			printf("Refine succesfully at %d-th iteration, total time is %f\n", count+1, func_spent);
			return true;
		}

		// solve for interval of input neurons
		for(i=0; i < fp->num_pixels; i++){
			double solved_lb, solved_ub;
			error = GRBsetdblattrelement(model, "Obj", i, 1.0);
        	handle_gurobi_error(error, env);
			// ModelSense, default value 1 indicates minimization, and -1 meaning maximization
			// lower bound solving
			error = GRBsetintattr(model, "ModelSense", 1);
    		handle_gurobi_error(error, env);
			error = GRBupdatemodel(model);
			handle_gurobi_error(error, env);
			error = GRBoptimize(model);
			handle_gurobi_error(error, env);
			error = GRBgetdblattr(model, GRB_DBL_ATTR_OBJVAL, &solved_lb);
			handle_gurobi_error(error, env);
			// upper bound solving
			error = GRBsetintattr(model, "ModelSense", -1);
    		handle_gurobi_error(error, env);
			error = GRBupdatemodel(model);
			handle_gurobi_error(error, env);
			error = GRBoptimize(model);
			handle_gurobi_error(error, env);
			error = GRBgetdblattr(model, GRB_DBL_ATTR_OBJVAL, &solved_ub);
			handle_gurobi_error(error, env);
			// Update the corresponding input lower and upper bound for next deeppoly execution
			// if(((solved_lb - lp_solving_error) > -fp->input_inf[i]) || ((solved_ub + lp_solving_error) < fp->input_sup[i])){
			// 	printf("The start layer neuron %zu originally has interval [%.4f, %.4f]\n", i, -fp->input_inf[i], fp->input_sup[i]);
			// 	printf("The start layer neuron %zu was updates to [%.4f, %.4f]\n", i, (solved_lb - lp_solving_error), solved_ub + lp_solving_error);
			// }
			fp->input_inf[i] = -(solved_lb - lp_solving_error);
			fp->input_sup[i] = solved_ub + lp_solving_error;
			// revert obj coeff back to 0
			error = GRBsetdblattrelement(model, "Obj", i, 0.0);
			handle_gurobi_error(error, env);
		}

		// solve for relu interval of unstable relu nodes
		int relu_refine_count = 0;
		for(i=0; i < numlayers; i++){
			layer_t * cur_layer = fp->layers[i];
			if(!cur_layer->is_activation && (i < numlayers-1) && fp->layers[i+1]->is_activation){
				layer_t * next_layer = fp->layers[i+1];
				assert(cur_layer->dims == next_layer->dims);
				neuron_t ** relu_neurons = next_layer->neurons;
				for(j=0; j < cur_layer->dims; j++){
					if(relu_neurons[j]->ub!=0.0 && relu_neurons[j]->lb>=0){
						double solved_lb, solved_ub;
						error = GRBsetdblattrelement(model, "Obj", layer_var_start_idx[i]+j, 1.0);
						handle_gurobi_error(error, env);
						// ModelSense, default value 1 indicates minimization, and -1 meaning maximization
						// lower bound solving
						error = GRBsetintattr(model, "ModelSense", 1);
						handle_gurobi_error(error, env);
						error = GRBupdatemodel(model);
						handle_gurobi_error(error, env);
						error = GRBoptimize(model);
						handle_gurobi_error(error, env);
						error = GRBgetdblattr(model, GRB_DBL_ATTR_OBJVAL, &solved_lb);
						handle_gurobi_error(error, env);
						// update relu node lower bound
						double lp_solving_error = pow(10.0, -6.0) + ulp;
						cur_layer->neurons[j]->lb = fmin(-(solved_lb - lp_solving_error), cur_layer->neurons[j]->lb);
						if(cur_layer->neurons[j]->lb<0){
							relu_refine_count ++;
						}
						// upper bound solving
						error = GRBsetintattr(model, "ModelSense", -1);
						handle_gurobi_error(error, env);
						error = GRBupdatemodel(model);
						handle_gurobi_error(error, env);
						error = GRBoptimize(model);
						handle_gurobi_error(error, env);
						error = GRBgetdblattr(model, GRB_DBL_ATTR_OBJVAL, &solved_ub);
						handle_gurobi_error(error, env);
						// update relu node upper bound
						cur_layer->neurons[j]->ub = fmin(solved_ub + lp_solving_error, cur_layer->neurons[j]->ub);
						if(cur_layer->neurons[j]->ub<=0){
							relu_refine_count ++;
						}
						error = GRBsetdblattrelement(model, "Obj", layer_var_start_idx[i]+j, 0.0);
						handle_gurobi_error(error, env);
					}
				}
			}
		}
		printf("Refreshed ReLU nodes: %d\n",relu_refine_count);
		/* Free model */
  		GRBfreemodel(model);
  		/* Free environment */
  		GRBfreeenv(env);
	}
	if(optimstatus == GRB_OPTIMAL){
		printf("Need to do adversarial example finding or quantitative robustness\n");
	}
	clock_t func_end = clock();
	double func_spent = (double)(func_end - func_begin) / CLOCKS_PER_SEC;
	printf("fail refinement, # total iteration is %d,total time is %f\n", count, func_spent);
	return false;
}

bool is_spurious_sequential(elina_manager_t* man, elina_abstract0_t* element, elina_dim_t ground_truth_label, elina_dim_t poten_cex, bool layer_by_layer, bool is_blk_segmentation, int blk_size, bool is_sum_def_over_input, int * spurious_list, int spurious_count, int MAX_ITER){
	// firstly consider the default case, where like in SMU paper, to encode all the constraints within the network
	// if(is_blk_segmentation){
	// 	// return is_spurious_pair_two_neurons(man, element, ground_truth_label, poten_cex, spurious_list, spurious_count, MAX_ITER);
	// 	return is_spurious_MILP_encoding(man, element, ground_truth_label, poten_cex, spurious_list, spurious_count, MAX_ITER);
	// 	// return test_better_pair_neurons(man, element, ground_truth_label, poten_cex);
	// }
	int count, k;
	clock_t func_begin = clock();
	size_t i, j, n;
	fppoly_t *fp = fppoly_of_abstract0(element);
    size_t numlayers = fp->numlayers;
	double ulp = ldexpl(1.0,-52);
	int optimstatus;
	for(i=0; i < fp->num_pixels; i++){
		// set the input neurons back to the original input space
		fp->input_inf[i] = fp->original_input_inf[i];
		fp->input_sup[i] = fp->original_input_sup[i];
	}
	// For new CEX pruning, clear the previous analysis bounds
	clear_neurons_status(man, element);
	/* sanity checking code fragment
	run_deeppoly(man, element);
	for(i=0; i < fp->layers[numlayers-1]->dims; i++){
		// set the input neurons back to the original input space
		printf("lb and ub are %.4f, %.4f respectively\n", -fp->layers[numlayers-1]->neurons[i]->lb, fp->layers[numlayers-1]->neurons[i]->ub);
	}
	return false;
	*/

	// Refine for MAX_ITER times
	for(count = 0; count < MAX_ITER; count++){
		// run deeppoly() firstly to get all the constraints (for the relu ones in particular, since affine constraint doesn't change)
		run_deeppoly(man, element);
		// if(count == 1){
		// 	for(i=0; i < fp->num_pixels; i++){
		// 		printf("After second run of dp, input %zu has interval [%.4f, %.4f]\n", i, -fp->input_inf[i], fp->input_sup[i]);
		// 	}
		// 	for(i=0; i < numlayers; i++){
		// 		for(j=0; j < fp->layers[i]->dims; j++){
		// 			printf("After second run of dp, hidden neuron %zu has interval [%.4f, %.4f]\n", j, -fp->layers[i]->neurons[j]->lb, fp->layers[i]->neurons[j]->ub);
		// 		}
		// 	}
		// }
		printf("Refinement iteration %d\n", count+1);
		/* Create environment */
  		GRBenv *env   = NULL;
  		GRBmodel *model = NULL;
		int error = 0;	
		error = GRBemptyenv(&env);
		handle_gurobi_error(error, env);
		error = GRBsetintparam(env, "OutputFlag", 0);
		handle_gurobi_error(error, env);
		error = GRBstartenv(env);
		handle_gurobi_error(error, env);
		/* Create an empty model */
		error = GRBnewmodel(env, &model, "refinement_solver", 0, NULL, NULL, NULL, NULL, NULL);
		handle_gurobi_error(error, env);

		// The index starter for variables at different layers
		int layer_var_start_idx[numlayers];

		// fp->input_inf[i], fp->input_sup[i], add the input layer constraints
		layer_var_start_idx[0] = fp->num_pixels;
		for(i=0; i < fp->num_pixels; i++){
			error = GRBaddvar(model, 0, NULL, NULL, 0.0, -fp->input_inf[i], fp->input_sup[i], GRB_CONTINUOUS, NULL);
        	handle_gurobi_error(error, env);
		}
		
		// add constaints for each hidden and output layer
		for(i=0; i < numlayers; i++){
			layer_t * cur_layer = fp->layers[i];
			neuron_t ** cur_neurons = cur_layer->neurons;
			size_t num_cur_neurons = cur_layer->dims;
			if(i+1 < numlayers){
				// Set up the variable start index 
				layer_var_start_idx[i+1] = layer_var_start_idx[i] + num_cur_neurons;
			}
			int defined_var_start_idx;
			if(i==0){
				defined_var_start_idx = 0;
			}
			else{
				defined_var_start_idx = layer_var_start_idx[i-1];
			}

			if(cur_layer->is_activation){
				//current layer is ReLU layer, we add the constraints according to RELU behavior
				for(j=0; j < num_cur_neurons; j++){
					// add constraints for each ReLU node
					// need to handle non-stable (two lower constraints will be added) and stable constraint
					neuron_t * relu_node = cur_neurons[j];
					if(relu_node->ub == 0.0){
						// stable unactivated relu nodes
						expr_t * relu_expr = relu_node->lexpr;
						assert(relu_expr->type==SPARSE);
						error = GRBaddvar(model, 0, NULL, NULL, 0.0, -relu_node->lb, relu_node->ub, GRB_CONTINUOUS, NULL);
						handle_gurobi_error(error, env);
					}
					else if(relu_node->lb<0.0){
						// stable activated relu nodes
						expr_t * relu_expr = relu_node->lexpr;
						size_t num_pre_neurons = relu_expr->size;
						assert(relu_expr->type==SPARSE);
						assert(num_pre_neurons==1);
						error = GRBaddvar(model, 0, NULL, NULL, 0.0, -relu_node->lb, relu_node->ub, GRB_CONTINUOUS, NULL);
						handle_gurobi_error(error, env);
						int ind[2] = {layer_var_start_idx[i] + j, defined_var_start_idx + j};
						double val[2] = {-1.0 , relu_expr->sup_coeff[0]};
						error = GRBaddconstr(model, 2, ind, val, GRB_EQUAL, relu_expr->inf_cst, NULL);
						handle_gurobi_error(error, env);
					}
					else{
						// unstable relu nodes, add two lower constarints, and also handle FP error for upper constraint
						expr_t * relu_expr = relu_node->uexpr;
						size_t num_pre_neurons = relu_expr->size;
						assert(relu_expr->type==SPARSE);
						assert(num_pre_neurons==1);
						// The lower bound setting already indicate that relu >=0
						error = GRBaddvar(model, 0, NULL, NULL, 0.0, -relu_node->lb, relu_node->ub, GRB_CONTINUOUS, NULL);
						handle_gurobi_error(error, env);
						int ind[2] = {layer_var_start_idx[i] + j, defined_var_start_idx + j};
						double val[2] = {-1.0 , 1.0};
						// add lower bound, y >= x, -y+x <= 0
						error = GRBaddconstr(model, 2, ind, val, GRB_LESS_EQUAL, 0.0, NULL);
						handle_gurobi_error(error, env);
						int ind2[2] = {layer_var_start_idx[i] + j, defined_var_start_idx + j};
						double over_slope = relu_expr->sup_coeff[0]+ ulp;
						double val2[2] = {-1.0, over_slope};
						int pre = cur_layer->predecessors[0]-1;
						double in_lb = fp->layers[pre]->neurons[j]->lb;
						assert(in_lb>=0);
						double over_b = (fabs(in_lb)+ulp)*over_slope + ulp;
						// add upper bound, y <= ax+b, -y+ax >= -b
						error = GRBaddconstr(model, 2, ind2, val2, GRB_GREATER_EQUAL, -over_b, NULL);
						handle_gurobi_error(error, env);
					}
					// update model
					error = GRBupdatemodel(model);
					handle_gurobi_error(error, env);
				}
			}
			else{
				// current layer is affine layer
				for(j=0; j < num_cur_neurons; j++){
					neuron_t * affine_node = cur_neurons[j];
					expr_t * affine_expr = affine_node->lexpr;
					size_t num_pre_neurons = affine_expr->size;
					assert(affine_expr->type==DENSE);
					error = GRBaddvar(model, 0, NULL, NULL, 0.0, -affine_node->lb, affine_node->ub, GRB_CONTINUOUS, NULL);
					handle_gurobi_error(error, env);
					int ind[num_pre_neurons+1];
  					double val[num_pre_neurons+1];
					for(n=0; n < num_pre_neurons; n++){
						ind[n] = defined_var_start_idx + n;
						val[n] = affine_expr->sup_coeff[n];
					}
					ind[num_pre_neurons] = layer_var_start_idx[i] + j;
					val[num_pre_neurons] = -1.0;
					error = GRBaddconstr(model, num_pre_neurons+1, ind, val, GRB_EQUAL, affine_expr->inf_cst, NULL);
					handle_gurobi_error(error, env);
					// update model
					error = GRBupdatemodel(model);
					handle_gurobi_error(error, env);
				}
			}
		}

		// add constraints for previously spurious labels
		for(k=0; k < spurious_count; k++){
			int spu_label = spurious_list[k];
			// we have out[ground_truth_label] - out[spu_label] > 0, for practical concern, we expand to >=
			int var_start_idx = layer_var_start_idx[numlayers - 1];
			int ind[2] = {var_start_idx+ground_truth_label,var_start_idx+spu_label};
			double val[2] = {1.0, -1.0};
			error = GRBaddconstr(model, 2, ind, val, GRB_GREATER_EQUAL, 0.0, NULL);
			handle_gurobi_error(error, env);
		}

		// add constraints regarding the current potential adversarial labels we try to eliminate, out[ground_truth_label] - out[spu_label] <= 0
		int var_start_idx = layer_var_start_idx[numlayers - 1];
		int ind[2] = {var_start_idx+ground_truth_label,var_start_idx+poten_cex};
		double val[2] = {1.0, -1.0};
		error = GRBaddconstr(model, 2, ind, val, GRB_LESS_EQUAL, 0.0, NULL);
		// update model
		error = GRBupdatemodel(model);
		handle_gurobi_error(error, env);

		// Simply check the feasibility, without objective function, if infeasible, then successfully prove spurious, return True
		error = GRBoptimize(model);
		handle_gurobi_error(error, env);
		/* Capture solution information */
		error = GRBgetintattr(model, GRB_INT_ATTR_STATUS, &optimstatus);
		handle_gurobi_error(error, env);
		if(optimstatus == GRB_INFEASIBLE){
			GRBfreemodel(model);
			GRBfreeenv(env);
			clock_t func_end = clock();
			double func_spent = (double)(func_end - func_begin) / CLOCKS_PER_SEC;
			printf("Refine succesfully at %d-th iteration, total time is %f\n", count+1, func_spent);
			return true;
		}
		// Else, if detect feasibility, do the solving
		// solve for interval of input neurons
		for(i=0; i < fp->num_pixels; i++){
			double solved_lb, solved_ub;
			error = GRBsetdblattrelement(model, "Obj", i, 1.0);
        	handle_gurobi_error(error, env);
			// ModelSense, default value 1 indicates minimization, and -1 meaning maximization
			// lower bound solving
			error = GRBsetintattr(model, "ModelSense", 1);
    		handle_gurobi_error(error, env);
			error = GRBupdatemodel(model);
			handle_gurobi_error(error, env);
			error = GRBoptimize(model);
			handle_gurobi_error(error, env);
			error = GRBgetdblattr(model, GRB_DBL_ATTR_OBJVAL, &solved_lb);
			handle_gurobi_error(error, env);
			// printf("The start layer neuron %zu originally has lower bound %.4f, solved_lb is %.4f\n", i, -fp->input_inf[i], solved_lb);
			// upper bound solving
			error = GRBsetintattr(model, "ModelSense", -1);
    		handle_gurobi_error(error, env);
			error = GRBupdatemodel(model);
			handle_gurobi_error(error, env);
			error = GRBoptimize(model);
			handle_gurobi_error(error, env);
			error = GRBgetdblattr(model, GRB_DBL_ATTR_OBJVAL, &solved_ub);
			// printf("The start layer neuron %zu originally has upper bound %.4f, solved_lb is %.4f\n", i, fp->input_sup[i], solved_ub);
			handle_gurobi_error(error, env);
			// Update the corresponding input lower and upper bound for next deeppoly execution
			double lp_solving_error = pow(10.0, -6.0) + ulp;
			// if(((solved_lb - lp_solving_error) > -fp->input_inf[i]) || ((solved_ub + lp_solving_error) < fp->input_sup[i])){
			// 	printf("The start layer neuron %zu originally has interval [%.4f, %.4f]\n", i, -fp->input_inf[i], fp->input_sup[i]);
			// 	printf("The start layer neuron %zu was updates to [%.4f, %.4f]\n", i, (solved_lb - lp_solving_error), solved_ub + lp_solving_error);
			// }
			fp->input_inf[i] = -(solved_lb - lp_solving_error);
			fp->input_sup[i] = solved_ub + lp_solving_error;
			// revert obj coeff back to 0
			error = GRBsetdblattrelement(model, "Obj", i, 0.0);
			handle_gurobi_error(error, env);
		}

		// solve for relu interval of unstable relu nodes
		int relu_refine_count = 0;
		for(i=0; i < numlayers; i++){
			layer_t * cur_layer = fp->layers[i];
			if(!cur_layer->is_activation && (i < numlayers-1) && fp->layers[i+1]->is_activation){
				layer_t * next_layer = fp->layers[i+1];
				assert(cur_layer->dims == next_layer->dims);
				neuron_t ** relu_neurons = next_layer->neurons;
				for(j=0; j < cur_layer->dims; j++){
					if(relu_neurons[j]->ub!=0.0 && relu_neurons[j]->lb>=0){
						double solved_lb, solved_ub;
						error = GRBsetdblattrelement(model, "Obj", layer_var_start_idx[i]+j, 1.0);
						handle_gurobi_error(error, env);
						// ModelSense, default value 1 indicates minimization, and -1 meaning maximization
						// lower bound solving
						error = GRBsetintattr(model, "ModelSense", 1);
						handle_gurobi_error(error, env);
						error = GRBupdatemodel(model);
						handle_gurobi_error(error, env);
						error = GRBoptimize(model);
						handle_gurobi_error(error, env);
						error = GRBgetdblattr(model, GRB_DBL_ATTR_OBJVAL, &solved_lb);
						handle_gurobi_error(error, env);
						// update relu node lower bound
						double lp_solving_error = pow(10.0, -6.0) + ulp;
						cur_layer->neurons[j]->lb = fmin(-(solved_lb - lp_solving_error), cur_layer->neurons[j]->lb);
						if(cur_layer->neurons[j]->lb<0){
							printf("The refreshed pos Relu is layer %zu, index %zu\n", i, j);
							relu_refine_count ++;
						}
						// upper bound solving
						error = GRBsetintattr(model, "ModelSense", -1);
						handle_gurobi_error(error, env);
						error = GRBupdatemodel(model);
						handle_gurobi_error(error, env);
						error = GRBoptimize(model);
						handle_gurobi_error(error, env);
						error = GRBgetdblattr(model, GRB_DBL_ATTR_OBJVAL, &solved_ub);
						handle_gurobi_error(error, env);
						// update relu node upper bound
						cur_layer->neurons[j]->ub = fmin(solved_ub + lp_solving_error, cur_layer->neurons[j]->ub);
						if(cur_layer->neurons[j]->ub<=0){
							printf("The refreshed neg Relu is layer %zu, index %zu\n", i, j);
							relu_refine_count ++;
						}
						error = GRBsetdblattrelement(model, "Obj", layer_var_start_idx[i]+j, 0.0);
						handle_gurobi_error(error, env);
					}
				}
			}
		}
		printf("Refreshed ReLU nodes: %d\n",relu_refine_count);
		/* Free model */
  		GRBfreemodel(model);
  		/* Free environment */
  		GRBfreeenv(env);
	}
	if(optimstatus == GRB_OPTIMAL){
		printf("Need to do adversarial example finding or quantitative robustness\n");
	}
	clock_t func_end = clock();
	double func_spent = (double)(func_end - func_begin) / CLOCKS_PER_SEC;
	printf("fail refinement, # total iteration is %d,total time is %f\n", count+1, func_spent);
	return false;
}

bool is_spurious(elina_manager_t* man, elina_abstract0_t* element, elina_dim_t ground_truth_label, elina_dim_t poten_cex, bool layer_by_layer, bool is_blk_segmentation, int blk_size, bool is_sum_def_over_input, int * spurious_list, int spurious_count, int MAX_ITER){
	// firstly consider the default case, where like in SMU paper, to encode all the constraints within the network
	if(is_blk_segmentation){
		// return is_spurious_sequential(man, element, ground_truth_label, poten_cex, layer_by_layer, is_blk_segmentation, blk_size, is_sum_def_over_input, spurious_list, spurious_count, MAX_ITER);
		return is_spurious_pair_three_neurons(man, element, ground_truth_label, poten_cex, spurious_list, spurious_count, MAX_ITER);
		// cdd_numerical_inconsistent();
		// return false;
	}
	int count, k;
	clock_t func_begin = clock();
	size_t i, j, n;
	fppoly_t *fp = fppoly_of_abstract0(element);
    size_t numlayers = fp->numlayers;
	double ulp = ldexpl(1.0,-52);
	int optimstatus;
	for(i=0; i < fp->num_pixels; i++){
		// set the input neurons back to the original input space
		fp->input_inf[i] = fp->original_input_inf[i];
		fp->input_sup[i] = fp->original_input_sup[i];
	}
	clear_neurons_status(man, element);
	// Refine for MAX_ITER times
	for(count = 0; count < MAX_ITER; count++){
		run_deeppoly(man, element);
		printf("Refinement (parallel solving) iteration %d\n", count+1);
		/* Create environment */
  		GRBenv *env   = NULL;
  		GRBmodel *model = NULL;
		int error = 0;	
		error = GRBemptyenv(&env);
		handle_gurobi_error(error, env);
		error = GRBsetintparam(env, "OutputFlag", 0);
		handle_gurobi_error(error, env);
		error = GRBstartenv(env);
		handle_gurobi_error(error, env);
		/* Create an empty model */
		error = GRBnewmodel(env, &model, "refinement_solver", 0, NULL, NULL, NULL, NULL, NULL);
		handle_gurobi_error(error, env);
		// The index starter for variables at different layers
		int layer_var_start_idx[numlayers];
		// fp->input_inf[i], fp->input_sup[i], add the input layer constraints
		layer_var_start_idx[0] = fp->num_pixels;
		for(i=0; i < fp->num_pixels; i++){
			error = GRBaddvar(model, 0, NULL, NULL, 0.0, -fp->input_inf[i], fp->input_sup[i], GRB_CONTINUOUS, NULL);
        	handle_gurobi_error(error, env);
		}
		// add constaints for each hidden and output layer
		for(i=0; i < numlayers; i++){
			layer_t * cur_layer = fp->layers[i];
			neuron_t ** cur_neurons = cur_layer->neurons;
			size_t num_cur_neurons = cur_layer->dims;
			if(i+1 < numlayers){
				// Set up the variable start index 
				layer_var_start_idx[i+1] = layer_var_start_idx[i] + num_cur_neurons;
			}
			int defined_var_start_idx;
			if(i==0){
				defined_var_start_idx = 0;
			}
			else{
				defined_var_start_idx = layer_var_start_idx[i-1];
			}

			if(cur_layer->is_activation){
				//current layer is ReLU layer, we add the constraints according to RELU behavior
				for(j=0; j < num_cur_neurons; j++){
					// add constraints for each ReLU node
					// need to handle non-stable (two lower constraints will be added) and stable constraint
					neuron_t * relu_node = cur_neurons[j];
					if(relu_node->ub == 0.0){
						// stable unactivated relu nodes
						expr_t * relu_expr = relu_node->lexpr;
						assert(relu_expr->type==SPARSE);
						error = GRBaddvar(model, 0, NULL, NULL, 0.0, -relu_node->lb, relu_node->ub, GRB_CONTINUOUS, NULL);
						handle_gurobi_error(error, env);
					}
					else if(relu_node->lb<0.0){
						// stable activated relu nodes
						expr_t * relu_expr = relu_node->lexpr;
						size_t num_pre_neurons = relu_expr->size;
						assert(relu_expr->type==SPARSE);
						assert(num_pre_neurons==1);
						error = GRBaddvar(model, 0, NULL, NULL, 0.0, -relu_node->lb, relu_node->ub, GRB_CONTINUOUS, NULL);
						handle_gurobi_error(error, env);
						int ind[2] = {layer_var_start_idx[i] + j, defined_var_start_idx + j};
						double val[2] = {-1.0 , relu_expr->sup_coeff[0]};
						error = GRBaddconstr(model, 2, ind, val, GRB_EQUAL, relu_expr->inf_cst, NULL);
						handle_gurobi_error(error, env);
					}
					else{
						// unstable relu nodes, add two lower constarints, and also handle FP error for upper constraint
						expr_t * relu_expr = relu_node->uexpr;
						size_t num_pre_neurons = relu_expr->size;
						assert(relu_expr->type==SPARSE);
						assert(num_pre_neurons==1);
						// The lower bound setting already indicate that relu >=0
						error = GRBaddvar(model, 0, NULL, NULL, 0.0, -relu_node->lb, relu_node->ub, GRB_CONTINUOUS, NULL);
						handle_gurobi_error(error, env);
						int ind[2] = {layer_var_start_idx[i] + j, defined_var_start_idx + j};
						double val[2] = {-1.0 , 1.0};
						// add lower bound, y >= x, -y+x <= 0
						error = GRBaddconstr(model, 2, ind, val, GRB_LESS_EQUAL, 0.0, NULL);
						handle_gurobi_error(error, env);
						int ind2[2] = {layer_var_start_idx[i] + j, defined_var_start_idx + j};
						double over_slope = relu_expr->sup_coeff[0]+ ulp;
						double val2[2] = {-1.0, over_slope};
						int pre = cur_layer->predecessors[0]-1;
						double in_lb = fp->layers[pre]->neurons[j]->lb;
						assert(in_lb>=0);
						double over_b = (fabs(in_lb)+ulp)*over_slope + ulp;
						// add upper bound, y <= ax+b, -y+ax >= -b
						error = GRBaddconstr(model, 2, ind2, val2, GRB_GREATER_EQUAL, -over_b, NULL);
						handle_gurobi_error(error, env);
					}
					// update model
					error = GRBupdatemodel(model);
					handle_gurobi_error(error, env);
				}
			}
			else{
				// current layer is affine layer
				for(j=0; j < num_cur_neurons; j++){
					neuron_t * affine_node = cur_neurons[j];
					expr_t * affine_expr = affine_node->lexpr;
					size_t num_pre_neurons = affine_expr->size;
					assert(affine_expr->type==DENSE);
					error = GRBaddvar(model, 0, NULL, NULL, 0.0, -affine_node->lb, affine_node->ub, GRB_CONTINUOUS, NULL);
					handle_gurobi_error(error, env);
					int ind[num_pre_neurons+1];
  					double val[num_pre_neurons+1];
					for(n=0; n < num_pre_neurons; n++){
						ind[n] = defined_var_start_idx + n;
						val[n] = affine_expr->sup_coeff[n];
					}
					ind[num_pre_neurons] = layer_var_start_idx[i] + j;
					val[num_pre_neurons] = -1.0;
					error = GRBaddconstr(model, num_pre_neurons+1, ind, val, GRB_EQUAL, affine_expr->inf_cst, NULL);
					handle_gurobi_error(error, env);
					// update model
					error = GRBupdatemodel(model);
					handle_gurobi_error(error, env);
				}
			}
		}

		// add constraints for previously spurious labels
		for(k=0; k < spurious_count; k++){
			int spu_label = spurious_list[k];
			// we have out[ground_truth_label] - out[spu_label] > 0, for practical concern, we expand to >=
			int var_start_idx = layer_var_start_idx[numlayers - 1];
			int ind[2] = {var_start_idx+ground_truth_label,var_start_idx+spu_label};
			double val[2] = {1.0, -1.0};
			error = GRBaddconstr(model, 2, ind, val, GRB_GREATER_EQUAL, 0.0, NULL);
			handle_gurobi_error(error, env);
		}

		// add constraints regarding the current potential adversarial labels we try to eliminate, out[ground_truth_label] - out[spu_label] <= 0
		int var_start_idx = layer_var_start_idx[numlayers - 1];
		int ind[2] = {var_start_idx+ground_truth_label,var_start_idx+poten_cex};
		double val[2] = {1.0, -1.0};
		error = GRBaddconstr(model, 2, ind, val, GRB_LESS_EQUAL, 0.0, NULL);
		// update model
		error = GRBupdatemodel(model);
		handle_gurobi_error(error, env);

		// Simply check the feasibility, without objective function, if infeasible, then successfully prove spurious, return True
		error = GRBoptimize(model);
		handle_gurobi_error(error, env);
		/* Capture solution information */
		error = GRBgetintattr(model, GRB_INT_ATTR_STATUS, &optimstatus);
		handle_gurobi_error(error, env);
		if(optimstatus == GRB_INFEASIBLE){
			GRBfreemodel(model);
			GRBfreeenv(env);
			clock_t func_end = clock();
			double func_spent = (double)(func_end - func_begin) / CLOCKS_PER_SEC;
			printf("Refine succesfully at %d-th iteration, total time is %f\n", count+1, func_spent);
			return true;
		}
		
		// If feasible, transfer this current model (all constraints and objective function to be 0) to be Multiple Scenarios
		// So that one call can handle all scenario solving
		error = GRBsetintattr(model, "NumScenarios", fp->num_pixels);
		handle_gurobi_error(error, env);
		error = GRBupdatemodel(model);
		handle_gurobi_error(error, env);
		for(i=0; i < fp->num_pixels; i++){
			error = GRBsetintparam(GRBgetenv(model), "ScenarioNumber", i);
			handle_gurobi_error(error, GRBgetenv(model));
			error = GRBsetdblattrelement(model, "ScenNObj", i, 1.0);
			handle_gurobi_error(error, env);
			error = GRBupdatemodel(model);
			handle_gurobi_error(error, env);
		}
		error = GRBsetintattr(model, "ModelSense", 1); //minimization
		handle_gurobi_error(error, env);
		error = GRBupdatemodel(model);
		handle_gurobi_error(error, env);
		error = GRBoptimize(model);
		handle_gurobi_error(error, env);
		double solved_lb, solved_ub;
		double lp_solving_error = pow(10.0, -6.0) + ulp;
		for(i=0; i < fp->num_pixels; i++){
			error = GRBsetintparam(GRBgetenv(model), "ScenarioNumber", i);
			handle_gurobi_error(error, GRBgetenv(model));
			error = GRBgetdblattr(model, "ScenNObjVal", &solved_lb);
			handle_gurobi_error(error, env);
			// printf("The start layer neuron %zu originally has lower bound %.4f, solved_lb is %.4f\n", i, -fp->input_inf[i], solved_lb);
			fp->input_inf[i] = -(solved_lb - lp_solving_error);
		}
		error = GRBsetintattr(model, "ModelSense", -1); //maximization
		handle_gurobi_error(error, env);
		error = GRBupdatemodel(model);
		handle_gurobi_error(error, env);
		error = GRBoptimize(model);
		handle_gurobi_error(error, env);
		for(i=0; i < fp->num_pixels; i++){
			error = GRBsetintparam(GRBgetenv(model), "ScenarioNumber", i);
			handle_gurobi_error(error, GRBgetenv(model));
			error = GRBgetdblattr(model, "ScenNObjVal", &solved_ub);
			handle_gurobi_error(error, env);
			fp->input_sup[i] = solved_ub + lp_solving_error;
		}
		
		// multiple scenarios of unstable relu nodes
		error = GRBsetintattr(model, "NumScenarios", 0);
		handle_gurobi_error(error, env);
		error = GRBupdatemodel(model);
		handle_gurobi_error(error, env);
		int counter = 0;
		int unstable_relu_count = 0;
		int relu_refine_count = 0; 
		// count unstable relu number to create scenarios
		for(i=0; i < numlayers; i++){
			layer_t * cur_layer = fp->layers[i];
			if(!cur_layer->is_activation && (i < numlayers-1) && fp->layers[i+1]->is_activation){
				layer_t * next_layer = fp->layers[i+1];
				neuron_t ** relu_neurons = next_layer->neurons;
				for(j=0; j < cur_layer->dims; j++){
					if(relu_neurons[j]->ub!=0.0 && relu_neurons[j]->lb>=0){
						unstable_relu_count ++;
					}
				}
			}
		}
		// printf("The number of unstable neuron is %d\n", unstable_relu_count);
		error = GRBsetintattr(model, "NumScenarios", unstable_relu_count);
		handle_gurobi_error(error, env);
		int * layer_info = (int *)malloc(unstable_relu_count*sizeof(int));
		int * index_info = (int *)malloc(unstable_relu_count*sizeof(int));
		for(i=0; i < numlayers; i++){
			layer_t * cur_layer = fp->layers[i];
			if(!cur_layer->is_activation && (i < numlayers-1) && fp->layers[i+1]->is_activation){
				layer_t * next_layer = fp->layers[i+1];
				neuron_t ** relu_neurons = next_layer->neurons;
				for(j=0; j < cur_layer->dims; j++){
					if(relu_neurons[j]->ub!=0.0 && relu_neurons[j]->lb>=0){
						error = GRBsetintparam(GRBgetenv(model), "ScenarioNumber", counter);
						handle_gurobi_error(error, GRBgetenv(model));
						error = GRBsetdblattrelement(model, "ScenNObj", layer_var_start_idx[i]+j, 1.0);
						handle_gurobi_error(error, env);
						// printf("The recored unstable relu info are %d, %zu, %zu\n", counter, i, j);
						layer_info[counter] = i;
						index_info[counter] = j;
						counter ++;
					}
				}
			}
		}
		error = GRBsetintattr(model, "ModelSense", 1); //minimization
		handle_gurobi_error(error, env);
		error = GRBupdatemodel(model);
		handle_gurobi_error(error, env);
		error = GRBoptimize(model);
		handle_gurobi_error(error, env);
		for(i=0; i < unstable_relu_count; i++){
			layer_t * cur_layer = fp->layers[layer_info[i]];
			error = GRBsetintparam(GRBgetenv(model), "ScenarioNumber", i);
			handle_gurobi_error(error, GRBgetenv(model));
			error = GRBgetdblattr(model, "ScenNObjVal", &solved_lb);
			handle_gurobi_error(error, env);
			cur_layer->neurons[index_info[i]]->lb = fmin(-(solved_lb - lp_solving_error), cur_layer->neurons[index_info[i]]->lb);
			if(cur_layer->neurons[index_info[i]]->lb<0){
				printf("The refreshed pos Relu is layer %zu, index %zu\n",layer_info[i], index_info[i]);
				relu_refine_count ++;
			}	
		}
		error = GRBsetintattr(model, "ModelSense", -1); //maximization
		handle_gurobi_error(error, env);
		error = GRBupdatemodel(model);
		handle_gurobi_error(error, env);
		error = GRBoptimize(model);
		handle_gurobi_error(error, env);
		for(i=0; i < unstable_relu_count; i++){
			layer_t * cur_layer = fp->layers[layer_info[i]];
			error = GRBsetintparam(GRBgetenv(model), "ScenarioNumber", i);
			handle_gurobi_error(error, GRBgetenv(model));
			error = GRBgetdblattr(model, "ScenNObjVal", &solved_ub);
			handle_gurobi_error(error, env);
			cur_layer->neurons[index_info[i]]->ub = fmin(solved_ub + lp_solving_error, cur_layer->neurons[index_info[i]]->ub);
			if(cur_layer->neurons[index_info[i]]->ub<=0){
				printf("The refreshed neg Relu is layer %zu, index %zu\n",layer_info[i], index_info[i]);
				relu_refine_count ++;
			}
		}
		
		printf("Refreshed ReLU nodes: %d\n",relu_refine_count);
		free(layer_info);
		free(index_info);
		/* Free model */
  		GRBfreemodel(model);
  		/* Free environment */
  		GRBfreeenv(env);
	}
	if(optimstatus == GRB_OPTIMAL){
		printf("Need to do adversarial example finding or quantitative robustness\n");
	}
	clock_t func_end = clock();
	double func_spent = (double)(func_end - func_begin) / CLOCKS_PER_SEC;
	printf("fail refinement, # total iteration is %d,total time is %f\n", count+1, func_spent);
	return false;
}

double label_deviation_lb(elina_manager_t* man, elina_abstract0_t* element, elina_dim_t y, elina_dim_t x, bool layer_by_layer, bool is_residual, bool is_blk_segmentation, int blk_size, bool is_early_terminate, int early_termi_thre, bool is_sum_def_over_input, bool is_refinement){
	fppoly_t *fp = fppoly_of_abstract0(element);
	fppoly_internal_t * pr = fppoly_init_from_manager(man,ELINA_FUNID_ASSIGN_LINEXPR_ARRAY);
	expr_t * sub = (expr_t *)malloc(sizeof(expr_t));
	//sub->size = size;
	sub->inf_cst = 0;
	sub->sup_cst = 0;
	sub->inf_coeff = (double*)malloc(2*sizeof(double));
	sub->sup_coeff = (double*)malloc(2*sizeof(double));
	sub->dim =(size_t *)malloc(2*sizeof(size_t));
	sub->size = 2;
	sub->type = SPARSE;
	sub->inf_coeff[0] = -1;
	sub->sup_coeff[0] = 1;
	sub->dim[0] = y;
	sub->inf_coeff[1] = 1;
	sub->sup_coeff[1] = -1;
	sub->dim[1] = x;
	double lb = INFINITY;
	int k;
	if(is_blk_segmentation && is_refinement){
		// only apply modular for refinement process, not for original deeppoly execution
		is_blk_segmentation = false;
	}
	expr_t * backsubstituted_lexpr = copy_expr(sub);
	// printf("The auxilinary neuron is %zu - %zu\n", y, x);
	if(layer_by_layer){
		k = fp->numlayers - 1;
		while (k >= -1)
		{
			double cur_lb = get_lb_using_prev_layer(man, fp, &backsubstituted_lexpr, k, is_blk_segmentation);
			lb = fmin(lb, cur_lb);
			if (k < 0 || lb < 0)
				break;
			if(is_blk_segmentation && fp->layers[k]->is_end_layer_of_blk){
				k = fp->layers[k]->start_idx_in_same_blk;
			}
			else{
				k = fp->layers[k]->predecessors[0] - 1;
			}
		}
	}
	else{
		lb = get_lb_using_previous_layers(man, fp, &sub, fp->numlayers, layer_by_layer, is_residual, is_blk_segmentation, blk_size, is_early_terminate, early_termi_thre, is_sum_def_over_input, is_refinement);
	}
	if(sub){
		free_expr(sub);
		sub = NULL;
	}
	if(backsubstituted_lexpr){
		free_expr(backsubstituted_lexpr);
		backsubstituted_lexpr = NULL;
	}
	return -lb;
}

long int max(long int a, long int b){
	return a> b? a : b;
}

void handle_convolutional_layer(elina_manager_t* man, elina_abstract0_t* element, double *filter_weights, double * filter_bias, size_t * input_size, size_t *filter_size, size_t num_filters, size_t *strides, size_t *output_size, size_t pad_top, size_t pad_left, bool has_bias, size_t *predecessors, size_t num_predecessors, bool layer_by_layer, bool is_residual, bool is_blk_segmentation, int blk_size, bool is_early_terminate, int early_termi_thre, bool is_sum_def_over_input, bool is_refinement){
	assert(num_predecessors==1);
	fppoly_t *fp = fppoly_of_abstract0(element);
	size_t numlayers = fp->numlayers;
	size_t i, j;
	size_t num_pixels = input_size[0]*input_size[1]*input_size[2];
	
	output_size[2] = num_filters;
	size_t num_out_neurons = output_size[0]*output_size[1]*output_size[2];
	fppoly_add_new_layer(fp,num_out_neurons, predecessors, num_predecessors, false);
	neuron_t ** out_neurons = fp->layers[numlayers]->neurons;
	size_t out_x, out_y, out_z;
    size_t inp_x, inp_y, inp_z;
	size_t x_shift, y_shift;

	for(out_x=0; out_x < output_size[0]; out_x++) {
	    for(out_y = 0; out_y < output_size[1]; out_y++) {
		 for(out_z=0; out_z < output_size[2]; out_z++) {
		 	 //The one-dimensional index of the output pixel [out_x,out_y, out_z]
		     size_t mat_x = out_x*output_size[1]*output_size[2] + out_y*output_size[2] + out_z;
		     // filter_size[0]*filter_size[1] is the size of filter, should be 3*3
		     // input_size[2] is the channel size of the current input image, 3*3 parameters per channel
		     size_t num_coeff = input_size[2]*filter_size[0]*filter_size[1];
		     // input_size[2]*filter_size[0]*filter_size[1] is the parameters needed to compute this one node in the output
		     size_t actual_coeff = 0;
		     double *coeff = (double *)malloc(num_coeff*sizeof(double));
		     //double *coeff willl store the true paras 
		     size_t *dim = (size_t *)malloc(num_coeff*sizeof(double));
		      //double *coeff willl store the true input image pixel values
		     i=0;
		     for(inp_z=0; inp_z <input_size[2]; inp_z++) {
			 for(x_shift = 0; x_shift < filter_size[0]; x_shift++) {
			     for(y_shift =0; y_shift < filter_size[1]; y_shift++) {
				     long int x_val = out_x*strides[0]+x_shift-pad_top;	
			  	     long int y_val = out_y*strides[1]+y_shift-pad_left;
			  	     //The [x_val, y_val] position in the input
			  	     if(y_val<0 || y_val >= (long int)input_size[1]){
			     			continue;
			  	     }
				     
			  	     if(x_val<0 || x_val >= (long int)input_size[0]){
			     			continue;
			  	     }
				     size_t mat_y = x_val*input_size[1]*input_size[2] + y_val*input_size[2] + inp_z;
				     if(mat_y>=num_pixels){		 
			     			continue;
		          	     }
				     size_t filter_index = x_shift*filter_size[1]*input_size[2]*output_size[2] + y_shift*input_size[2]*output_size[2] + inp_z*output_size[2] + out_z;
				     coeff[i] = filter_weights[filter_index];
				     dim[i] = mat_y;
				     actual_coeff++;
				     i++;
			     }
			   }
		    }
		   double cst = has_bias? filter_bias[out_z] : 0;
		   //coeff is the array containing the paras for this output neuron; dim is the corresponding input dimension to compute with the neuron
		   //For this neuron, cst is the bias to the neuron
		   //actual_coeff is the number of parameters
			out_neurons[mat_x]->lexpr = create_sparse_expr(coeff,cst,dim,actual_coeff);
		   sort_sparse_expr(out_neurons[mat_x]->lexpr); 
		   out_neurons[mat_x]->uexpr = out_neurons[mat_x]->lexpr;
		   free(coeff);
		   free(dim);
	        }
	    }
	}
		
	update_state_using_previous_layers_parallel(man,fp,numlayers, layer_by_layer, is_residual, is_blk_segmentation, blk_size, is_early_terminate, early_termi_thre, is_sum_def_over_input, is_refinement);
	//fppoly_fprint(stdout,man,fp,NULL);
	//fflush(stdout);
	return;
}

void handle_residual_layer(elina_manager_t *man, elina_abstract0_t *element, size_t num_neurons, size_t *predecessors, size_t num_predecessors, bool layer_by_layer, bool is_residual, bool is_blk_segmentation, int blk_size, bool is_early_terminate, int early_termi_thre, bool is_sum_def_over_input, bool is_refinement){
	assert(num_predecessors==2);
	fppoly_t * fp = fppoly_of_abstract0(element);
	size_t numlayers = fp->numlayers;
	fppoly_add_new_layer(fp,num_neurons, predecessors, num_predecessors, false);
	size_t i;
	neuron_t **neurons = fp->layers[numlayers]->neurons;
	//printf("START\n");
	//fflush(stdout);
	for(i=0; i < num_neurons; i++){
		double *coeff = (double*)malloc(sizeof(double));
		coeff[0] = 1;
		size_t *dim = (size_t*)malloc(sizeof(size_t));
		dim[0] = i;
		neurons[i]->lexpr = create_sparse_expr(coeff,0,dim,1);
		neurons[i]->uexpr = neurons[i]->lexpr;
	}
	//printf("FINISH\n");
	//fflush(stdout);
	update_state_using_previous_layers_parallel(man,fp,numlayers, layer_by_layer, is_residual, is_blk_segmentation, blk_size, is_early_terminate, early_termi_thre, is_sum_def_over_input, is_refinement);
}

void free_neuron(neuron_t *neuron){
	if(neuron->uexpr && neuron->uexpr!=neuron->lexpr){
		free_expr(neuron->uexpr);
		neuron->uexpr = NULL;
	}
	if(neuron->lexpr){
		free_expr(neuron->lexpr);
		neuron->lexpr = NULL;
	}
	if(neuron->summary_lexpr!=NULL){
		free_expr(neuron->summary_lexpr);
		neuron->summary_lexpr = NULL;
	}
	if(neuron->summary_uexpr!=NULL){
		free_expr(neuron->summary_uexpr);
		neuron->summary_uexpr = NULL;
	}
	if(neuron->backsubstituted_lexpr!=NULL){
		free_expr(neuron->backsubstituted_lexpr);
		neuron->backsubstituted_lexpr = NULL;
	}
	if(neuron->backsubstituted_uexpr!=NULL){
		free_expr(neuron->backsubstituted_uexpr);
		neuron->backsubstituted_uexpr = NULL;
	}
	free(neuron);
}

void free_non_lstm_layer_expr(elina_manager_t *man, elina_abstract0_t *abs, size_t layerno){
    fppoly_t *fp = fppoly_of_abstract0(abs);
    if(layerno >= fp->numlayers){
        fprintf(stdout,"the layer does not exist\n");
        return;
    }
    layer_t * layer = fp->layers[layerno];
    size_t dims = layer->dims;
    size_t i;
    for(i=0; i < dims; i++){
        neuron_t *neuron = layer->neurons[i];
        if(neuron->uexpr!=neuron->lexpr){
            free_expr(neuron->uexpr);
        }
        if(neuron->lexpr){
            free_expr(neuron->lexpr);
        }
    }
}

void layer_free(layer_t * layer){
	size_t dims = layer->dims;
	size_t i;
	for(i=0; i < dims; i++){
		free_neuron(layer->neurons[i]);
	} 
	free(layer->neurons);
	layer->neurons = NULL;
	if(layer->h_t_inf!=NULL){
		free(layer->h_t_inf);
		layer->h_t_inf = NULL;
	}

	if(layer->h_t_sup!=NULL){
		free(layer->h_t_sup);
		layer->h_t_sup = NULL;
	}
	
	if(layer->c_t_inf!=NULL){
		free(layer->c_t_inf);
		layer->c_t_inf = NULL;
	}

	if(layer->c_t_sup!=NULL){
		free(layer->c_t_sup);
		layer->c_t_sup = NULL;
	}
	free(layer);
	layer = NULL;
}

void fppoly_free(elina_manager_t *man, fppoly_t *fp){
	size_t i;
	size_t output_size = fp->layers[fp->numlayers-1]->dims;
	for(i=0; i < fp->numlayers; i++){
		// printf("Free layer %zu in process, layer type:%d\n", i, fp->layers[i]->is_activation);
		layer_free(fp->layers[i]);
		// printf("Free layer %zu ends\n", i);
	}
	free(fp->layers);
	fp->layers = NULL;
	free(fp->input_inf);
	fp->input_inf = NULL;
        if(fp->input_lexpr!=NULL && fp->input_uexpr!=NULL){
		for(i=0; i < fp->num_pixels; i++){
			free(fp->input_lexpr[i]);
			free(fp->input_uexpr[i]);
		}
	
		free(fp->input_lexpr);
		fp->input_lexpr = NULL;
		free(fp->input_uexpr);
		fp->input_uexpr = NULL;
        }
	free(fp->input_sup);
	fp->input_sup = NULL;

    free(fp->spatial_indices);
    fp->spatial_indices = NULL;
    free(fp->spatial_neighbors);
    fp->spatial_neighbors = NULL;

	free(fp);
	fp = NULL;
}

void fppoly_fprint(FILE* stream, elina_manager_t* man, fppoly_t* fp, char** name_of_dim){
	size_t i;
	for(i = 0; i < fp->numlayers; i++){
		fprintf(stream,"layer: %zu\n", i);
		layer_fprint(stream, fp->layers[i], name_of_dim);
	}
	size_t output_size = fp->layers[fp->numlayers-1]->dims;
	size_t numlayers = fp->numlayers;
	neuron_t **neurons = fp->layers[numlayers-1]->neurons;
	fprintf(stream,"OUTPUT bounds: \n");
	for(i=0; i < output_size;i++){
		fprintf(stream,"%zu: [%g,%g] \n",i,-neurons[i]->lb,neurons[i]->ub);
	}

}

elina_interval_t * box_for_neuron(elina_manager_t* man, elina_abstract0_t * abs, size_t layerno, size_t neuron_no){
	fppoly_t *fp = fppoly_of_abstract0(abs);
	if(layerno >= fp->numlayers){
		fprintf(stdout,"the layer does not exist\n");
		return NULL;
	}
	layer_t * layer = fp->layers[layerno];
	size_t dims = layer->dims;
	if(neuron_no >= dims){
		fprintf(stdout,"the neuron does not exist\n");
		return NULL;
	}
	neuron_t * neuron = layer->neurons[neuron_no];
	elina_interval_t * res = elina_interval_alloc();
	elina_interval_set_double(res,-neuron->lb,neuron->ub);
	return res;
}

elina_interval_t ** box_for_layer(elina_manager_t* man, elina_abstract0_t * abs, size_t layerno){
	fppoly_t *fp = fppoly_of_abstract0(abs);
	if(layerno >= fp->numlayers){
		fprintf(stdout,"the layer does not exist\n");
		return NULL;
	}
	layer_t * layer = fp->layers[layerno];
	size_t dims = layer->dims;
	elina_interval_t ** itv_arr = (elina_interval_t **)malloc(dims*sizeof(elina_interval_t *));
	size_t i;
	for(i=0; i< dims; i++){
		itv_arr[i] = box_for_neuron(man, abs, layerno, i);
		
	}
	return itv_arr;
}

size_t get_num_neurons_in_layer(elina_manager_t* man, elina_abstract0_t * abs, size_t layerno){
	fppoly_t *fp = fppoly_of_abstract0(abs);
	if(layerno >= fp->numlayers){
		fprintf(stdout,"the layer does not exist\n");
		return 0;
	}
	layer_t * layer = fp->layers[layerno];
	size_t dims = layer->dims;
	
	return dims;
}

elina_linexpr0_t * get_expr_for_output_neuron(elina_manager_t *man, elina_abstract0_t *abs, size_t i, bool is_lower){
	fppoly_internal_t *pr = fppoly_init_from_manager(man, ELINA_FUNID_ASSIGN_LINEXPR_ARRAY);
	fppoly_t *fp = fppoly_of_abstract0(abs);
	
	size_t output_size = fp->layers[fp->numlayers-1]->dims;
	if(i >= output_size){
		return NULL;
	}
	size_t num_pixels = fp->num_pixels;
	expr_t * expr = NULL;
	if(is_lower){
		expr = fp->layers[fp->numlayers-1]->neurons[i]->lexpr;
	}
	else{
		expr = fp->layers[fp->numlayers-1]->neurons[i]->uexpr;
	}
	elina_linexpr0_t * res = NULL;
	size_t j,k;
	if((fp->input_lexpr!=NULL) && (fp->input_uexpr!=NULL)){
		if(is_lower){
			expr =  replace_input_poly_cons_in_lexpr(pr, expr, fp);
		}
		else{
			expr =  replace_input_poly_cons_in_uexpr(pr, expr, fp);
		}
	}
	size_t expr_size = expr->size;
	if(expr->type==SPARSE){
		sort_sparse_expr(expr);
		res = elina_linexpr0_alloc(ELINA_LINEXPR_SPARSE,expr_size);
	}
	else{
		res = elina_linexpr0_alloc(ELINA_LINEXPR_DENSE,expr_size);
	}
	elina_linexpr0_set_cst_interval_double(res,-expr->inf_cst,expr->sup_cst);
	
	for(j=0;j < expr_size; j++){
		if(expr->type==DENSE){
			k = j;
		}
		else{
			k = expr->dim[j];
		}
		elina_linexpr0_set_coeff_interval_double(res,k,-expr->inf_coeff[j],expr->sup_coeff[j]);
	}
	if((fp->input_lexpr!=NULL) && (fp->input_uexpr!=NULL)){
		free_expr(expr);
	}
	return res;
}

elina_linexpr0_t * get_lexpr_for_output_neuron(elina_manager_t *man,elina_abstract0_t *abs, size_t i){
	return get_expr_for_output_neuron(man,abs,i, true);
}

elina_linexpr0_t * get_uexpr_for_output_neuron(elina_manager_t *man,elina_abstract0_t *abs, size_t i){
	return get_expr_for_output_neuron(man,abs,i, false);
}

void update_bounds_for_neuron(elina_manager_t *man, elina_abstract0_t *abs, size_t layerno, size_t neuron_no, double lb, double ub){
	fppoly_t *fp = fppoly_of_abstract0(abs);
	if(layerno >= fp->numlayers){
		fprintf(stdout,"the layer does not exist\n");
		return;
	}
	layer_t * layer = fp->layers[layerno];
	neuron_t * neuron = layer->neurons[neuron_no];
	neuron->lb = -lb;
	neuron->ub = ub;
}

void update_activation_upper_bound_for_neuron(elina_manager_t *man, elina_abstract0_t *abs, size_t layerno, size_t neuron_no, double* coeff, size_t *dim, size_t size){
	//So far from what I see, this function is called for refinepoly domain
	fppoly_t *fp = fppoly_of_abstract0(abs);
	if(layerno >= fp->numlayers){
		fprintf(stdout,"the layer does not exist\n");
		return;
	}
	if(!fp->layers[layerno]->is_activation){
		fprintf(stdout, "the layer is not an activation layer\n");
		return;
	}
	layer_t * layer = fp->layers[layerno];
	neuron_t * neuron = layer->neurons[neuron_no];
	free_expr(neuron->uexpr);
	neuron->uexpr = NULL;
	neuron->uexpr = create_sparse_expr(coeff+1, coeff[0], dim, size);
	sort_sparse_expr(neuron->uexpr);
}

void update_activation_lower_bound_for_neuron(elina_manager_t *man, elina_abstract0_t *abs, size_t layerno, size_t neuron_no, double* coeff, size_t *dim, size_t size){
	//So far from what I see, this function is called for refinepoly domain
	fppoly_t *fp = fppoly_of_abstract0(abs);
	if(layerno >= fp->numlayers){
		fprintf(stdout,"the layer does not exist\n");
		return;
	}
	if(!fp->layers[layerno]->is_activation){
		fprintf(stdout, "the layer is not an activation layer\n");
		return;
	}
	layer_t * layer = fp->layers[layerno];
	neuron_t * neuron = layer->neurons[neuron_no];
	free_expr(neuron->lexpr);
	neuron->lexpr = NULL;
	neuron->lexpr = create_sparse_expr(coeff+1, coeff[0], dim, size);
	sort_sparse_expr(neuron->lexpr);
}
