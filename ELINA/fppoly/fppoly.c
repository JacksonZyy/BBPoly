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
#include "gurobi_c.h"


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
	res->input_lexpr = NULL;
	res->input_uexpr = NULL;
	size_t i;
	for(i=0; i < num_pixels; i++){
		res->input_inf[i] = -inf_array[i];
		res->input_sup[i] = sup_array[i];
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

void handle_fully_connected_layer_with_backsubstitute(elina_manager_t* man, elina_abstract0_t* element, double **weights, double * cst, size_t num_out_neurons, size_t num_in_neurons, size_t * predecessors, size_t num_predecessors, bool alloc, fnn_op OP, bool layer_by_layer, bool is_residual, bool is_blk_segmentation, int blk_size, bool is_early_terminate, int early_termi_thre, bool is_sum_def_over_input, bool var_cancel_heuristic){
    assert(num_predecessors==1);
    fppoly_t *fp = fppoly_of_abstract0(element);
    size_t numlayers = fp->numlayers;
    if(alloc){
        fppoly_add_new_layer(fp,num_out_neurons, predecessors, num_predecessors, false);
    }
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
		update_state_layer_by_layer_parallel(man,fp,numlayers,layer_by_layer, is_residual, is_blk_segmentation, blk_size, is_early_terminate, early_termi_thre, is_sum_def_over_input, var_cancel_heuristic);
	}
	else{
		update_state_using_previous_layers_parallel(man,fp,numlayers,layer_by_layer, is_residual, is_blk_segmentation, blk_size, is_early_terminate, early_termi_thre, is_sum_def_over_input, var_cancel_heuristic);
	}
    return;
}

void handle_sub_layer(elina_manager_t* man, elina_abstract0_t * abs,  double *cst, bool is_minuend, size_t size, size_t *predecessors, size_t num_predecessors, bool layer_by_layer, bool is_residual, bool is_blk_segmentation, int blk_size, bool is_early_terminate, int early_termi_thre, bool is_sum_def_over_input, bool var_cancel_heuristic){
	if(is_minuend==true){
		handle_fully_connected_layer_with_backsubstitute(man, abs, NULL, cst, size, size, predecessors, num_predecessors, true, SUB1, layer_by_layer,is_residual, is_blk_segmentation, blk_size, is_early_terminate, early_termi_thre, is_sum_def_over_input, var_cancel_heuristic);
	}
	else{
        	handle_fully_connected_layer_with_backsubstitute(man, abs, NULL, cst, size, size, predecessors, num_predecessors, true, SUB2, layer_by_layer, is_residual, is_blk_segmentation, blk_size, is_early_terminate, early_termi_thre, is_sum_def_over_input, var_cancel_heuristic);
	}
}

void handle_mul_layer(elina_manager_t* man, elina_abstract0_t * abs, double *bias,  size_t size, size_t *predecessors, size_t num_predecessors, bool layer_by_layer, bool is_residual, bool is_blk_segmentation, int blk_size, bool is_early_terminate, int early_termi_thre, bool is_sum_def_over_input, bool var_cancel_heuristic){
        handle_fully_connected_layer_with_backsubstitute(man, abs, NULL, bias, size, size, predecessors, num_predecessors, true, MUL, layer_by_layer, is_residual, is_blk_segmentation, blk_size, is_early_terminate, early_termi_thre, is_sum_def_over_input, var_cancel_heuristic);
}

void handle_fully_connected_layer_no_alloc(elina_manager_t* man, elina_abstract0_t * abs, double **weights, double *bias,   size_t size, size_t num_pixels, size_t *predecessors, size_t num_predecessors, bool layer_by_layer, bool is_residual, bool is_blk_segmentation, int blk_size, bool is_early_terminate, int early_termi_thre, bool is_sum_def_over_input, bool var_cancel_heuristic){
    handle_fully_connected_layer_with_backsubstitute(man, abs, weights, bias, size, num_pixels, predecessors, num_predecessors, false, MATMULT, layer_by_layer, is_residual, is_blk_segmentation, blk_size, is_early_terminate, early_termi_thre, is_sum_def_over_input, var_cancel_heuristic);
}


void handle_fully_connected_layer(elina_manager_t* man, elina_abstract0_t * abs, double **weights, double *bias,   size_t size, size_t num_pixels, size_t *predecessors, size_t num_predecessors, bool layer_by_layer, bool is_residual, bool is_blk_segmentation, int blk_size, bool is_early_terminate, int early_termi_thre, bool is_sum_def_over_input, bool var_cancel_heuristic){
     handle_fully_connected_layer_with_backsubstitute(man, abs, weights, bias, size, num_pixels, predecessors, num_predecessors, true, MATMULT, layer_by_layer, is_residual, is_blk_segmentation, blk_size, is_early_terminate, early_termi_thre, is_sum_def_over_input, var_cancel_heuristic);
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

bool is_greater(elina_manager_t* man, elina_abstract0_t* element, elina_dim_t y, elina_dim_t x, bool layer_by_layer, bool is_residual, bool is_blk_segmentation, int blk_size, bool is_early_terminate, int early_termi_thre, bool is_sum_def_over_input, bool var_cancel_heuristic){
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
	// printf("The auxilinary neuron is %zu - %zu\n", y, x);
	if(layer_by_layer){
		k = fp->numlayers - 1;
		while (k >= -1)
		{
			double cur_lb = get_lb_using_prev_layer(man, fp, &backsubstituted_lexpr, k);
			lb = fmin(lb, cur_lb);
			if (k < 0 || lb < 0)
				break;
			k = fp->layers[k]->predecessors[0] - 1;
		}
	}
	else{
		lb = get_lb_using_previous_layers(man, fp, &sub, fp->numlayers, layer_by_layer, is_residual, is_blk_segmentation, blk_size, is_early_terminate, early_termi_thre, is_sum_def_over_input, var_cancel_heuristic);
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

bool is_spurious(elina_manager_t* man, elina_abstract0_t* element, elina_dim_t ground_truth_label, elina_dim_t poten_cex, bool layer_by_layer, bool is_blk_segmentation, int blk_size, bool is_sum_def_over_input, int * spurious_list, int spurious_count, int MAX_ITER){
	// firstly consider the default case, where like in SMU paper, to encode all the constraints within the network
	return false;
}

double label_deviation_lb(elina_manager_t* man, elina_abstract0_t* element, elina_dim_t y, elina_dim_t x, bool layer_by_layer, bool is_residual, bool is_blk_segmentation, int blk_size, bool is_early_terminate, int early_termi_thre, bool is_sum_def_over_input, bool var_cancel_heuristic){
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
	// printf("The auxilinary neuron is %zu - %zu\n", y, x);
	if(layer_by_layer){
		k = fp->numlayers - 1;
		while (k >= -1)
		{
			double cur_lb = get_lb_using_prev_layer(man, fp, &backsubstituted_lexpr, k);
			lb = fmin(lb, cur_lb);
			if (k < 0 || lb < 0)
				break;
			k = fp->layers[k]->predecessors[0] - 1;
		}
	}
	else{
		lb = get_lb_using_previous_layers(man, fp, &sub, fp->numlayers, layer_by_layer, is_residual, is_blk_segmentation, blk_size, is_early_terminate, early_termi_thre, is_sum_def_over_input, var_cancel_heuristic);
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

void handle_convolutional_layer(elina_manager_t* man, elina_abstract0_t* element, double *filter_weights, double * filter_bias, size_t * input_size, size_t *filter_size, size_t num_filters, size_t *strides, size_t *output_size, size_t pad_top, size_t pad_left, bool has_bias, size_t *predecessors, size_t num_predecessors, bool layer_by_layer, bool is_residual, bool is_blk_segmentation, int blk_size, bool is_early_terminate, int early_termi_thre, bool is_sum_def_over_input, bool var_cancel_heuristic){
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
		
	update_state_using_previous_layers_parallel(man,fp,numlayers, layer_by_layer, is_residual, is_blk_segmentation, blk_size, is_early_terminate, early_termi_thre, is_sum_def_over_input, var_cancel_heuristic);
	//fppoly_fprint(stdout,man,fp,NULL);
	//fflush(stdout);
	return;
}

void handle_residual_layer(elina_manager_t *man, elina_abstract0_t *element, size_t num_neurons, size_t *predecessors, size_t num_predecessors, bool layer_by_layer, bool is_residual, bool is_blk_segmentation, int blk_size, bool is_early_terminate, int early_termi_thre, bool is_sum_def_over_input, bool var_cancel_heuristic){
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
	update_state_using_previous_layers_parallel(man,fp,numlayers, layer_by_layer, is_residual, is_blk_segmentation, blk_size, is_early_terminate, early_termi_thre, is_sum_def_over_input, var_cancel_heuristic);
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
