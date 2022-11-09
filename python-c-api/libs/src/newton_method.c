/**
 *  @defgroup PrivateNewtonMethod Private functions in newton_method.c
 *  @brief Private functions in newton_method.c
 *  @{
*/
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "wrapper_newton.h"
#define NEWTONLIB_RETURN_OK (0) //!< Exit status of private function is success
#define NEWTONLIB_RETURN_NG (1) //!< Exit status of private function is failed
#define NEWTONLIB_MEPS (1e-12)  //!< machine epsilon value

/**
 * @struct NewtonLib_JacobianParam_t
 * @brief structure of argument of calc_Jacobian_matrix function
*/
struct NewtonLib_JacobianParam_t {
    int32_t ndim;                //!< number of dimensions
    double *vec;                 //!< vector of x
    double delta;                //!< fixed difference value
    NEWTONLIB_CALLBACK callback; //!< callback function
};

/**
 * @fn static int32_t copy_array(int32_t ndim, const double *input, double *output)
 * @brief copy array
 * @param[in]  ndim   dimension
 * @param[in]  input  input array
 * @param[out] output output array
 * @return NEWTONLIB_RETURN_OK success
 *         NEWTONLIB_RETURN_NG failed
*/
static int32_t copy_array(int32_t ndim, const double *input, double *output);
/**
 * @fn static double norm(int32_t ndim, const double *vec)
 * @brief calculate norm
 * @param[in] ndim dimension
 * @param[in] vec  vector
 * @return norm
*/
static double norm(int32_t ndim, const double *vec);
/**
 * @fn static int32_t calc_Jacobian_matrix(const struct NewtonLib_JacobianParam_t *param, double *work, double *jacobian)
 * @brief calculate Jacobian matrix
 * @param[in]  param    function arguments
 * @param[in]  work     workspace
 * @param[out] jacobian Jacobian matrix
 * @return NEWTONLIB_RETURN_OK success
 *         NEWTONLIB_RETURN_NG failed
*/
static int32_t calc_Jacobian_matrix(const struct NewtonLib_JacobianParam_t *param, double *work, double *jacobian);
/**
 * @fn static int32_t pivot_selection(int32_t ndim, int32_t col, double *matrix)
 * @brief select pivot
 * @param[in]  ndim   dimension
 * @param[in]  col    target column
 * @param[in]  matrix coefficient matrix
 * @return pivot
*/
static int32_t pivot_selection(int32_t ndim, int32_t col, double *matrix);
/**
 * @fn static int32_t swap(int32_t ndim, int32_t row, int32_t pivot, double *matrix, double *vec)
 * @brief select pivot
 * @param[in]    ndim   dimension
 * @param[in]    row    target row
 * @param[in]    pivot  pivot
 * @param[inout] matrix coefficient matrix
 * @param[inout] vec    column vector
 * @return NEWTONLIB_RETURN_OK success
 *         NEWTONLIB_RETURN_NG failed
*/
static int32_t swap(int32_t ndim, int32_t row, int32_t pivot, double *matrix, double *vec);
/**
 * @fn static int32_t solve_sle(int32_t ndim, double *matrix, double *vec)
 * @brief solve simultaneous linear equations
 * @param[in]    ndim   dimension
 * @param[in]    matrix target matrix
 * @param[inout] vec    right vector and answer
 * @return NEWTONLIB_RETURN_OK success
 *         NEWTONLIB_RETURN_NG failed
*/
static int32_t solve_sle(int32_t ndim, double *matrix, double *vec);
/** @} */ // end of PrivateNewtonMethod

int32_t NewtonLib_newton_method(const struct NewtonLib_ArgParam_t *args, struct NewtonLib_Output_t *output) {
    int32_t ret = (int32_t)NEWTONLIB_FAILED;
    int32_t func_val;
    int32_t ndim, max_iter;
    int32_t iter, idx;
    double tol;
    double *hat, *diff_vec, *jacobian, *work;
    NEWTONLIB_CALLBACK callback;
    struct NewtonLib_JacobianParam_t param;

    //! initialize
    diff_vec = NULL;
    jacobian = NULL;
    work = NULL;
    memset(&param, 0, sizeof(struct NewtonLib_JacobianParam_t));

    //! validate arguments
    if ((NULL != args) && (NULL != output)) {
        //! setup args
        ndim = args->ndim;
        max_iter = args->max_iter;
        tol = args->tol;
        callback = args->callback;
        //! setup output
        hat = output->hat;
        output->is_convergent = (int32_t)NEWTONLIB_DIVERGENCE;

        //! copy initialized vector
        (void)copy_array(ndim, (const double *)(args->vec), hat);
        //! malloc
        diff_vec = (double *)malloc(sizeof(double) * ndim);
        jacobian = (double *)malloc(sizeof(double) * ndim * ndim);
        work = (double *)malloc(sizeof(double) * ndim * 3);
        if ((NULL == diff_vec) || (NULL == jacobian) || (NULL == work)) {
            goto EXIT_NEWTON_METHOD;
        }
        //! setup param
        param.ndim = ndim;
        param.vec = hat;
        param.delta = (fabs(args->delta) < (double)NEWTONLIB_MEPS) ? (double)NEWTONLIB_MEPS : args->delta;
        param.callback = callback;

        for (iter = 0; iter < max_iter; iter++) {
            //! convergence test
            if (norm(ndim, (const double *)hat) < tol) {
                output->is_convergent = (int32_t)NEWTONLIB_CONVERGENT;
                break;
            }
            //! Step1: calculate Jacobian matrix
            func_val = calc_Jacobian_matrix((const struct NewtonLib_JacobianParam_t *)&param, work, jacobian);
            if ((int32_t)NEWTONLIB_RETURN_OK != func_val) {
                ret = (int32_t)NEWTONLIB_JACOBIAN;
                goto EXIT_NEWTON_METHOD;
            }
            //! Step2: calculate function value
            func_val = callback(ndim, (const double *)hat, diff_vec);
            if ((int32_t)NEWTONLIB_SUCCESS != func_val) {
                ret = (int32_t)NEWTONLIB_OBJFUNCVAL;
                goto EXIT_NEWTON_METHOD;
            }
            for (idx = 0; idx < ndim; idx++) {
                diff_vec[idx] = -diff_vec[idx];
            }
            //! Step3: calculate diff_vec
            func_val = solve_sle(ndim, jacobian, diff_vec);
            if ((int32_t)NEWTONLIB_RETURN_OK != func_val) {
                ret = (int32_t)NEWTONLIB_SOLVE_SLE;
                goto EXIT_NEWTON_METHOD;
            }
            //! Step4: update vec
            for (idx = 0; idx < ndim; idx++) {
                hat[idx] += diff_vec[idx];
            }
        }
        ret = (int32_t)NEWTONLIB_SUCCESS;
    }

EXIT_NEWTON_METHOD:

    //! finalize
    if (NULL != diff_vec) {
        free(diff_vec);
        diff_vec = NULL;
    }
    if (NULL != jacobian) {
        free(jacobian);
        jacobian = NULL;
    }
    if (NULL != work) {
        free(work);
        work = NULL;
    }

    return ret;
}

/**
 *  @addtogroup PrivateNewtonMethod
 *  @{
*/
static int32_t copy_array(int32_t ndim, const double *input, double *output) {
    int32_t idx;

    for (idx = 0; idx < ndim; idx++) {
        output[idx] = input[idx];
    }

    return (int32_t)NEWTONLIB_RETURN_OK;
}

static double norm(int32_t ndim, const double *vec) {
    int32_t idx;
    double sum, out;
    sum = 0.0;

    for (idx = 0; idx < ndim; idx++) {
        sum += vec[idx] * vec[idx];
    }
    out = sqrt(sum);

    return out;
}

static int32_t calc_Jacobian_matrix(const struct NewtonLib_JacobianParam_t *param, double *work, double *jacobian) {
    int32_t ret = (int32_t)NEWTONLIB_RETURN_NG;
    int32_t row, col, idx, func_val;
    int32_t ndim;
    double delta, val;
    double *vec, *left, *right;
    NEWTONLIB_CALLBACK callback;
    ndim = param->ndim;
    delta = param->delta;
    callback = param->callback;
    vec = &work[0];
    right = &work[ndim];
    left = &work[ndim*2];
    (void)copy_array(ndim, (const double *)param->vec, &work[0]);

    for (col = 0; col < ndim; col++) {
        val = vec[col];
        //! calculate f(x + delta)
        vec[col] = val + delta;
        func_val = callback(ndim, (const double *)vec, right);
        if ((int32_t)NEWTONLIB_SUCCESS != func_val) {
            goto EXIT_CALC_JACOBIAN_MATRIX;
        }
        //! calculate f(x - delta)
        vec[col] = val - delta;
        func_val = callback(ndim, (const double *)vec, left);
        if ((int32_t)NEWTONLIB_SUCCESS != func_val) {
            goto EXIT_CALC_JACOBIAN_MATRIX;
        }
        for (row = 0; row < ndim; row++) {
            idx = row * ndim + col;
            jacobian[idx] = 0.5 * (right[row] - left[row]) / delta;
        }
        vec[col] = val;
    }
    ret = (int32_t)NEWTONLIB_RETURN_OK;

EXIT_CALC_JACOBIAN_MATRIX:

    return ret;
}

static int32_t pivot_selection(int32_t ndim, int32_t col, double *matrix) {
    int32_t idx;
    int32_t pos;
    double max_val, val;

    //! set diagonal element
    max_val = fabs(matrix[col * ndim + col]);
    pos = col;

    //! search maximum value in current column
    for (idx = col + 1; idx < ndim; idx++) {
        val = fabs(matrix[idx * ndim + col]);

        if (max_val < val) {
            max_val = val;
            pos = idx;
        }
    }

    return pos;
}

static int32_t swap(int32_t ndim, int32_t row, int32_t pivot, double *matrix, double *vec) {
    int32_t col;
    double tmp;

    //! swap current row in matrix
    for (col = 0; col < ndim; col++) {
        tmp = matrix[row * ndim + col];
        matrix[row * ndim + col] = matrix[pivot * ndim + col];
        matrix[pivot * ndim + col] = tmp;
    }
    //! swap column vector
    tmp = vec[row];
    vec[row] = vec[pivot];
    vec[pivot] = tmp;

    return (int32_t)NEWTONLIB_RETURN_OK;
}

static int32_t solve_sle(int32_t ndim, double *matrix, double *vec) {
    int32_t ret = (int32_t)NEWTONLIB_RETURN_NG;
    int32_t pivot, idx;
    int32_t row, col;
    double diag, scale, sum;

    //! forward elimination
    for (idx = 0; idx < ndim - 1; idx++) {
        //! select pivot
        pivot = pivot_selection(ndim, idx, matrix);
        //! swap if maximum value is not diagonal element
        if (idx != pivot) {
            swap(ndim, idx, pivot, matrix, vec);
        }
        //! get diagonal element
        diag = matrix[idx * ndim + idx];
        //! check value of diagonal element
        if (fabs(diag) < (double)NEWTONLIB_MEPS) {
            goto EXIT_SOLVE_SLE;
        }
        for (row = idx + 1; row < ndim; row++) {
            scale = matrix[row * ndim + idx] / diag;

            for (col = idx; col < ndim; col++) {
                matrix[row * ndim + col] -= matrix[idx * ndim + col] * scale;
            }
            vec[row] -= vec[idx] * scale;
        }
    }
    //! backward substitution
    for (row = ndim - 1; row >= 0; row--) {
        sum = vec[row];

        for (col = row + 1; col < ndim; col++) {
            sum -= matrix[row * ndim + col] * vec[col];
        }
        vec[row] = sum / matrix[row * ndim + row];
    }
    ret = (int32_t)NEWTONLIB_RETURN_OK;

EXIT_SOLVE_SLE:

    return ret;
}

/** @} */ // end of PrivateNewtonMethod
