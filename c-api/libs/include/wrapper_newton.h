#ifndef WRAPPER_NEWTON_H_
#define WRAPPER_NEWTON_H_

/**
 * @defgroup NewtonLib Wrapper Newton API
 * C library of newton method
 * @{
*/

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>
#define NEWTONLIB_SUCCESS    (0x00)
#define NEWTONLIB_FAILED     (0x10)
#define NEWTONLIB_JACOBIAN   (0x11)
#define NEWTONLIB_OBJFUNCVAL (0x12)
#define NEWTONLIB_SOLVE_SLE  (0x13)
#define NEWTONLIB_CONVERGENT (1)
#define NEWTONLIB_DIVERGENCE (0)
typedef int32_t (*NEWTONLIB_CALLBACK)(int32_t ndim, const double *input, double *output);

/**
 * @struct NewtonLib_ArgParam_t
 * @brief structure of argument of newton_method function
*/
struct NewtonLib_ArgParam_t {
    int32_t ndim;                //!< number of dimensions
    double *vec;                 //!< vector of x
    int32_t max_iter;            //!< maximum iteration
    double tol;                  //!< relative tolerance
    double delta;                //!< fixed difference value
    NEWTONLIB_CALLBACK callback; //!< callback function
};
/**
 * @struct NewtonLib_Output_t
 * @brief structure of output of newton_method function
*/
struct NewtonLib_Output_t {
    double *hat;            //!< estimated vector of x
    int32_t is_convergent;  //!< convergent status
};

/**
 * @fn extern int32_t NewtonLib_newton_method(const struct NewtonLib_ArgParam_t *args, struct NewtonLib_Output_t *output)
 * @brief execute newton method
 * @param[in]  args   function argument
 * @param[out] output output data
 * @return NEWTONLIB_SUCCESS    success
 *         NEWTONLIB_FAILED     failed
 *         NEWTONLIB_JACOBIAN   failed to calculate Jacobian matrix
 *         NEWTONLIB_OBJFUNCVAL failed to calculate objective function value
 *         NEWTONLIB_SOLVE_SLE  failed to solve simultaneous linear equations
*/
extern int32_t NewtonLib_newton_method(const struct NewtonLib_ArgParam_t *args, struct NewtonLib_Output_t *output);

#ifdef __cplusplus
}
#endif

/** @} */ // end of NewtonLib

#endif
