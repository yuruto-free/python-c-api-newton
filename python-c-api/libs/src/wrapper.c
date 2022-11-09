#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <numpy/arrayobject.h>

/**
 *  @defgroup PrivateWrapper Private functions in wrapper.c
 *  @brief Private functions in wrapper.c
 *  @{
*/

//! include C API header
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include "wrapper_newton.h"
#define NEWTONLIB_DELTA (1e-3)
#define NEWTONLIB_TOL (1e-10)

//! define objective function in python code
static PyObject *py_objective_function = NULL;

/**
 * @fn static int32_t alternative_callback_function(int32_t ndim, const double *input, double *output);
 * @brief call python function from c library
 * @param[in]  ndim  number of dimensions
 * @param[in]  input input vector
 * @param[out] outpu outpu vector
 * @return NEWTONLIB_SUCCESS success
 *         NEWTONLIB_FAILED  failed
*/
static int32_t alternative_callback_function(int32_t ndim, const double *input, double *output) {
    int32_t ret = (int32_t)NEWTONLIB_FAILED;
    int32_t idx;
    PyObject *numpy_list, *py_data, *py_out;
    PyArrayObject *np_arr = NULL;
    npy_intp dims[1] = {0};

    if (NULL != py_objective_function) {
        //! set argument
        dims[0] = ndim;
        numpy_list = PyArray_SimpleNew(1, &dims[0], NPY_DOUBLE);
        np_arr = (PyArrayObject *)numpy_list;

        //! store input to numpy array
        for (idx = 0; idx < ndim; idx++) {
            py_data = PyFloat_FromDouble(input[idx]);
            PyArray_SETITEM(np_arr, PyArray_GETPTR1(np_arr, idx), py_data);
            Py_DECREF(py_data);
        }
        //! call function
        py_out = PyObject_CallFunctionObjArgs(py_objective_function, np_arr, NULL);
        Py_DECREF(numpy_list);

        //! check result
        if (py_out && PyArray_Check(py_out)) {
            np_arr = (PyArrayObject *)py_out;

            //! store output from numpy array
            for (idx = 0; idx < ndim; idx++) {
                output[idx] = *((double *)PyArray_GETPTR1(np_arr, idx));
            }
            Py_DECREF(py_out);
            ret = (int32_t)NEWTONLIB_SUCCESS;
        }
    }

    return ret;
}

/**
 * @fn static void store_result(PyObject **py_output, int32_t ndim, double *hat, int32_t is_convergent)
 * @brief convert output data of c format to python object
 * @param[out] py_output     output python object
 * @param[in]  ndim          dimension
 * @param[in]  hat           estimated vector of x
 * @param[in]  is_convergent convergent status
 * @return none
*/
static void store_result(PyObject **py_output, int32_t ndim, double *hat, int32_t is_convergent) {
    PyObject *numpy_list, *py_data, *py_convergent;
    PyArrayObject *np_arr;
    int32_t idx;
    npy_intp dims[1] = {0};

    dims[0] = ndim;
    numpy_list = PyArray_SimpleNew(1, &dims[0], NPY_DOUBLE);
    np_arr = (PyArrayObject *)numpy_list;

    for (idx = 0; idx < ndim; idx++) {
        py_data = PyFloat_FromDouble(hat[idx]);
        PyArray_SETITEM(np_arr, PyArray_GETPTR1(np_arr, idx), py_data);
        Py_DECREF(py_data);
    }
    py_convergent = ((int32_t)NEWTONLIB_CONVERGENT == is_convergent) ? Py_True : Py_False;
    Py_INCREF(py_convergent);
    PyList_Append(*py_output, numpy_list);
    PyList_Append(*py_output, py_convergent);
}

/**
 * @fn static PyObject *update_function(PyObject *self, PyObject *args)
 * @brief set objective function used by Newton's method
 * @param[in] self   python object
 * @param[in] args   arguments from python script
 * @return Py_None
*/
static PyObject *update_function(PyObject *self, PyObject *args) {
    PyObject *tmp;

    if (!PyArg_ParseTuple(args, "O:set_callback", &tmp)) {
        PyErr_SetString(PyExc_TypeError, "Invalid argument");
        goto EXIT_SET_OBJECTIVE_FUNCTION;
    }
    if (!PyCallable_Check(tmp)) {
        PyErr_SetString(PyExc_TypeError, "parameter must be callable");
        goto EXIT_SET_OBJECTIVE_FUNCTION;
    }
    Py_XINCREF(tmp);                    //! add a reference to new callback
    Py_XDECREF(py_objective_function);  //! dispose of previous callback
    py_objective_function = tmp;

EXIT_SET_OBJECTIVE_FUNCTION:

    Py_RETURN_NONE;
}

/**
 * @fn static PyObject *newtonlib(PyObject *self, PyObject *args)
 * @brief wrapper function of newton_method
 * @param[in] self   python object
 * @param[in] args   arguments from python script
 * @param[in] keywds keyword arguments from python script
 * @return py_output
*/
static PyObject *newtonlib(PyObject *self, PyObject *args, PyObject *keywords) {
    PyObject *py_vec, *py_output;
    PyArrayObject *np_arr;
    int32_t idx;
    int32_t func_val;
    int32_t ndim;
    double *vec, *hat;
    static char *kwlist[] = {"vec", "max_iter", "tol", "delta", NULL};
    struct NewtonLib_ArgParam_t params;
    struct NewtonLib_Output_t output;

    //! initialize
    py_vec = NULL;
    py_output = PyList_New(0);
    vec = NULL;
    hat = NULL;
    params.tol = (double)NEWTONLIB_TOL;
    params.delta = (double)NEWTONLIB_DELTA;

    //! Step1: check arguments
    if (!PyArg_ParseTupleAndKeywords(args, keywords, "Oi|dd", kwlist, &py_vec, &params.max_iter, &params.tol, &params.delta)) {
        PyErr_SetString(PyExc_TypeError, "Invalid argument");
        goto EXIT_NEWTONLIB;
    }
    //! Step2: check object type
    if (!PyArray_Check(py_vec)) {
        PyErr_SetString(PyExc_TypeError, "'vec' needs to numpy array");
        goto EXIT_NEWTONLIB;
    }
    //! Step3: get vector size
    np_arr = (PyArrayObject *)py_vec;
    ndim = (int32_t)PyArray_SIZE(np_arr);
    //! Step4: malloc
    vec = (double *)malloc(sizeof(double) * ndim);
    hat = (double *)malloc(sizeof(double) * ndim);
    if ((NULL == vec) || (NULL == hat)) {
        PyErr_SetString(PyExc_MemoryError, "lack of memory");
        goto EXIT_NEWTONLIB;
    }
    //! Step5: copy data
    for (idx = 0; idx < ndim; idx++) {
        vec[idx] = *((double *)PyArray_GETPTR1(np_arr, idx));
    }
    //! Step6: call newton_method function
    params.ndim = ndim;
    params.vec = vec;
    params.callback = alternative_callback_function;
    output.hat = hat;
    func_val = NewtonLib_newton_method((const struct NewtonLib_ArgParam_t *)&params, &output);
    if ((int32_t)NEWTONLIB_SUCCESS != func_val) {
        switch ((int)func_val) {
            case NEWTONLIB_JACOBIAN:
                PyErr_SetString(PyExc_RuntimeError, "[Newton's method]Failed to calculate Jacobian matrix.");
                break;
            case NEWTONLIB_OBJFUNCVAL:
                PyErr_SetString(PyExc_RuntimeError, "[Newton's method]Failed to calculate objective function value.");
                break;
            case NEWTONLIB_SOLVE_SLE:
                PyErr_SetString(PyExc_RuntimeError, "[Newton's method]Failed to solve simultaneous linear equations.");
                break;
            default:
                break;
        }
        goto EXIT_NEWTONLIB;
    }
    //! Step7: store result
    store_result(&py_output, ndim, hat, output.is_convergent);

EXIT_NEWTONLIB:

    //! finalize
    if (NULL != vec) {
        free(vec);
        vec = NULL;
    }
    if (NULL != hat) {
        free(hat);
        hat = NULL;
    }

    return py_output;
}

/** @} */ // end of PrivateWrapper

//! register c functions to use python script
static PyMethodDef wrapper_methods[] = {
    {
        /**
         *  @defgroup PublicSetterAPI
         *  @brief setter module name of objective function
         *  @{
        */
        "set_objective_function",
        /** @} */ // end of PublicSetterAPI
        //! target c function
        update_function,
        //! argument type
        METH_VARARGS,
        //! output message of __doc__ function
        "Set objective function used by Newton's method",
    },
    {
        /**
         *  @defgroup PublicWrapperAPI
         *  @brief newton method module name
         *  @{
        */
        "newton_method",
        /** @} */ // end of PublicWrapperAPI
        //! target c function
        (PyCFunction)newtonlib,
        //! argument type
        METH_VARARGS | METH_KEYWORDS,
        //! output message of __doc__ function
        "Calculate approximate solutions using Newton's method",
    },
    {NULL, NULL, 0, NULL},
};

//! defile python module
static struct PyModuleDef wrapper_newtonlib = {
    PyModuleDef_HEAD_INIT,
    "wrapper_newtonlib",
    NULL,
    -1,
    wrapper_methods,
};

/**
 * @fn PyMODINIT_FUNC PyInit_wrapper_newton(void)
 * @brief PyInit function
*/
PyMODINIT_FUNC PyInit_wrapper_newtonlib(void) {
    import_array();

    return PyModule_Create(&wrapper_newtonlib);
}
