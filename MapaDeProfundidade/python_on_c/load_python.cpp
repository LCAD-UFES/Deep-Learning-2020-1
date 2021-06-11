#define PY_SSIZE_T_CLEAN
#include <Python.h>
// This line must be included before all numpy array imports
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/ndarrayobject.h>
// #include "numpy/ndarray.hpp"
#include <iostream>
#include "opencv2/opencv.hpp"


static PyObject *deepMapper, *inferencia, *initialize_function;

int initialize(){
    Py_Initialize();
    import_array();

    deepMapper = PyImport_ImportModule("deep_mapper");

    if (deepMapper == nullptr)
    {
        PyErr_Print();
        printf("error: fail to import module\n");
        return 1;
    }

    initialize_function = PyObject_GetAttrString(deepMapper, (char *)"initialize");
    if (initialize_function == nullptr)
    {
        PyErr_Print();
        printf("error: fail to get dictionary\n");
        return 1;
    }

    if (initialize_function && PyCallable_Check(initialize_function))
    {
        PyObject_CallObject(initialize_function, nullptr);
        inferencia = PyObject_GetAttrString(deepMapper, (char *)"inferenceDepth");
        if (inferencia == NULL || !PyCallable_Check(inferencia))
        {
            Py_Finalize();
            exit(printf("Error: Could not load the inferenceDepth.\n"));
        }

        if (PyErr_Occurred())
            PyErr_Print();
    }
    else
    {
        if (PyErr_Occurred())
            PyErr_Print();
        fprintf(stderr, "Cannot initiate object \n");
    }

    return 0;
}

int
inferDepth(){
    //A PARTIR DAQUI É O PROCESSO CHAMADO A CADA NOVA IMAGEM
    //simula o recebimento de uma imagem da IARA
    cv::Mat img = cv::imread("/home/thiago/mestrado/deeplearning_alberto/AdaBins/test_imgs/classroom__rgb_00283.jpg", cv::IMREAD_COLOR);
    cv::Mat cloned = img.clone();
    
    int nElem = 640 * 480 * 3;
    uchar *m = new uchar[nElem];
    
    std::memcpy(m, cloned.data, nElem * sizeof(uchar));
    
    npy_intp mdim[] = {480, 640, 3};
    PyObject *mat = PyArray_SimpleNewFromData(3, mdim, NPY_UINT8, (void *)m);
    
    PyObject* python_result_array = (PyObject*) PyObject_CallFunction(inferencia, (char *) "(O)", mat);
    
	if (PyErr_Occurred())
	        PyErr_Print();
    
	uchar *result_array = (uchar*)PyByteArray_AsString(python_result_array);
    
	if (PyErr_Occurred())
        PyErr_Print();
    
    cv::Mat saida(480, 640, CV_16UC1, result_array);
    std::cout << saida.channels();
    printf("CHEGUEI AQUI 3\n");
    cv::namedWindow("output");
    cv::imshow("output", saida);
    printf("CHEGUEI AQUI 4\n");
    cv::waitKey(0);
    Py_XDECREF(inferencia);
    Py_DECREF(deepMapper);

    if (Py_FinalizeEx() < 0)
    {
        return 120;
    }
}

int main(int argc, char *argv[])
{
    initialize();
    inferDepth();
    
    return 0;
}