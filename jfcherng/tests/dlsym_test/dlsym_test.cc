#include <iostream>
#include <string>
#include <dlfcn.h>

// To compile: g++ test.cc -ldl -o test

using namespace std;

int main (int argc, char *argv[]) {
  void *func_addr = NULL;

  // string canonical_name = "relu_op_jfcherng_xla_impl";

  void *handle = dlopen("/usr/local/lib/python3.5/dist-packages/tensorflow/python_pywrap_tensorflow_internal.so", RTLD_NOW);
  string canonical_name = "relu_op_jfcherng_xla_impl";

  if (!handle) {
    cout << dlerror() << endl;
  }

  cout << handle << endl;
  func_addr = dlsym(RTLD_DEFAULT, canonical_name.c_str());
  // func_addr = dlsym(handle, canonical_name.c_str());
  cout << func_addr << endl;

  return 0;
}
