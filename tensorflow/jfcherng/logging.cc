#include <cstdlib>
#include "tensorflow/jfcherng/logging.h"

namespace tensorflow {
namespace jfcherng {

template <>
std::string getenv<std::string>(const std::string &envName) {
    const char *const envVal = std::getenv(envName.c_str());
    return envVal ? envVal : "";
}

} // end namespace jfcherng
} // end namespace tensorflow
