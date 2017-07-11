#include <cstdlib>
#include <string>
#include "tensorflow/jfcherng/logging.h"

namespace tensorflow {
namespace jfcherng {

int getenv_int(const std::string &env_name) {
    const char *const env_val = std::getenv(env_name.c_str());
    return std::stoi(std::string{ env_val ? env_val : "0" });
}

std::string getenv_str(const std::string &env_name) {
    const char *const env_val = std::getenv(env_name.c_str());
    return std::string{ env_val ? env_val : "" };
}

} // end namespace jfcherng
} // end namespace tensorflow
