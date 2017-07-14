#ifndef TENSORFLOW_JFCHERNG_LOGGING_H_
#define TENSORFLOW_JFCHERNG_LOGGING_H_

#include <cstdlib>
#include <sstream>
#include <string>

// The codes for foreground and background colours are:
//          foreground background
// black        30         40
// red          31         41
// green        32         42
// yellow       33         43
// blue         34         44
// magenta      35         45
// cyan         36         46
// white        37         47
//
// Additionally, you can use these:
// reset             0  (everything back to normal)
// bold/bright       1  (often a brighter shade of the same colour)
// underline         4
// inverse           7  (swap foreground and background colours)
// bold/bright off  21
// underline off    24
// inverse off      27

#define JFCHERNG_VLOG(lvl, type) \
    if (TF_PREDICT_FALSE(::tensorflow::jfcherng::getenv<int>("JFCHERNG_DEBUG"))) \
        VLOG(lvl) << "\033[1;33m" << "jfcherng: " \
                  << (type == "" ? "" : "\033[1;36m" type ": ") \
                  << "\033[0m"

namespace tensorflow {
namespace jfcherng {

/**
 * Get the value of an environment variable.
 * @param  envName [the name of the environment variable]
 * @return         [the value of the environment variable]
 */
template <typename T>
T getenv(const std::string &envName) {
    const char * const envVal = std::getenv(envName.c_str());

    // when cannot find `envName`, we return a default constructed `T`
    if (!envVal) return T{};

    // convert `envVal` into `T` type
    T result;
    std::stringstream ss{envVal};
    ss >> result;

    return result;
}

template <>
std::string getenv<std::string>(const std::string &envName);

} // end namespace jfcherng
} // end namespace tensorflow

#endif
