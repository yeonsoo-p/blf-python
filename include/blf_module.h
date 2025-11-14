/*----------------------------------------------------------------------------
| BLF Python Module Header - Common definitions, macros, and structures
|-----------------------------------------------------------------------------
| This header provides common definitions for the BLF Python C extension module
| Platform: Windows (MinGW GCC/G++)
 ----------------------------------------------------------------------------*/

#ifndef BLF_MODULE_H
#define BLF_MODULE_H

// Python API definitions
#define PY_SSIZE_T_CLEAN
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
#include <Vector/DBC.h>
#include <numpy/arrayobject.h>
#include <stdint.h>
#include <string>
#include <unordered_map>
#include <vector>
/*----------------------------------------------------------------------------
| Helper Functions and Structures
 ----------------------------------------------------------------------------*/

// The standard-ish hash_combine logic (based on Boost)
template <class T>
inline void hash_combine(std::size_t& seed, const T& v) {
    std::hash<T> hasher;
    // The magic number is derived from the golden ratio
    seed ^= hasher(v) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
}

// Sanitize std::string to Python Unicode by skipping invalid UTF-8 sequences
inline PyObject* sanitized_PyUnicode_FromString(const std::string& str) {
    std::string sanitized;
    sanitized.reserve(str.size());

    for (size_t i = 0; i < str.size(); ) {
        unsigned char c = str[i];

        // ASCII (0x00-0x7F)
        if (c <= 0x7F) {
            sanitized += c;
            i++;
        }
        // 2-byte UTF-8 (0xC0-0xDF)
        else if ((c >= 0xC0 && c <= 0xDF) && i + 1 < str.size()) {
            unsigned char c2 = str[i + 1];
            if ((c2 & 0xC0) == 0x80) {  // Valid continuation byte
                sanitized += c;
                sanitized += c2;
                i += 2;
            } else {
                i++;  // Skip invalid byte
            }
        }
        // 3-byte UTF-8 (0xE0-0xEF)
        else if ((c >= 0xE0 && c <= 0xEF) && i + 2 < str.size()) {
            unsigned char c2 = str[i + 1];
            unsigned char c3 = str[i + 2];
            if ((c2 & 0xC0) == 0x80 && (c3 & 0xC0) == 0x80) {  // Valid continuation bytes
                sanitized += c;
                sanitized += c2;
                sanitized += c3;
                i += 3;
            } else {
                i++;  // Skip invalid byte
            }
        }
        // 4-byte UTF-8 (0xF0-0xF7)
        else if ((c >= 0xF0 && c <= 0xF7) && i + 3 < str.size()) {
            unsigned char c2 = str[i + 1];
            unsigned char c3 = str[i + 2];
            unsigned char c4 = str[i + 3];
            if ((c2 & 0xC0) == 0x80 && (c3 & 0xC0) == 0x80 && (c4 & 0xC0) == 0x80) {  // Valid continuation bytes
                sanitized += c;
                sanitized += c2;
                sanitized += c3;
                sanitized += c4;
                i += 4;
            } else {
                i++;  // Skip invalid byte
            }
        }
        // Invalid byte - skip it
        else {
            i++;
        }
    }

    return PyUnicode_FromString(sanitized.c_str());
}

// Helper to create a hash key from message ID and channel
struct MessageChannelKey {
    uint32_t message_id;
    uint16_t channel;

    bool operator==(const MessageChannelKey& other) const {
        return message_id == other.message_id && channel == other.channel;
    }
};

// Hash function for MessageChannelKey
struct MessageChannelKeyHash {
    std::size_t operator()(const MessageChannelKey& key) const {
        std::size_t seed = 0;
        hash_combine(seed, key.message_id);
        hash_combine(seed, key.channel);
        return seed;
    }
};

/*----------------------------------------------------------------------------
| Macro Utilities
 ----------------------------------------------------------------------------*/

// Function signature macros
#define BLF_FASTCALL(func) static PyObject* func(BLFObject* self, PyObject* const* args, Py_ssize_t nargs)
#define BLF_NOARGS(func)   static PyObject* func(BLFObject* self, PyObject* Py_UNUSED(args))

// Validation macros
#define BLF_CHECK_PARSED                                                \
    do {                                                                \
        if (!self->parsed) {                                            \
            PyErr_SetString(PyExc_RuntimeError, "BLF file not loaded"); \
            return NULL;                                                \
        }                                                               \
    } while (0)

#define BLF_CHECK_NARGS(expected)                                                                                                                     \
    do {                                                                                                                                              \
        if (nargs != (expected)) {                                                                                                                    \
            PyErr_Format(PyExc_TypeError, "%s() takes exactly %d argument%s (%zd given)", __func__, (expected), ((expected) == 1 ? "" : "s"), nargs); \
            return NULL;                                                                                                                              \
        }                                                                                                                                             \
    } while (0)

// Argument extraction macros
#define BLF_GET_STRING_ARG(var, index)                                       \
    do {                                                                     \
        PyObject* arg_obj = args[index];                                     \
        if (!PyUnicode_Check(arg_obj)) {                                     \
            PyErr_Format(PyExc_TypeError,                                    \
                         "Argument %d must be a string, got %s",             \
                         (int)(index),                                       \
                         Py_TYPE(arg_obj)->tp_name);                         \
            return NULL;                                                     \
        }                                                                    \
        if (!PyUnicode_IS_COMPACT_ASCII(arg_obj)) {                          \
            PyErr_Format(PyExc_ValueError,                                   \
                         "Argument %d must be ASCII-only string",            \
                         (int)(index));                                      \
            return NULL;                                                     \
        }                                                                    \
        Py_ssize_t size;                                                     \
        var = PyUnicode_AsUTF8AndSize(arg_obj, &size);                       \
        if (var == NULL) {                                                   \
            if (!PyErr_Occurred()) {                                         \
                PyErr_SetString(PyExc_RuntimeError,                          \
                                "Failed to decode string argument");         \
            }                                                                \
            return NULL;                                                     \
        }                                                                    \
        if (strlen(var) != (size_t)size) {                                   \
            PyErr_SetString(PyExc_ValueError,                                \
                            "String argument contains embedded null bytes"); \
            return NULL;                                                     \
        }                                                                    \
    } while (0)

// Message lookup macro - raises KeyError if not found
#define BLF_FIND_MESSAGE(msg_var, msg_name)                               \
    auto msg_var##_it = self->messages_data.find(msg_name);               \
    if (msg_var##_it == self->messages_data.end()) {                      \
        PyErr_Format(PyExc_KeyError, "Message '%s' not found", msg_name); \
        return NULL;                                                      \
    }                                                                     \
    const MessageData& msg_var = msg_var##_it->second;

// Signal lookup macro - raises KeyError if not found
#define BLF_FIND_SIGNAL(msg_var, sig_var, sig_name)                      \
    auto sig_var##_it = msg_var.signal_metadata.find(sig_name);          \
    if (sig_var##_it == msg_var.signal_metadata.end()) {                 \
        PyErr_Format(PyExc_KeyError, "Signal '%s' not found", sig_name); \
        return NULL;                                                     \
    }                                                                    \
    const SignalMetadata& sig_var = sig_var##_it->second;

// Python object creation macros
#define BLF_NEW_DICT(var)                                                  \
    do {                                                                   \
        var = PyDict_New();                                                \
        if (var == NULL) {                                                 \
            PyErr_SetString(PyExc_MemoryError, "Failed to allocate dict"); \
            return NULL;                                                   \
        }                                                                  \
    } while (0)

#define BLF_NEW_LIST(var, size)                                            \
    do {                                                                   \
        var = PyList_New(size);                                            \
        if (var == NULL) {                                                 \
            PyErr_SetString(PyExc_MemoryError, "Failed to allocate list"); \
            return NULL;                                                   \
        }                                                                  \
    } while (0)

/*----------------------------------------------------------------------------
| Data Structures
 ----------------------------------------------------------------------------*/

// Signal metadata for type information and units
struct SignalMetadata {
    std::string name;
    std::string unit;
    double      factor;
    double      offset;
    bool        is_signed;
    uint32_t    bit_size;
};

// Temporary structure for collecting signal data during parsing (scaled values)
struct TempSignalData {
    std::string         name;
    std::vector<double> values; // Store scaled physical values
};

// Temporary structure for collecting message data during parsing
struct TempMessageData {
    std::string                                     name;
    uint32_t                                        id;
    std::unordered_map<std::string, TempSignalData> signals;
    std::vector<double>                             timestamps;
    std::unordered_map<std::string, SignalMetadata> signal_metadata;
};

// Final optimized structure with consolidated 2D array
// Layout: data includes ALL signals including Time at column 0
struct MessageData {
    std::string                                     name;
    uint32_t                                        id;
    std::vector<std::string>                        signal_names;     // Ordered list: ["Time", signal1, signal2, ...]
    std::unordered_map<std::string, size_t>         signal_index_map; // Signal name -> column index (O(1) lookup)
    std::vector<double>                             data;             // Flattened 2D: Time in col 0, signals in cols 1+
    std::unordered_map<std::string, SignalMetadata> signal_metadata;  // Metadata per signal (including Time)
    size_t                                          num_samples;
    size_t                                          num_signals; // Total number of signals (including Time)

    // Helper method to get signal column index
    int get_signal_index(const std::string& signal_name) const {
        auto it = signal_index_map.find(signal_name);
        if (it == signal_index_map.end()) {
            return -1;
        }
        return static_cast<int>(it->second);
    }
};

// Channel-DBC mapping structure
struct ChannelDBCMapping {
    int         channel;
    std::string dbc_path;
};

// BLF Python object structure
typedef struct {
    PyObject_HEAD;
    std::unordered_map<std::string, MessageData>                                   messages_data;
    std::unordered_map<MessageChannelKey, size_t, MessageChannelKeyHash>           dbc_network_cache; // Maps to network index instead of pointer
    std::vector<Vector::DBC::Network>                                              networks;          // Store networks
    std::vector<int>                                                               network_channels;  // Store which channel each network belongs to
    int                                                                            initialized;
    int                                                                            parsed;
} BLFObject;

#endif // BLF_MODULE_H
