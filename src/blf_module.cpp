/*----------------------------------------------------------------------------
| BLF Python Module - Read and decode BLF files with DBC databases
|-----------------------------------------------------------------------------
| This Python C extension module provides functionality to:
| - Read BLF files and extract CAN messages
| - Decode messages using DBC databases
| - Store decoded signals as NumPy arrays organized by message/signal names
|
| Platform: Windows (MinGW GCC/G++)
 ----------------------------------------------------------------------------*/

#define PY_SSIZE_T_CLEAN
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
#include <numpy/arrayobject.h>

#include <chrono>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>
#include <windows.h>

#include "binlog.h"
#include <Vector/DBC.h>

// Fast timing for Windows
#ifdef _WIN32
static LARGE_INTEGER g_frequency;
static bool          g_frequency_initialized = false;

inline LARGE_INTEGER get_performance_counter() {
    LARGE_INTEGER counter;
    QueryPerformanceCounter(&counter);
    return counter;
}

inline void init_performance_frequency() {
    if (!g_frequency_initialized) {
        QueryPerformanceFrequency(&g_frequency);
        g_frequency_initialized = true;
    }
}

inline long long counter_diff_ns(LARGE_INTEGER start, LARGE_INTEGER end) {
    return ((end.QuadPart - start.QuadPart) * 1000000000LL) / g_frequency.QuadPart;
}
#endif

// Structure to hold signal data during collection
struct SignalData {
    std::string         name;
    std::vector<double> values;
    // Note: timestamps are stored in parent MessageData to avoid duplication
};

// Structure to hold message data
struct MessageData {
    std::string                                 name;
    uint32_t                                    id;
    std::unordered_map<std::string, SignalData> signals;
    std::vector<double>                         timestamps;
};

// The standard-ish hash_combine logic (based on Boost)
template <class T>
inline void hash_combine(std::size_t& seed, const T& v) {
    std::hash<T> hasher;
    // The magic number is derived from the golden ratio
    seed ^= hasher(v) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
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

        // 1. Hash the first number (uint32_t) and combine
        hash_combine(seed, key.message_id);

        // 2. Hash the second number (uint16_t) and combine
        hash_combine(seed, key.channel);

        return seed;
    }
};

// BLF object structure
typedef struct {
    PyObject_HEAD
        std::unordered_map<std::string, MessageData>*                                    messages_data;
    std::unordered_map<MessageChannelKey, Vector::DBC::Network*, MessageChannelKeyHash>* dbc_network_cache;
    std::vector<Vector::DBC::Network>                                                    networks;         // Store networks to keep Network pointers valid
    std::vector<int>                                                                     network_channels; // Store which channel each network belongs to
    int                                                                                  initialized;
    int                                                                                  parsed;
} BLFObject;

// Extract raw value from CAN data bytes
static uint64_t extractRawValue(const uint8_t* data, size_t dataLen, const Vector::DBC::Signal& signal) {
    uint64_t rawValue = 0;

    if (signal.byteOrder == Vector::DBC::ByteOrder::BigEndian) {
        // Motorola (Big Endian) byte order
        uint32_t bitPos = signal.startBit;
        for (uint32_t i = 0; i < signal.bitSize; ++i) {
            uint32_t byteIdx = bitPos / 8;
            uint32_t bitIdx  = 7 - (bitPos % 8);

            if (byteIdx < dataLen) {
                uint64_t bit = (data[byteIdx] >> bitIdx) & 0x01;
                rawValue |= (bit << (signal.bitSize - 1 - i));
            }

            bitPos++;
        }
    } else {
        // Intel (Little Endian) byte order
        uint32_t bitPos = signal.startBit;
        for (uint32_t i = 0; i < signal.bitSize; ++i) {
            uint32_t byteIdx = bitPos / 8;
            uint32_t bitIdx  = bitPos % 8;

            if (byteIdx < dataLen) {
                uint64_t bit = (data[byteIdx] >> bitIdx) & 0x01;
                rawValue |= (bit << i);
            }

            bitPos++;
        }
    }

    // Handle signed values
    if (signal.valueType == Vector::DBC::ValueType::Signed) {
        // Sign-extend if necessary (but only if not full 64-bit)
        // For 64-bit signals, (1ULL << 64) is undefined behavior, and no extension is needed
        if (signal.bitSize < 64) {
            uint64_t signBit = 1ULL << (signal.bitSize - 1);
            if (rawValue & signBit) {
                uint64_t mask = (1ULL << signal.bitSize) - 1;
                rawValue |= ~mask;
            }
        }
    }

    return rawValue;
}

// BLF.__new__
static PyObject* BLF_new(PyTypeObject* type, PyObject* args, PyObject* kwds) {
    BLFObject* self;
    (void)args;
    (void)kwds;
    self = (BLFObject*)type->tp_alloc(type, 0);
    if (self != NULL) {
        self->messages_data     = nullptr;
        self->dbc_network_cache = nullptr;
        self->initialized       = 0;
        self->parsed            = 0;
    }
    return (PyObject*)self;
}

// Helper function to convert Python path object (str or Path) to C string
static const char* get_path_string(PyObject* path_obj) {
    // If it's already a string, return it directly
    if (PyUnicode_Check(path_obj)) {
        return PyUnicode_AsUTF8(path_obj);
    }

    // If it's a Path object, call __str__() or __fspath__()
    if (PyObject_HasAttrString(path_obj, "__fspath__")) {
        PyObject* fspath = PyObject_CallMethod(path_obj, "__fspath__", NULL);
        if (fspath && PyUnicode_Check(fspath)) {
            const char* result = PyUnicode_AsUTF8(fspath);
            Py_DECREF(fspath);
            return result;
        }
        Py_XDECREF(fspath);
    }

    // Try __str__() as fallback
    PyObject* str_obj = PyObject_Str(path_obj);
    if (str_obj && PyUnicode_Check(str_obj)) {
        const char* result = PyUnicode_AsUTF8(str_obj);
        Py_DECREF(str_obj);
        return result;
    }
    Py_XDECREF(str_obj);

    return nullptr;
}

// BLF.__init__
static int BLF_init(BLFObject* self, PyObject* args, PyObject* kwds) {
    PyObject* blf_filepath_obj;
    PyObject* channel_dbc_list;

    static char* kwlist[] = {(char*)"blf_filepath", (char*)"channel_dbc_list", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "OO", kwlist,
                                     &blf_filepath_obj, &channel_dbc_list)) {
        return -1;
    }

    // Convert BLF filepath (str or Path)
    const char* blf_filepath = get_path_string(blf_filepath_obj);
    if (!blf_filepath) {
        PyErr_SetString(PyExc_TypeError, "blf_filepath must be a string or Path object");
        return -1;
    }

    // Parse list of (channel, dbc_filepath) tuples
    if (!PyList_Check(channel_dbc_list)) {
        PyErr_SetString(PyExc_TypeError, "channel_dbc_list must be a list of (channel, dbc_filepath) tuples");
        return -1;
    }

    Py_ssize_t numEntries = PyList_Size(channel_dbc_list);
    if (numEntries == 0) {
        PyErr_SetString(PyExc_ValueError, "At least one (channel, dbc_filepath) tuple must be provided");
        return -1;
    }

    // Store mapping of channel -> network index for cache building
    // Special handling: channel -1 means "wildcard" (matches all channels)
    std::unordered_map<int, size_t> channel_to_network_idx;

    // Parse each (channel, dbc_filepath) tuple
    for (Py_ssize_t i = 0; i < numEntries; ++i) {
        PyObject* tuple = PyList_GetItem(channel_dbc_list, i);

        if (!PyTuple_Check(tuple) || PyTuple_Size(tuple) != 2) {
            PyErr_SetString(PyExc_TypeError, "Each entry must be a (channel, dbc_filepath) tuple");
            return -1;
        }

        // Extract channel
        PyObject* channel_obj = PyTuple_GetItem(tuple, 0);
        if (!PyLong_Check(channel_obj)) {
            PyErr_SetString(PyExc_TypeError, "Channel must be an integer");
            return -1;
        }
        long channel = PyLong_AsLong(channel_obj);
        if (channel < -1 || channel > 65535) {
            PyErr_SetString(PyExc_ValueError, "Channel must be between -1 (wildcard) and 65535");
            return -1;
        }
        int channel_id = static_cast<int>(channel); // Use int to support -1

        // Extract DBC filepath (str or Path)
        PyObject*   dbc_path_obj = PyTuple_GetItem(tuple, 1);
        const char* dbc_filepath = get_path_string(dbc_path_obj);
        if (!dbc_filepath) {
            PyErr_SetString(PyExc_TypeError, "DBC filepath must be a string or Path object");
            return -1;
        }

        // Load DBC file
        Vector::DBC::Network network;
        std::ifstream        ifs(dbc_filepath);

        if (!ifs.is_open()) {
            PyErr_Format(PyExc_IOError, "Could not open DBC file: %s", dbc_filepath);
            return -1;
        }

        ifs >> network;
        ifs.close();

        if (!network.successfullyParsed) {
            PyErr_Format(PyExc_ValueError, "Failed to parse DBC file: %s", dbc_filepath);
            return -1;
        }

        // Store network and remember which channel it belongs to
        size_t network_idx = self->networks.size();
        self->networks.push_back(std::move(network));
        self->network_channels.push_back(channel_id); // Track channel for this network
        channel_to_network_idx[channel_id] = network_idx;
    }

    // Allocate cache for (message_id, channel) -> network lookups
    // Cache will be populated on-the-fly during BLF reading (first search is always cache miss)
    self->dbc_network_cache = new std::unordered_map<MessageChannelKey, Vector::DBC::Network*, MessageChannelKeyHash>();

    // Open BLF file
    BLHANDLE hFile = BLCreateFile(blf_filepath, GENERIC_READ);

    if (BLINVALID_HANDLE_VALUE == hFile) {
        PyErr_Format(PyExc_IOError, "Could not open BLF file: %s", blf_filepath);
        return -1;
    }

    // Allocate messages map on heap - owned by this BLF object
    self->messages_data = new std::unordered_map<std::string, MessageData>();
    self->initialized   = 1;

    auto& messagesData = *self->messages_data;

    // Timing accumulators for loop breakdown (in nanoseconds)
    long long total_measured_iteration_ns = 0;
    long long peek_read_time_ns           = 0;
    long long dbc_lookup_time_ns          = 0;
    long long decode_setup_time_ns        = 0;
    long long extract_signals_time_ns     = 0;
    long long timing_overhead_ns          = 0;
    long long gap_time_ns                 = 0;
    long long iteration_count             = 0;
    long long messages_decoded            = 0;

    // Read all objects from BLF file
    VBLObjectHeaderBase base;
    int32_t             bSuccess = 1;

    init_performance_frequency();
    auto loop_start    = get_performance_counter();
    auto prev_iter_end = loop_start;

    while (bSuccess) {
        iteration_count++;
        auto iter_start = get_performance_counter();
        gap_time_ns += counter_diff_ns(prev_iter_end, iter_start);

        auto peek_start = get_performance_counter();

        int peekResult = BLPeekObject(hFile, &base);

        if (!peekResult)
            break;
        uint32_t msgId        = 0;
        uint16_t msgChannel   = 0;
        uint8_t  msgDlc       = 0;
        uint8_t  msgData[64]  = {0};
        uint64_t timestamp    = 0;
        bool     validMessage = false;

        switch (base.mObjectType) {
        case BL_OBJ_TYPE_CAN_MESSAGE: {
            VBLCANMessage message;
            memset(&message, 0, sizeof(VBLCANMessage));
            message.mHeader.mBase = base;
            bSuccess              = BLReadObjectSecure(hFile, &message.mHeader.mBase, sizeof(VBLCANMessage));

            if (bSuccess) {
                msgId      = message.mID;
                msgChannel = message.mChannel;
                msgDlc     = message.mDLC;
                memcpy(msgData, message.mData, (msgDlc < 8) ? msgDlc : 8);
                timestamp    = message.mHeader.mObjectTimeStamp;
                validMessage = true;
                BLFreeObject(hFile, &message.mHeader.mBase);
            }
            break;
        }

        case BL_OBJ_TYPE_CAN_MESSAGE2: {
            VBLCANMessage2 message2;
            memset(&message2, 0, sizeof(VBLCANMessage2));
            message2.mHeader.mBase = base;
            bSuccess               = BLReadObjectSecure(hFile, &message2.mHeader.mBase, sizeof(VBLCANMessage2));

            if (bSuccess) {
                msgId      = message2.mID;
                msgChannel = message2.mChannel;
                msgDlc     = message2.mDLC;
                memcpy(msgData, message2.mData, (msgDlc < 8) ? msgDlc : 8);
                timestamp    = message2.mHeader.mObjectTimeStamp;
                validMessage = true;
                BLFreeObject(hFile, &message2.mHeader.mBase);
            }
            break;
        }

        case BL_OBJ_TYPE_CAN_FD_MESSAGE: {
            VBLCANFDMessage fdmessage;
            memset(&fdmessage, 0, sizeof(VBLCANFDMessage));
            fdmessage.mHeader.mBase = base;
            bSuccess                = BLReadObjectSecure(hFile, &fdmessage.mHeader.mBase, sizeof(VBLCANFDMessage));

            if (bSuccess) {
                msgId           = fdmessage.mID;
                msgChannel      = fdmessage.mChannel;
                msgDlc          = fdmessage.mDLC;
                uint8_t dataLen = (fdmessage.mValidDataBytes < 64) ? fdmessage.mValidDataBytes : 64;
                memcpy(msgData, fdmessage.mData, dataLen);
                timestamp    = fdmessage.mHeader.mObjectTimeStamp;
                validMessage = true;
                BLFreeObject(hFile, &fdmessage.mHeader.mBase);
            }
            break;
        }

        case BL_OBJ_TYPE_CAN_FD_MESSAGE_64: {
            VBLCANFDMessage64 fdmessage64;
            memset(&fdmessage64, 0, sizeof(VBLCANFDMessage64));
            fdmessage64.mHeader.mBase = base;
            bSuccess                  = BLReadObjectSecure(hFile, &fdmessage64.mHeader.mBase, sizeof(VBLCANFDMessage64));

            if (bSuccess) {
                msgId           = fdmessage64.mID;
                msgChannel      = fdmessage64.mChannel;
                msgDlc          = fdmessage64.mDLC;
                uint8_t dataLen = (fdmessage64.mValidDataBytes < 64) ? fdmessage64.mValidDataBytes : 64;
                memcpy(msgData, fdmessage64.mData, dataLen);
                timestamp    = fdmessage64.mHeader.mObjectTimeStamp;
                validMessage = true;
                BLFreeObject(hFile, &fdmessage64.mHeader.mBase);
            }
            break;
        }

        default:
            bSuccess = BLSkipObject(hFile, &base);
            break;
        }

        auto peek_end = get_performance_counter();
        peek_read_time_ns += counter_diff_ns(peek_start, peek_end);

        // Process valid CAN message
        if (validMessage) {
            auto lookup_start = get_performance_counter();
            gap_time_ns += counter_diff_ns(peek_end, lookup_start);

            // Find matching message in DBC files using cache
            // The cache maps (message_id, channel) -> network
            const Vector::DBC::Message* dbcMessage = nullptr;

            // Create cache key
            MessageChannelKey cache_key{msgId, msgChannel};

            // Check cache first
            auto cacheIt = self->dbc_network_cache->find(cache_key);
            if (cacheIt != self->dbc_network_cache->end()) {
                // Cache hit - look up message in the cached network
                Vector::DBC::Network* cached_network = cacheIt->second;
                auto                  msgIt          = cached_network->messages.find(msgId);
                if (msgIt != cached_network->messages.end()) {
                    dbcMessage = &msgIt->second;
                }
            } else {
                // Cache miss - search all networks and cache the result
                // Check if message channel matches the network's channel
                for (size_t i = 0; i < self->networks.size(); ++i) {
                    auto& network         = self->networks[i];
                    int   network_channel = self->network_channels[i];

                    // Skip if channel doesn't match (unless network channel is wildcard -1)
                    if (network_channel != -1 && network_channel != static_cast<int>(msgChannel)) {
                        continue;
                    }

                    auto msgIt = network.messages.find(msgId);
                    if (msgIt != network.messages.end()) {
                        dbcMessage = &msgIt->second;
                        // Cache this lookup for future messages on this channel
                        (*self->dbc_network_cache)[cache_key] = &network;
                        break;
                    }
                }
            }

            auto lookup_end = get_performance_counter();
            dbc_lookup_time_ns += counter_diff_ns(lookup_start, lookup_end);

            if (dbcMessage) {
                messages_decoded++;
                auto setup_start = get_performance_counter();
                gap_time_ns += counter_diff_ns(lookup_end, setup_start);

                // Convert timestamp from nanoseconds to seconds
                double timestampSec = timestamp / 1e9;

                // Get or create message data storage
                std::string msgName         = dbcMessage->name;
                auto&       msgData_storage = messagesData[msgName];

                if (msgData_storage.name.empty()) {
                    msgData_storage.name = msgName;
                    msgData_storage.id   = msgId;
                }

                // Add timestamp to message
                msgData_storage.timestamps.push_back(timestampSec);

                auto setup_end = get_performance_counter();
                decode_setup_time_ns += counter_diff_ns(setup_start, setup_end);

                // Decode all signals
                auto extract_start = get_performance_counter();
                gap_time_ns += counter_diff_ns(setup_end, extract_start);
                for (const auto& sigPair : dbcMessage->signals) {
                    const Vector::DBC::Signal& signal = sigPair.second;

                    // Extract raw value
                    uint64_t rawValue = extractRawValue(msgData, msgDlc, signal);

                    // Convert to physical value
                    double physicalValue;
                    if (signal.valueType == Vector::DBC::ValueType::Signed) {
                        int64_t signedRaw = static_cast<int64_t>(rawValue);
                        physicalValue     = static_cast<double>(signedRaw) * signal.factor + signal.offset;
                    } else {
                        physicalValue = static_cast<double>(rawValue) * signal.factor + signal.offset;
                    }

                    // Store signal value (timestamps are stored in parent MessageData)
                    auto& sigData = msgData_storage.signals[signal.name];
                    if (sigData.name.empty()) {
                        sigData.name = signal.name;
                    }
                    sigData.values.push_back(physicalValue);
                }
                auto extract_end = get_performance_counter();
                extract_signals_time_ns += counter_diff_ns(extract_start, extract_end);
                total_measured_iteration_ns += counter_diff_ns(iter_start, extract_end);
                prev_iter_end = extract_end;
            } else {
                auto lookup_end2 = get_performance_counter();
                timing_overhead_ns += counter_diff_ns(lookup_start, lookup_end2);
                total_measured_iteration_ns += counter_diff_ns(iter_start, lookup_end2);
                prev_iter_end = lookup_end2;
            }
        } else {
            auto overhead_end = get_performance_counter();
            timing_overhead_ns += counter_diff_ns(peek_end, overhead_end);
            total_measured_iteration_ns += counter_diff_ns(iter_start, overhead_end);
            prev_iter_end = overhead_end;
        }
    }

    BLCloseHandle(hFile);

    auto loop_end = get_performance_counter();

    self->parsed = 1;

    // Calculate total loop time
    long long total_loop_ns         = counter_diff_ns(loop_start, loop_end);
    double    total_loop_ms         = total_loop_ns / 1000000.0;
    double    measured_iteration_ms = total_measured_iteration_ns / 1000000.0;
    double    unmeasured_ms         = total_loop_ms - measured_iteration_ms;

    double sum_of_components_ms        = (peek_read_time_ns + dbc_lookup_time_ns + decode_setup_time_ns + extract_signals_time_ns + timing_overhead_ns + gap_time_ns) / 1000000.0;
    double unaccounted_in_iteration_ms = measured_iteration_ms - sum_of_components_ms;

    std::cout << "BLF_init loop timing breakdown:" << std::endl;
    std::cout << "  Iterations: " << iteration_count << ", Messages decoded: " << messages_decoded << std::endl;
    std::cout << "  Peek/Read messages:      " << (peek_read_time_ns / 1000000.0) << " ms  (" << (peek_read_time_ns / iteration_count) << " ns/iter)" << std::endl;
    std::cout << "  DBC lookup:              " << (dbc_lookup_time_ns / 1000000.0) << " ms  (" << (dbc_lookup_time_ns / iteration_count) << " ns/iter)" << std::endl;
    std::cout << "  Decode setup:            " << (decode_setup_time_ns / 1000000.0) << " ms  (" << (decode_setup_time_ns / iteration_count) << " ns/iter)" << std::endl;
    std::cout << "  Extract signals:         " << (extract_signals_time_ns / 1000000.0) << " ms  (" << (extract_signals_time_ns / iteration_count) << " ns/iter)" << std::endl;
    std::cout << "  Other overhead:          " << (timing_overhead_ns / 1000000.0) << " ms  (" << (timing_overhead_ns / iteration_count) << " ns/iter)" << std::endl;
    std::cout << "  Inter-iteration gaps:    " << (gap_time_ns / 1000000.0) << " ms  (" << (gap_time_ns / iteration_count) << " ns/iter)" << std::endl;
    std::cout << "  UNKNOWN:                 " << unaccounted_in_iteration_ms << " ms  (" << (unaccounted_in_iteration_ms * 1000000.0 / iteration_count) << " ns/iter)" << std::endl;
    std::cout << "  ---" << std::endl;
    std::cout << "  Sum of components:       " << sum_of_components_ms << " ms" << std::endl;
    std::cout << "  Total iteration time:    " << measured_iteration_ms << " ms" << std::endl;
    std::cout << "  Loop control overhead:   " << unmeasured_ms << " ms" << std::endl;
    std::cout << "  Total loop time:         " << total_loop_ms << " ms" << std::endl;

    return 0;
}

// BLF.__del__
static void BLF_dealloc(BLFObject* self) {
    if (self->initialized) {
        if (self->messages_data != nullptr) {
            delete self->messages_data;
            self->messages_data = nullptr;
        }
        if (self->dbc_network_cache != nullptr) {
            delete self->dbc_network_cache;
            self->dbc_network_cache = nullptr;
        }
        // networks vector will be automatically destroyed
    }
    Py_TYPE(self)->tp_free((PyObject*)self);
}

// BLF.get_signal(message_name, signal_name) -> numpy array
static PyObject* BLF_get_signal(BLFObject* self, PyObject* args) {
    const char* message_name;
    const char* signal_name;

    if (!self->parsed) {
        PyErr_SetString(PyExc_RuntimeError, "BLF file not loaded");
        return NULL;
    }

    if (!PyArg_ParseTuple(args, "ss", &message_name, &signal_name)) {
        return NULL;
    }

    // Find message
    auto msgIt = self->messages_data->find(message_name);
    if (msgIt == self->messages_data->end()) {
        PyErr_Format(PyExc_KeyError, "Message '%s' not found", message_name);
        return NULL;
    }

    const MessageData& msgData = msgIt->second;

    // Handle "Time" signal specially
    if (strcmp(signal_name, "Time") == 0) {
        npy_intp  dims[1] = {static_cast<npy_intp>(msgData.timestamps.size())};
        PyObject* array   = PyArray_SimpleNewFromData(1, dims, NPY_DOUBLE,
                                                      const_cast<double*>(msgData.timestamps.data()));
        if (array == NULL) {
            return NULL;
        }

        // Set base object to keep BLF object alive
        Py_INCREF(self);
        if (PyArray_SetBaseObject((PyArrayObject*)array, (PyObject*)self) < 0) {
            Py_DECREF(self);
            Py_DECREF(array);
            return NULL;
        }

        return array;
    }

    // Find signal
    auto sigIt = msgData.signals.find(signal_name);
    if (sigIt == msgData.signals.end()) {
        PyErr_Format(PyExc_KeyError, "Signal '%s' not found in message '%s'",
                     signal_name, message_name);
        return NULL;
    }

    const SignalData& sigData = sigIt->second;

    // Create NumPy array wrapping vector data (zero-copy)
    npy_intp  dims[1] = {static_cast<npy_intp>(sigData.values.size())};
    PyObject* array   = PyArray_SimpleNewFromData(1, dims, NPY_DOUBLE,
                                                  const_cast<double*>(sigData.values.data()));
    if (array == NULL) {
        return NULL;
    }

    // Set base object to keep BLF object alive
    Py_INCREF(self);
    if (PyArray_SetBaseObject((PyArrayObject*)array, (PyObject*)self) < 0) {
        Py_DECREF(self);
        Py_DECREF(array);
        return NULL;
    }

    return array;
}

// BLF.messages property
static PyObject* BLF_get_messages(BLFObject* self, void* Py_UNUSED(closure)) {
    if (!self->parsed) {
        PyErr_SetString(PyExc_RuntimeError, "BLF file not loaded");
        return NULL;
    }

    PyObject* list = PyList_New(self->messages_data->size());
    if (list == NULL) {
        return NULL;
    }

    size_t i = 0;
    for (const auto& msgPair : *self->messages_data) {
        PyObject* name = PyUnicode_FromString(msgPair.first.c_str());
        if (name == NULL) {
            Py_DECREF(list);
            return NULL;
        }
        PyList_SET_ITEM(list, i++, name);
    }

    return list;
}

// BLF.get_signals(message_name) -> list
static PyObject* BLF_get_signals(BLFObject* self, PyObject* args) {
    const char* message_name;

    if (!self->parsed) {
        PyErr_SetString(PyExc_RuntimeError, "BLF file not loaded");
        return NULL;
    }

    if (!PyArg_ParseTuple(args, "s", &message_name)) {
        return NULL;
    }

    // Find message
    auto msgIt = self->messages_data->find(message_name);
    if (msgIt == self->messages_data->end()) {
        PyErr_Format(PyExc_KeyError, "Message '%s' not found", message_name);
        return NULL;
    }

    const MessageData& msgData = msgIt->second;

    // Create list with "Time" + all signal names
    PyObject* list = PyList_New(msgData.signals.size() + 1);
    if (list == NULL) {
        return NULL;
    }

    // Add "Time" first
    PyList_SET_ITEM(list, 0, PyUnicode_FromString("Time"));

    // Add all signal names
    size_t i = 1;
    for (const auto& sigPair : msgData.signals) {
        PyObject* name = PyUnicode_FromString(sigPair.first.c_str());
        if (name == NULL) {
            Py_DECREF(list);
            return NULL;
        }
        PyList_SET_ITEM(list, i++, name);
    }

    return list;
}

// BLF.get_message_count(message_name) -> int
static PyObject* BLF_get_message_count(BLFObject* self, PyObject* args) {
    const char* message_name;

    if (!self->parsed) {
        PyErr_SetString(PyExc_RuntimeError, "BLF file not loaded");
        return NULL;
    }

    if (!PyArg_ParseTuple(args, "s", &message_name)) {
        return NULL;
    }

    // Find message
    auto msgIt = self->messages_data->find(message_name);
    if (msgIt == self->messages_data->end()) {
        PyErr_Format(PyExc_KeyError, "Message '%s' not found", message_name);
        return NULL;
    }

    const MessageData& msgData = msgIt->second;
    return PyLong_FromSize_t(msgData.timestamps.size());
}

// Method definitions
static PyMethodDef BLF_methods[] = {
    {       "get_signal",        (PyCFunction)BLF_get_signal, METH_VARARGS,
     "Get signal data by message and signal name as numpy array"                },
    {      "get_signals",       (PyCFunction)BLF_get_signals, METH_VARARGS,
     "Get list of signal names for a message"                                   },
    {"get_message_count", (PyCFunction)BLF_get_message_count, METH_VARARGS,
     "Get number of samples for a message"                                      },
    {               NULL,                               NULL,            0, NULL}
};

// Property definitions
static PyGetSetDef BLF_getsetters[] = {
    {"messages", (getter)BLF_get_messages, NULL,
     "List of all message names", NULL                     },
    {      NULL,                     NULL, NULL, NULL, NULL}
};

// Type definition
static PyTypeObject BLFType = {
    PyVarObject_HEAD_INIT(NULL, 0) "blf_python.BLF",                     /* tp_name */
    .tp_basicsize         = sizeof(BLFObject),                           /* tp_basicsize */
    .tp_itemsize          = 0,                                           /* tp_itemsize */
    .tp_dealloc           = (destructor)BLF_dealloc,                     /* tp_dealloc */
    .tp_vectorcall_offset = 0,                                           /* tp_vectorcall_offset */
    .tp_getattr           = 0,                                           /* tp_getattr */
    .tp_setattr           = 0,                                           /* tp_setattr */
    .tp_as_async          = 0,                                           /* tp_as_async */
    .tp_repr              = 0,                                           /* tp_repr */
    .tp_as_number         = 0,                                           /* tp_as_number */
    .tp_as_sequence       = 0,                                           /* tp_as_sequence */
    .tp_as_mapping        = 0,                                           /* tp_as_mapping */
    .tp_hash              = 0,                                           /* tp_hash */
    .tp_call              = 0,                                           /* tp_call */
    .tp_str               = 0,                                           /* tp_str */
    .tp_getattro          = 0,                                           /* tp_getattro */
    .tp_setattro          = 0,                                           /* tp_setattro */
    .tp_as_buffer         = 0,                                           /* tp_as_buffer */
    .tp_flags             = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,    /* tp_flags */
    .tp_doc               = "BLF file reader with DBC decoding support", /* tp_doc */
    .tp_traverse          = 0,                                           /* tp_traverse */
    .tp_clear             = 0,                                           /* tp_clear */
    .tp_richcompare       = 0,                                           /* tp_richcompare */
    .tp_weaklistoffset    = 0,                                           /* tp_weaklistoffset */
    .tp_iter              = 0,                                           /* tp_iter */
    .tp_iternext          = 0,                                           /* tp_iternext */
    .tp_methods           = BLF_methods,                                 /* tp_methods */
    .tp_members           = 0,                                           /* tp_members */
    .tp_getset            = BLF_getsetters,                              /* tp_getset */
    .tp_base              = 0,                                           /* tp_base */
    .tp_dict              = 0,                                           /* tp_dict */
    .tp_descr_get         = 0,                                           /* tp_descr_get */
    .tp_descr_set         = 0,                                           /* tp_descr_set */
    .tp_dictoffset        = 0,                                           /* tp_dictoffset */
    .tp_init              = (initproc)BLF_init,                          /* tp_init */
    .tp_alloc             = 0,                                           /* tp_alloc */
    .tp_new               = BLF_new,                                     /* tp_new */
    .tp_free              = 0,                                           /* tp_free */
    .tp_is_gc             = 0,                                           /* tp_is_gc */
    .tp_bases             = 0,                                           /* tp_bases */
    .tp_mro               = 0,                                           /* tp_mro */
    .tp_cache             = 0,                                           /* tp_cache */
    .tp_subclasses        = 0,                                           /* tp_subclasses */
    .tp_weaklist          = 0,                                           /* tp_weaklist */
    .tp_del               = 0,                                           /* tp_del */
    .tp_version_tag       = 0,                                           /* tp_version_tag */
    .tp_finalize          = 0,                                           /* tp_finalize */
    .tp_vectorcall        = 0,                                           /* tp_vectorcall */
    .tp_watched           = 0,                                           /* tp_watched */
    .tp_versions_used     = 0,                                           /* tp_versions_used */
};

// Module definition
static PyModuleDef blfmodule = {
    PyModuleDef_HEAD_INIT,
    .m_name     = "blf_python",
    .m_doc      = "BLF file reader and decoder with DBC support",
    .m_size     = -1,
    .m_methods  = NULL,
    .m_slots    = NULL,
    .m_traverse = NULL,
    .m_clear    = NULL,
    .m_free     = NULL,
};

// Module initialization
PyMODINIT_FUNC PyInit_blf_python(void) {
    PyObject* m;

    // Initialize NumPy C API
    import_array();

    if (PyType_Ready(&BLFType) < 0)
        return NULL;

    m = PyModule_Create(&blfmodule);
    if (m == NULL)
        return NULL;

    Py_INCREF(&BLFType);
    if (PyModule_AddObject(m, "BLF", (PyObject*)&BLFType) < 0) {
        Py_DECREF(&BLFType);
        Py_DECREF(m);
        return NULL;
    }

    return m;
}
