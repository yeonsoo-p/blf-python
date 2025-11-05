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

#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <string>
#include <vector>
#include <windows.h>


#include "binlog.h"
#include <Vector/DBC.h>

// Structure to hold signal data during collection
struct SignalData {
    std::string         name;
    std::vector<double> values;
    std::vector<double> timestamps;
};

// Structure to hold message data
struct MessageData {
    std::string                       name;
    uint32_t                          id;
    std::map<std::string, SignalData> signals;
    std::vector<double>               timestamps;
};

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
        uint64_t signBit = 1ULL << (signal.bitSize - 1);
        if (rawValue & signBit) {
            uint64_t mask = (1ULL << signal.bitSize) - 1;
            rawValue |= ~mask;
        }
    }

    return rawValue;
}

// Read BLF file and decode messages
static PyObject* read_blf_file(PyObject* Py_UNUSED(self), PyObject* args, PyObject* kwargs) {
    const char* blf_filepath;
    PyObject*   dbc_filepaths_list;
    int         channel = -1; // -1 means all channels

    static char* kwlist[] = {(char*)"blf_filepath", (char*)"dbc_filepaths", (char*)"channel", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "sO|i", kwlist,
                                     &blf_filepath, &dbc_filepaths_list, &channel)) {
        return NULL;
    }

    // Convert DBC filepaths list to C++ vector
    if (!PyList_Check(dbc_filepaths_list)) {
        PyErr_SetString(PyExc_TypeError, "dbc_filepaths must be a list");
        return NULL;
    }

    std::vector<std::string> dbcFiles;
    Py_ssize_t               numDbcFiles = PyList_Size(dbc_filepaths_list);

    for (Py_ssize_t i = 0; i < numDbcFiles; ++i) {
        PyObject* item = PyList_GetItem(dbc_filepaths_list, i);
        if (!PyUnicode_Check(item)) {
            PyErr_SetString(PyExc_TypeError, "All DBC filepaths must be strings");
            return NULL;
        }
        dbcFiles.push_back(PyUnicode_AsUTF8(item));
    }

    if (dbcFiles.empty()) {
        PyErr_SetString(PyExc_ValueError, "At least one DBC file must be provided");
        return NULL;
    }

    // Load all DBC files
    std::vector<Vector::DBC::Network> networks;
    for (const auto& dbcFile : dbcFiles) {
        Vector::DBC::Network network;
        std::ifstream        ifs(dbcFile);

        if (!ifs.is_open()) {
            PyErr_Format(PyExc_IOError, "Could not open DBC file: %s", dbcFile.c_str());
            return NULL;
        }

        ifs >> network;
        ifs.close();

        if (!network.successfullyParsed) {
            PyErr_Format(PyExc_ValueError, "Failed to parse DBC file: %s", dbcFile.c_str());
            return NULL;
        }

        networks.push_back(std::move(network));
    }

    // Open BLF file
    BLHANDLE hFile = BLCreateFile(blf_filepath, GENERIC_READ);

    if (BLINVALID_HANDLE_VALUE == hFile) {
        PyErr_Format(PyExc_IOError, "Could not open BLF file: %s", blf_filepath);
        return NULL;
    }

    // Storage for messages
    std::map<std::string, MessageData> messagesData;

    // Read all objects from BLF file
    VBLObjectHeaderBase base;
    int32_t             bSuccess = 1;

    while (bSuccess && BLPeekObject(hFile, &base)) {
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

        // Process valid CAN message
        if (validMessage) {
            // Filter by channel if specified
            if (channel >= 0 && msgChannel != static_cast<uint16_t>(channel)) {
                continue;
            }

            // Find matching message in DBC files
            const Vector::DBC::Message* dbcMessage = nullptr;
            for (const auto& network : networks) {
                auto msgIt = network.messages.find(msgId);
                if (msgIt != network.messages.end()) {
                    dbcMessage = &msgIt->second;
                    break;
                }
            }

            if (dbcMessage) {
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

                // Decode all signals
                for (const auto& sigPair : dbcMessage->signals) {
                    const Vector::DBC::Signal& signal = sigPair.second;

                    // Extract raw value
                    uint64_t rawValue = extractRawValue(msgData, msgDlc, signal);

                    // Convert to physical value
                    double physicalValue;
                    if (signal.valueType == Vector::DBC::ValueType::Signed) {
                        int64_t signedRaw = static_cast<int64_t>(rawValue);
                        physicalValue     = signal.rawToPhysicalValue(static_cast<double>(signedRaw));
                    } else {
                        physicalValue = signal.rawToPhysicalValue(static_cast<double>(rawValue));
                    }

                    // Store signal value
                    msgData_storage.signals[signal.name].values.push_back(physicalValue);
                }
            }
        }
    }

    BLCloseHandle(hFile);

    // Create Python dictionary to return
    PyObject* result = PyDict_New();

    for (const auto& msgPair : messagesData) {
        const MessageData& msgData = msgPair.second;

        // Create dictionary for this message
        PyObject* msgDict = PyDict_New();

        // Add "Time" signal (timestamps)
        npy_intp  timeDims[1] = {static_cast<npy_intp>(msgData.timestamps.size())};
        PyObject* timeArray   = PyArray_SimpleNew(1, timeDims, NPY_DOUBLE);
        double*   timeData    = (double*)PyArray_DATA((PyArrayObject*)timeArray);
        std::memcpy(timeData, msgData.timestamps.data(), msgData.timestamps.size() * sizeof(double));
        PyDict_SetItemString(msgDict, "Time", timeArray);
        Py_DECREF(timeArray);

        // Add all signal arrays
        for (const auto& sigPair : msgData.signals) {
            const SignalData& sigData = sigPair.second;

            npy_intp  dims[1]   = {static_cast<npy_intp>(sigData.values.size())};
            PyObject* array     = PyArray_SimpleNew(1, dims, NPY_DOUBLE);
            double*   arrayData = (double*)PyArray_DATA((PyArrayObject*)array);
            std::memcpy(arrayData, sigData.values.data(), sigData.values.size() * sizeof(double));

            PyDict_SetItemString(msgDict, sigPair.first.c_str(), array);
            Py_DECREF(array);
        }

        PyDict_SetItemString(result, msgPair.first.c_str(), msgDict);
        Py_DECREF(msgDict);
    }

    return result;
}

// Module methods
static PyMethodDef BlfMethods[] = {
    {"read_blf",
     (PyCFunction)read_blf_file,
     METH_VARARGS | METH_KEYWORDS,
     "Read and decode BLF file.\n\n"
     "Args:\n"
     "    blf_filepath (str): Path to BLF file\n"
     "    dbc_filepaths (list): List of DBC file paths\n"
     "    channel (int, optional): CAN channel filter (-1 for all channels)\n\n"
     "Returns:\n"
     "    dict: Dictionary of messages, each containing signal numpy arrays\n" },
    {NULL, NULL, 0, NULL}
};

// Module definition
static PyModuleDef blfmodule = {
    PyModuleDef_HEAD_INIT,
    .m_name = "blf_python",
    .m_doc = "BLF file reader and decoder with DBC support",
    .m_size = -1,
    .m_methods = BlfMethods,
    .m_slots = NULL,
    .m_traverse = NULL,
    .m_clear = NULL,
    .m_free = NULL,
};

// Module initialization
PyMODINIT_FUNC PyInit_blf_python(void) {
    import_array(); // Initialize NumPy C API
    return PyModule_Create(&blfmodule);
}
