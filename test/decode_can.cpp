/*----------------------------------------------------------------------------
| CAN Message Decoder - Decodes CAN messages using DBC database
|-----------------------------------------------------------------------------
| This program takes a CAN ID, data bytes (in hex), and a DBC file, then
| decodes the message and prints all signal names and values.
|
| Usage: decode_can <CAN_ID> <DATA_HEX> <DBC_FILE>
| Example: decode_can 0x60B 4337690195 4B6A01 example/IMU.dbc
|
| Platform: Windows only (MinGW GCC/G++)
 ----------------------------------------------------------------------------*/

#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <string>
#include <cstring>
#include <cstdint>
#include <vector>

#include <Vector/DBC.h>

/**
 * Parse hex string to bytes
 * Accepts formats like "43 37 69 01" or "43376901"
 */
std::vector<uint8_t> parseHexData(const std::string& hexStr) {
    std::vector<uint8_t> data;
    std::string cleanHex;

    // Remove spaces and other non-hex characters
    for (char c : hexStr) {
        if ((c >= '0' && c <= '9') || (c >= 'A' && c <= 'F') || (c >= 'a' && c <= 'f')) {
            cleanHex += c;
        }
    }

    // Parse pairs of hex digits
    for (size_t i = 0; i + 1 < cleanHex.length(); i += 2) {
        std::string byteStr = cleanHex.substr(i, 2);
        uint8_t byte = static_cast<uint8_t>(std::stoul(byteStr, nullptr, 16));
        data.push_back(byte);
    }

    return data;
}

/**
 * Parse CAN ID (supports 0x prefix and decimal)
 */
uint32_t parseCanId(const std::string& idStr) {
    if (idStr.substr(0, 2) == "0x" || idStr.substr(0, 2) == "0X") {
        return std::stoul(idStr, nullptr, 16);
    } else {
        return std::stoul(idStr, nullptr, 10);
    }
}

/**
 * Extract raw value from CAN data bytes
 */
uint64_t extractRawValue(const std::vector<uint8_t>& data, const Vector::DBC::Signal& signal) {
    uint64_t rawValue = 0;

    if (signal.byteOrder == Vector::DBC::ByteOrder::BigEndian) {
        // Motorola (Big Endian) byte order
        uint32_t bitPos = signal.startBit;
        for (uint32_t i = 0; i < signal.bitSize; ++i) {
            uint32_t byteIdx = bitPos / 8;
            uint32_t bitIdx = 7 - (bitPos % 8);

            if (byteIdx < data.size()) {
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
            uint32_t bitIdx = bitPos % 8;

            if (byteIdx < data.size()) {
                uint64_t bit = (data[byteIdx] >> bitIdx) & 0x01;
                rawValue |= (bit << i);
            }

            bitPos++;
        }
    }

    // Handle signed values
    if (signal.valueType == Vector::DBC::ValueType::Signed) {
        // Check if the sign bit is set
        uint64_t signBit = 1ULL << (signal.bitSize - 1);
        if (rawValue & signBit) {
            // Sign extend
            uint64_t mask = (1ULL << signal.bitSize) - 1;
            rawValue |= ~mask;
        }
    }

    return rawValue;
}

/**
 * Decode CAN message using DBC database
 */
int decodeCanMessage(uint32_t canId, const std::vector<uint8_t>& data, const std::string& dbcFile) {
    // Load DBC file
    Vector::DBC::Network network;
    std::ifstream ifs(dbcFile);

    if (!ifs.is_open()) {
        std::cerr << "Error: Could not open DBC file: " << dbcFile << std::endl;
        return -1;
    }

    ifs >> network;
    ifs.close();

    if (!network.successfullyParsed) {
        std::cerr << "Error: Failed to parse DBC file" << std::endl;
        return -1;
    }

    // Find the message with matching ID
    auto msgIt = network.messages.find(canId);
    if (msgIt == network.messages.end()) {
        std::cerr << "Error: Message with ID 0x" << std::hex << canId
                  << " not found in DBC file" << std::endl;
        return -1;
    }

    const Vector::DBC::Message& message = msgIt->second;

    // Print message info
    std::cout << "========================================" << std::endl;
    std::cout << "Message: " << message.name << std::endl;
    std::cout << "ID: 0x" << std::hex << std::uppercase << canId << std::dec << std::endl;
    std::cout << "DLC: " << data.size() << std::endl;
    std::cout << "Data: [";
    for (size_t i = 0; i < data.size(); ++i) {
        if (i > 0) std::cout << " ";
        std::cout << std::hex << std::uppercase << std::setw(2) << std::setfill('0')
                  << static_cast<int>(data[i]);
    }
    std::cout << "]" << std::dec << std::endl;
    std::cout << "========================================" << std::endl;

    // Decode each signal
    if (message.signals.empty()) {
        std::cout << "No signals defined for this message" << std::endl;
        return 0;
    }

    std::cout << "\nSignals:" << std::endl;
    for (const auto& sigPair : message.signals) {
        const Vector::DBC::Signal& signal = sigPair.second;

        // Extract raw value from data
        uint64_t rawValue = extractRawValue(data, signal);

        // Convert to physical value
        double physicalValue;
        if (signal.valueType == Vector::DBC::ValueType::Signed) {
            int64_t signedRaw = static_cast<int64_t>(rawValue);
            physicalValue = signal.rawToPhysicalValue(static_cast<double>(signedRaw));
        } else {
            physicalValue = signal.rawToPhysicalValue(static_cast<double>(rawValue));
        }

        // Print signal information
        std::cout << "  " << signal.name << ": ";
        std::cout << physicalValue;

        if (!signal.unit.empty()) {
            std::cout << " " << signal.unit;
        }

        // Check for value descriptions (enumerated values)
        auto valDescIt = signal.valueDescriptions.find(static_cast<int32_t>(rawValue));
        if (valDescIt != signal.valueDescriptions.end()) {
            std::cout << " (" << valDescIt->second << ")";
        }

        std::cout << " [raw: " << rawValue << "]" << std::endl;
    }

    return 0;
}

/**
 * Main entry point
 */
int main(int argc, char* argv[]) {
    if (argc < 4) {
        std::cout << "Usage: " << argv[0] << " <CAN_ID> <DATA_HEX> <DBC_FILE>" << std::endl;
        std::cout << "\nExamples:" << std::endl;
        std::cout << "  " << argv[0] << " 0x60B \"43 37 69 01 95 4B 6A 01\" example/IMU.dbc" << std::endl;
        std::cout << "  " << argv[0] << " 1547 4337690195 example/IMU.dbc" << std::endl;
        std::cout << "\nCAN_ID: Can be in hex (0x...) or decimal format" << std::endl;
        std::cout << "DATA_HEX: Hex bytes, spaces optional" << std::endl;
        std::cout << "DBC_FILE: Path to DBC database file" << std::endl;
        return 1;
    }

    try {
        // Parse command line arguments
        uint32_t canId = parseCanId(argv[1]);
        std::vector<uint8_t> data = parseHexData(argv[2]);
        std::string dbcFile = argv[3];

        if (data.empty()) {
            std::cerr << "Error: No valid hex data provided" << std::endl;
            return 1;
        }

        if (data.size() > 64) {
            std::cerr << "Error: Data too long (max 64 bytes for CAN FD)" << std::endl;
            return 1;
        }

        // Decode the message
        int result = decodeCanMessage(canId, data, dbcFile);

        if (result != 0) {
            return 1;
        }

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
