/*----------------------------------------------------------------------------
| BLF File Reader - Reads and prints CAN messages from BLF files
|-----------------------------------------------------------------------------
| This program reads BLF (Binary Logging Format) files and prints CAN messages
| including VBLCANMessage, VBLCANMessage2, VBLCANFDMessage, and VBLCANFDMessage64.
|
| Platform: Windows only (MinGW GCC/G++)
 ----------------------------------------------------------------------------*/

#define _CRT_SECURE_NO_WARNINGS

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <windows.h>

#include "binlog.h"

/******************************************************************************
 * Print CAN message to console
 ******************************************************************************/
void print_can_message(const char* msg_type, VBLObjectHeader* header,
                       uint16_t channel, uint32_t id, uint8_t dlc,
                       uint8_t flags, uint8_t* data, uint8_t data_len) {

    // Print message type
    printf("[%s] ", msg_type);

    // Print timestamp in seconds (converted from nanoseconds)
    uint64_t timestamp_ns  = header->mObjectTimeStamp;
    double   timestamp_sec = timestamp_ns / 1e9;
    printf("Time: %.9f s | ", timestamp_sec);

    // Print channel
    printf("Ch: %u | ", channel);

    // Check if extended ID (29-bit) - bit 31 (0x80000000) indicates extended
    if (id & 0x80000000) {
        printf("ID: 0x%08X (ext) | ", id & 0x1FFFFFFF);
    } else {
        printf("ID: 0x%03X (std) | ", id & 0x7FF);
    }

    // Print DLC
    printf("DLC: %u | ", dlc);

    // Print direction (bit 0 of flags)
    uint8_t dir = CAN_MSG_DIR(flags);
    printf("Dir: %s | ", (dir == 0) ? "Rx" : "Tx");

    // Print RTR flag if present
    uint8_t rtr = CAN_MSG_RTR(flags);
    if (rtr) {
        printf("RTR | ");
    }

    // Print data bytes
    printf("Data: [");
    for (uint8_t i = 0; i < data_len; i++) {
        if (i > 0)
            printf(" ");
        printf("%02X", data[i]);
    }
    printf("]\n");
}

/******************************************************************************
 * Print CAN FD 64 message (has different flag structure)
 ******************************************************************************/
void print_canfd64_message(VBLCANFDMessage64* msg) {
    printf("[CAN_FD_64] ");

    // Print timestamp
    uint64_t timestamp_ns  = msg->mHeader.mObjectTimeStamp;
    double   timestamp_sec = timestamp_ns / 1e9;
    printf("Time: %.9f s | ", timestamp_sec);

    // Print channel
    printf("Ch: %u | ", msg->mChannel);

    // Check if extended ID
    if (msg->mID & 0x80000000) {
        printf("ID: 0x%08X (ext) | ", msg->mID & 0x1FFFFFFF);
    } else {
        printf("ID: 0x%03X (std) | ", msg->mID & 0x7FF);
    }

    // Print DLC
    printf("DLC: %u | ", msg->mDLC);

    // Direction from mDir field
    printf("Dir: %s | ", (msg->mDir == 0) ? "Rx" : "Tx");

    // Check for FD flags
    if (msg->mFlags & 0x01) {
        printf("EDL | "); // Extended Data Length (FD)
    }
    if (msg->mFlags & 0x02) {
        printf("BRS | "); // Bit Rate Switch
    }
    if (msg->mFlags & 0x04) {
        printf("ESI | "); // Error State Indicator
    }

    // Print data bytes
    printf("Data: [");
    uint8_t data_len = (msg->mValidDataBytes < 64) ? msg->mValidDataBytes : 64;
    for (uint8_t i = 0; i < data_len; i++) {
        if (i > 0)
            printf(" ");
        printf("%02X", msg->mData[i]);
    }
    printf("]\n");
}

/******************************************************************************
 * Read BLF file and print CAN messages
 ******************************************************************************/
int read_blf_file(const char* filename) {
    BLHANDLE            hFile;
    VBLObjectHeaderBase base;
    VBLFileStatisticsEx statistics = {sizeof(statistics)};
    int32_t             bSuccess;
    uint32_t            msg_count = 0;

    printf("Opening file: %s\n", filename);
    printf("========================================\n\n");

    // Open file for reading
    hFile = BLCreateFile(filename, GENERIC_READ);

    if (BLINVALID_HANDLE_VALUE == hFile) {
        fprintf(stderr, "Error: Could not open file %s\n", filename);
        return -1;
    }

    // Get file statistics
    if (BLGetFileStatisticsEx(hFile, &statistics)) {
        printf("File Statistics:\n");
        printf("  Total Objects: %u\n", statistics.mObjectCount);
        printf("  Application ID: %u\n", statistics.mApplicationID);
        printf("  Application Major: %u\n", statistics.mApplicationMajor);
        printf("  Application Minor: %u\n", statistics.mApplicationMinor);
        printf("  Application Build: %u\n", statistics.mApplicationBuild);
        printf("\n");
    }

    bSuccess = 1;

    // Read all objects from file
    while (bSuccess && BLPeekObject(hFile, &base)) {
        switch (base.mObjectType) {
        case BL_OBJ_TYPE_CAN_MESSAGE: {
            VBLCANMessage message;
            memset(&message, 0, sizeof(VBLCANMessage));
            message.mHeader.mBase = base;
            bSuccess = BLReadObjectSecure(hFile, &message.mHeader.mBase, sizeof(VBLCANMessage));

            if (bSuccess) {
                print_can_message("CAN_MSG", &message.mHeader, message.mChannel,
                                message.mID, message.mDLC, message.mFlags,
                                message.mData, (message.mDLC < 8) ? message.mDLC : 8);
                msg_count++;
                BLFreeObject(hFile, &message.mHeader.mBase);
            }
            break;
        }

        case BL_OBJ_TYPE_CAN_MESSAGE2: {
            VBLCANMessage2 message2;
            memset(&message2, 0, sizeof(VBLCANMessage2));
            message2.mHeader.mBase = base;
            bSuccess = BLReadObjectSecure(hFile, &message2.mHeader.mBase, sizeof(VBLCANMessage2));

            if (bSuccess) {
                print_can_message("CAN_MSG2", &message2.mHeader, message2.mChannel,
                                message2.mID, message2.mDLC, message2.mFlags,
                                message2.mData, (message2.mDLC < 8) ? message2.mDLC : 8);
                msg_count++;
                BLFreeObject(hFile, &message2.mHeader.mBase);
            }
            break;
        }

        case BL_OBJ_TYPE_CAN_FD_MESSAGE: {
            VBLCANFDMessage fdmessage;
            memset(&fdmessage, 0, sizeof(VBLCANFDMessage));
            fdmessage.mHeader.mBase = base;
            bSuccess = BLReadObjectSecure(hFile, &fdmessage.mHeader.mBase, sizeof(VBLCANFDMessage));

            if (bSuccess) {
                uint8_t data_len = (fdmessage.mValidDataBytes < 64) ? fdmessage.mValidDataBytes : 64;
                print_can_message("CAN_FD", &fdmessage.mHeader, fdmessage.mChannel,
                                fdmessage.mID, fdmessage.mDLC, fdmessage.mFlags,
                                fdmessage.mData, data_len);
                msg_count++;
                BLFreeObject(hFile, &fdmessage.mHeader.mBase);
            }
            break;
        }

        case BL_OBJ_TYPE_CAN_FD_MESSAGE_64: {
            VBLCANFDMessage64 fdmessage64;
            memset(&fdmessage64, 0, sizeof(VBLCANFDMessage64));
            fdmessage64.mHeader.mBase = base;
            bSuccess = BLReadObjectSecure(hFile, &fdmessage64.mHeader.mBase, sizeof(VBLCANFDMessage64));

            if (bSuccess) {
                print_canfd64_message(&fdmessage64);
                msg_count++;
                BLFreeObject(hFile, &fdmessage64.mHeader.mBase);
            }
            break;
        }

        default:
            // Skip all other object types
            bSuccess = BLSkipObject(hFile, &base);
            break;
        }
    }

    printf("\n========================================\n");
    printf("Total CAN messages read: %u\n", msg_count);

    // Close file
    if (!BLCloseHandle(hFile)) {
        fprintf(stderr, "Error: Could not close file\n");
        return -1;
    }

    return 0;
}

/******************************************************************************
 * Main entry point
 ******************************************************************************/
int main(int argc, char* argv[]) {
    if (argc < 2) {
        printf("Usage: %s <blf_file>\n", argv[0]);
        printf("\nExample:\n");
        printf("  %s recording.blf\n", argv[0]);
        return 1;
    }

    const char* filename = argv[1];

    int result = read_blf_file(filename);

    if (result != 0) {
        fprintf(stderr, "Failed to read BLF file\n");
        return 1;
    }

    return 0;
}
